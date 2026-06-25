"""
Protocolo — FoA com decomposição dinâmica (foa_dag).

Variante do Federation of Agents (Giusti et al., 2025) que exercita a
decomposição do paper:

  1. DECOMPOSIÇÃO (mestre/Agent-0): decide quebrar OU NÃO o problema. Simples →
     1 subtarefa; complexo → uma sequência curta de subtarefas (até
     `cfg.DAG.max_subtasks`). Cada subtarefa é atribuída a um worker (reuso
     permitido) e marcada ou não como COMPLEXA.
  2. EXECUÇÃO em SEQUÊNCIA, com SÍNTESE PROGRESSIVA: cada subtarefa é resolvida
     e o mestre vai integrando o resultado numa "solução corrente". Uma subtarefa
     complexa é resolvida por um CLUSTER (mecanismo reflexivo de `cluster.py`);
     uma simples é resolvida por um único worker.
  3. A solução corrente após a última subtarefa é a resposta final.

Grafo:  START → decompose → execute → END
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, empty_usage, register, usage_delta,
)
from src.protocols.cluster import agent_costs_from_calls, run_cluster

_USAGE_KEYS = ("latency_s", "total_tokens", "master_tokens", "minion_tokens",
               "compute_cost", "n_model_calls")


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
def _decompose_msgs(question: str, n: int, maxs: int) -> list[dict]:
    system = (
        "You are the orchestrator (Agent-0) of a team of workers. Decide how to "
        "solve the problem.\n"
        "- If it is SIMPLE, use a SINGLE subtask.\n"
        f"- If it is COMPLEX, break it into a SHORT ordered sequence of at most "
        f"{maxs} subtasks. Each subtask must be a concrete, self-contained step "
        "whose result later steps can build on.\n"
        f"Assign each subtask to one worker (worker number 1..{n}; you MAY reuse "
        "a worker). Set complex=true ONLY for a subtask hard enough to deserve a "
        "small team (a cluster); otherwise complex=false.\n"
        'Output JSON ONLY: {"subtasks":[{"id":0,"task":"...","worker":1,'
        '"complex":false}]}'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content":
                f"Problem:\n{question}\n\nAvailable workers: {n}."}]


def _subtask_msgs(question: str, task: str, running: str) -> list[dict]:
    system = (
        "You are a worker solving ONE subtask of a larger problem. Use the "
        "solution so far if relevant. Solve only your subtask, show your work, "
        "and state its result clearly (end with '#### <number>' if it yields a "
        "number)."
    )
    user = f"Original problem:\n{question}\n\nYour subtask:\n{task}"
    if running:
        user += f"\n\nSolution so far:\n{running}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _synth_msgs(question: str, running: str, sub: dict, result: str,
                is_last: bool) -> list[dict]:
    system = (
        "You are the orchestrator integrating subtask results into a running "
        "solution. Given the solution so far and the newest subtask result, "
        "update the running solution. If the ORIGINAL problem is now fully "
        "solved, give the final answer ending with a line '#### <final number>'. "
        "Otherwise give the updated partial solution."
    )
    if is_last:
        system += (" This is the LAST subtask, so you MUST end with "
                   "'#### <final number>'.")
    user = (
        f"Original problem:\n{question}\n\n"
        f"Solution so far:\n{running or '(none yet)'}\n\n"
        f"Newest subtask (#{sub['id']}: {sub['task']}) result:\n{result}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ──────────────────────────────────────────────────────────────────────
# Parsing do plano
# ──────────────────────────────────────────────────────────────────────
def _parse_plan(text: str, n: int, maxs: int, question: str) -> list[dict]:
    plan = None
    try:
        a, b = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[a:b])
        items = data.get("subtasks") if isinstance(data, dict) else data
        if isinstance(items, list) and items:
            plan = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                plan.append({
                    "task": str(it.get("task") or it.get("subtask") or "Solve the problem"),
                    "worker": int(it.get("worker", 1)) if str(it.get("worker", 1)).isdigit() else 1,
                    "complex": bool(it.get("complex", False)),
                })
    except Exception:
        plan = None

    if not plan:
        # Fallback determinístico: 1 subtarefa (simples). Se houver >=2 workers,
        # cria um 2º passo complexo — útil para exercitar o ramo de cluster.
        if n >= 2:
            plan = [
                {"task": "Set up and compute the first part of the problem.",
                 "worker": 1, "complex": False},
                {"task": "Finish solving the problem using the previous result.",
                 "worker": 2, "complex": True},
            ]
        else:
            plan = [{"task": "Solve the problem.", "worker": 1, "complex": False}]

    plan = plan[:maxs]
    for i, p in enumerate(plan):
        p["id"] = i
        p["worker"] = min(max(1, p["worker"]), n)
    return plan


# ──────────────────────────────────────────────────────────────────────
# Estado
# ──────────────────────────────────────────────────────────────────────
class State(UsageState, total=False):
    question: str
    plan: list
    subtask_results: dict
    running: str
    answer: str
    calls: Annotated[list, operator.add]


@register
class FoADag(Protocol):
    name = "foa_dag"

    def build_graph(self):
        self.n_agents = self.hub.n_minions
        g = StateGraph(State)
        g.add_node("decompose", self._decompose)
        g.add_node("execute", self._execute)
        g.add_edge(START, "decompose")
        g.add_edge("decompose", "execute")
        g.add_edge("execute", END)
        return g.compile()

    def _decompose(self, state: State) -> dict[str, Any]:
        q = state["question"]
        gen = self.hub.master.chat(_decompose_msgs(q, self.n_agents, cfg.DAG.max_subtasks))
        plan = _parse_plan(gen.text, self.n_agents, cfg.DAG.max_subtasks, q)
        return {"plan": plan,
                "calls": [self._rec(gen, "master", "decompose")],
                **usage_delta(gen, is_master=True)}

    def _execute(self, state: State) -> dict[str, Any]:
        q = state["question"]
        plan = state["plan"]
        usage = empty_usage()
        calls: list[dict] = []
        results: dict[int, str] = {}
        running = ""

        def add(delta):
            for k in _USAGE_KEYS:
                usage[k] += delta.get(k, 0)
            usage["used_master"] = usage["used_master"] or delta.get("used_master", False)

        for idx, sub in enumerate(plan):
            is_last = idx == len(plan) - 1
            if sub["complex"]:
                # subtarefa difícil → cluster reflexivo (FoA padrão)
                ctx = (f"Original problem: {q}\nSolution so far: {running}"
                       if running else f"Original problem: {q}")
                cr = run_cluster(self.hub, sub["task"], context=ctx)
                result = cr.answer
                add(cr.usage)
                calls.extend(cr.calls)
            else:
                # subtarefa simples → 1 worker
                gen = self.hub.minion_agent().chat(_subtask_msgs(q, sub["task"], running))
                result = gen.text
                add(usage_delta(gen, is_master=False))
                calls.append(self._rec(gen, sub["worker"] - 1, "subtask"))  # 0-based
            results[sub["id"]] = result

            # síntese progressiva (mestre)
            gen = self.hub.master.chat(_synth_msgs(q, running, sub, result, is_last))
            running = gen.text
            add(usage_delta(gen, is_master=True))
            calls.append(self._rec(gen, "master", "synth"))

        return {"subtask_results": {str(k): v for k, v in results.items()},
                "running": running, "answer": running, "calls": calls, **usage}

    @staticmethod
    def _rec(gen, agent, phase) -> dict:
        return {"agent": agent, "phase": phase,
                "completion_tokens": gen.completion_tokens,
                "prompt_tokens": gen.prompt_tokens,
                "compute_cost": gen.compute_cost,
                "latency_s": gen.latency_s}

    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "plan": [], "subtask_results": {},
                "running": "", "calls": [], **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        calls = final_state.get("calls", [])
        plan = final_state.get("plan", [])
        return final_state.get("answer", ""), {
            "n_subtasks": len(plan),
            "plan": plan,
            "subtask_results": final_state.get("subtask_results", {}),
            "agent_costs": agent_costs_from_calls(calls),
            "master_calls": [c for c in calls if c["agent"] == "master"],
        }
