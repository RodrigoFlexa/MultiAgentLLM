"""
Protocolo — FoA com decomposição em DAG (foa_dag).

Variante do Federation of Agents (Giusti et al., 2025, arXiv:2509.20175) que
exercita as fases de DECOMPOSIÇÃO e DAG do paper, ausentes no `foa` (cluster).

Regras desta versão (definidas para o nosso estudo):
  * A DECOMPOSIÇÃO é proposta pelos TRABALHADORES: cada um dos N SLMs propõe um
    plano de subtarefas.
  * O MESTRE (Agent-0) funde as N propostas num ÚNICO plano de consenso com
    EXATAMENTE N subtarefas, formando um DAG (subtarefas + dependências).
  * Nº de subtarefas == nº de agentes: 1 agente resolve 1 subtarefa (agente i ↔
    subtarefa i). NÃO há atribuição por capacidade (frota homogênea) nem
    refinamento em cluster (cada agente trabalha sozinho no seu pedaço).
  * O MESTRE orquestra a execução pela ordem topológica do DAG (propagando os
    resultados dos predecessores) e faz a SÍNTESE final.

Grafo:  START → propose → merge → solve → synth → END
        (N SLMs)  (mestre)  (N SLMs)  (mestre)

Custo por pergunta: N (propor) + 1 (fundir/DAG) + N (resolver) + 1 (sintetizar)
= 2N + 2 chamadas. O custo POR AGENTE é registrado (quem trabalhou mais).
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, usage_delta,
)


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
def _propose_msgs(question: str, n: int) -> list[dict]:
    system = (
        "You are a planner. Break the problem into a plan of EXACTLY "
        f"{n} subtasks that together solve it. Each subtask has a short "
        "instruction and lists the ids of the earlier subtasks it depends on. "
        "Output JSON ONLY, no prose, in the form: "
        '[{"id":0,"task":"...","deps":[]},{"id":1,"task":"...","deps":[0]}]'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Problem:\n{question}"}]


def _merge_msgs(question: str, proposals: list[str], n: int) -> list[dict]:
    bloco = "\n\n".join(f"[Worker {i+1} plan]\n{p}" for i, p in enumerate(proposals))
    system = (
        "You are the orchestrator (Agent-0). Several workers proposed plans to "
        f"decompose the same problem. Merge them into a SINGLE consensus plan "
        f"with EXACTLY {n} subtasks forming a DAG. Output JSON ONLY: "
        '[{"id":0,"task":"...","deps":[]}, ...]'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Problem:\n{question}\n\nProposals:\n{bloco}"}]


def _subtask_msgs(question: str, sub: dict, dep_text: str) -> list[dict]:
    system = (
        "You solve ONE subtask of a larger problem. Use the prerequisite "
        "results if given. Reply with the result of THIS subtask only (a short "
        "value and a one-line justification)."
    )
    user = (f"Original problem:\n{question}\n\nYour subtask:\n{sub['task']}")
    if dep_text:
        user += f"\n\nResults of prerequisite subtasks:\n{dep_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _synth_msgs(question: str, plan: list[dict], results: dict) -> list[dict]:
    bloco = "\n".join(
        f"[Subtask {s['id']}] {s['task']}\n→ {results.get(s['id'], '(no result)')}"
        for s in plan
    )
    system = (
        "You are the orchestrator (Agent-0). Combine the subtask results into "
        "the final answer to the original problem. Think briefly and ALWAYS end "
        "with a line '#### <final number>'."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Problem:\n{question}\n\nSubtask results:\n{bloco}"}]


# ──────────────────────────────────────────────────────────────────────
# DAG: parsing robusto + ordenação topológica
# ──────────────────────────────────────────────────────────────────────
def _topo(plan: list[dict]) -> list[int] | None:
    """Ordem topológica (Kahn). Devolve None se houver ciclo."""
    deps = {p["id"]: set(p["deps"]) for p in plan}
    order, ready = [], [i for i in deps if not deps[i]]
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in deps:
            if i in deps[j]:
                deps[j].discard(i)
                if not deps[j] and j not in order and j not in ready:
                    ready.append(j)
    return order if len(order) == len(plan) else None


def _parse_plan(text: str, n: int, question: str) -> list[dict]:
    """Extrai o plano em JSON da saída do mestre; cai em uma cadeia linear de N
    subtarefas se a parsing falhar. Sempre devolve EXATAMENTE N subtarefas com
    ids 0..N-1 e dependências válidas e acíclicas."""
    plan: list[dict] | None = None
    try:
        a, b = text.index("["), text.rindex("]") + 1
        data = json.loads(text[a:b])
        if isinstance(data, list) and data:
            plan = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                task = (item.get("task") or item.get("subtask")
                        or item.get("description") or "Subtask")
                deps = item.get("deps") or item.get("dependencies") or []
                deps = [d for d in deps if isinstance(d, int)] if isinstance(deps, list) else []
                plan.append({"task": str(task), "deps": deps})
    except Exception:
        plan = None

    if not plan:
        plan = [{"task": f"Step {i+1} toward solving the problem",
                 "deps": ([i - 1] if i > 0 else [])} for i in range(n)]

    # exatamente N subtarefas
    plan = plan[:n]
    while len(plan) < n:
        i = len(plan)
        plan.append({"task": f"Step {i+1}", "deps": ([i - 1] if i > 0 else [])})

    # reindexa ids 0..n-1 e saneia dependências
    for i, p in enumerate(plan):
        p["id"] = i
    ids = set(range(n))
    for p in plan:
        p["deps"] = [d for d in p["deps"] if d in ids and d != p["id"]]

    # garante aciclicidade; se houver ciclo, vira cadeia linear
    if _topo(plan) is None:
        for p in plan:
            p["deps"] = [p["id"] - 1] if p["id"] > 0 else []
    return plan


# ──────────────────────────────────────────────────────────────────────
# Estado
# ──────────────────────────────────────────────────────────────────────
class State(UsageState, total=False):
    question: str
    proposals: list[str]
    plan: list
    subtask_results: dict
    answer: str
    calls: Annotated[list, operator.add]   # 1 registro por chamada (p/ custo por agente)


def _record(gen, agent, phase) -> dict:
    return {"agent": agent, "phase": phase,
            "completion_tokens": gen.completion_tokens,
            "prompt_tokens": gen.prompt_tokens,
            "compute_cost": gen.compute_cost,
            "latency_s": gen.latency_s}


@register
class FoADag(Protocol):
    name = "foa_dag"

    def build_graph(self):
        self.n_agents = self.hub.n_minions

        g = StateGraph(State)
        g.add_node("propose", self._propose)
        g.add_node("merge", self._merge)
        g.add_node("solve", self._solve)
        g.add_node("synth", self._synth)
        g.add_edge(START, "propose")
        g.add_edge("propose", "merge")
        g.add_edge("merge", "solve")
        g.add_edge("solve", "synth")
        g.add_edge("synth", END)
        return g.compile()

    # 1) trabalhadores propõem decomposições
    def _propose(self, state: State) -> dict[str, Any]:
        q = state["question"]
        gens = [self.hub.minion_agent().chat(_propose_msgs(q, self.n_agents),
                                             gen_cfg=cfg.CREATIVE)
                for _ in range(self.n_agents)]
        return {
            "proposals": [g.text for g in gens],
            "calls": [_record(g, i, "propose") for i, g in enumerate(gens)],
            **combine_usage(gens, is_master=False),
        }

    # 2) mestre funde as propostas num DAG de N subtarefas
    def _merge(self, state: State) -> dict[str, Any]:
        q = state["question"]
        gen = self.hub.master.chat(_merge_msgs(q, state["proposals"], self.n_agents))
        plan = _parse_plan(gen.text, self.n_agents, q)
        return {"plan": plan, "calls": [_record(gen, "master", "merge")],
                **usage_delta(gen, is_master=True)}

    # 3) orquestra a execução pelo DAG (1 agente por subtarefa)
    def _solve(self, state: State) -> dict[str, Any]:
        q = state["question"]
        plan = state["plan"]
        by_id = {p["id"]: p for p in plan}
        order = _topo(plan) or [p["id"] for p in plan]

        results: dict[int, str] = {}
        gens, calls = [], []
        for sid in order:
            sub = by_id[sid]
            dep_text = "\n".join(
                f"[Subtask {d}] {results.get(d, '')}" for d in sub["deps"]
            )
            gen = self.hub.minion_agent().chat(_subtask_msgs(q, sub, dep_text))
            results[sid] = gen.text
            gens.append(gen)
            calls.append(_record(gen, sid, "solve"))   # agente i ↔ subtarefa i
        return {"subtask_results": results, "calls": calls,
                **combine_usage(gens, is_master=False)}

    # 4) mestre sintetiza a resposta final
    def _synth(self, state: State) -> dict[str, Any]:
        q = state["question"]
        gen = self.hub.master.chat(
            _synth_msgs(q, state["plan"], state.get("subtask_results", {}))
        )
        return {"answer": gen.text, "calls": [_record(gen, "master", "synth")],
                **usage_delta(gen, is_master=True)}

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "proposals": [], "plan": [],
                "subtask_results": {}, "calls": [], **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        calls = final_state.get("calls", [])
        # custo POR AGENTE (só os SLMs 0..N-1): soma das fases propose + solve
        agent_costs = []
        for i in range(self.n_agents):
            cs = [c for c in calls if c["agent"] == i]
            agent_costs.append({
                "agent": i,
                "subtask": i,
                "n_calls": len(cs),
                "completion_tokens": sum(c["completion_tokens"] for c in cs),
                "prompt_tokens": sum(c["prompt_tokens"] for c in cs),
                "compute_cost": sum(c["compute_cost"] for c in cs),
                "latency_s": sum(c["latency_s"] for c in cs),
            })
        plan = final_state.get("plan", [])
        return final_state.get("answer", ""), {
            "plan": plan,
            "subtask_results": {str(k): v for k, v in
                                final_state.get("subtask_results", {}).items()},
            "agent_costs": agent_costs,
            "master_calls": [c for c in calls if c["agent"] == "master"],
        }
