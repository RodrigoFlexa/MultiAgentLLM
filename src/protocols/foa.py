"""
Protocolo — Federation of Agents (FoA), forma de RESOLVER (cluster colaborativo).

Giusti et al., "Federation of Agents" (2025, arXiv:2509.20175).

Sem decomposição/DAG (isso é o `foa_dag`). Aqui o problema é resolvido por UM
cluster colaborativo com o mecanismo reflexivo (ver `cluster.py`): a frota
rascunha, critica/pontua os pares com honestidade e re-resolve usando os
insights; sem consenso, vence a maior nota acumulada. Por fim, o ORQUESTRADOR
(o mestre) sintetiza a resposta final.

Grafo:  START → cluster → synth → END
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, empty_usage, register, usage_delta,
)
from src.protocols.cluster import agent_costs_from_calls, run_cluster

_SYNTH_SYSTEM = (
    "You are the orchestrator (Agent-0). Your team produced and critically "
    "refined solutions to the problem. Read them, weigh the more reliable "
    "reasoning (peer scores are provided as a hint), resolve any disagreement, "
    "and give the single correct final answer. Think step by step and ALWAYS "
    "end with a line '#### <final number>'."
)


class State(UsageState, total=False):
    question: str
    agent_answers: list
    scores: list
    converged: bool
    rounds: int
    answer: str
    calls: Annotated[list, operator.add]


@register
class FoA(Protocol):
    name = "foa"

    def build_graph(self):
        g = StateGraph(State)
        g.add_node("cluster", self._cluster)
        g.add_node("synth", self._synth)
        g.add_edge(START, "cluster")
        g.add_edge("cluster", "synth")
        g.add_edge("synth", END)
        return g.compile()

    def _cluster(self, state: State) -> dict[str, Any]:
        r = run_cluster(self.hub, state["question"])
        return {"agent_answers": r.agent_answers, "scores": r.scores,
                "converged": r.converged, "rounds": r.rounds,
                "calls": r.calls, **r.usage}

    def _synth(self, state: State) -> dict[str, Any]:
        answers = state.get("agent_answers", [])
        scores = state.get("scores", [])
        bloco = "\n\n".join(
            f"[Agent {i+1}] (peer score {scores[i]:.0f})\n{a}" if i < len(scores)
            else f"[Agent {i+1}]\n{a}" for i, a in enumerate(answers)
        )
        consenso = "The team reached consensus." if state.get("converged") else \
            "The team did NOT fully converge."
        msgs = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content":
                f"Problem:\n{state['question']}\n\n{consenso}\n\n"
                f"Team's final solutions:\n{bloco}"},
        ]
        gen = self.hub.master.chat(msgs)
        return {"answer": gen.text,
                "calls": [{"agent": "master", "phase": "synth",
                           "completion_tokens": gen.completion_tokens,
                           "prompt_tokens": gen.prompt_tokens,
                           "compute_cost": gen.compute_cost,
                           "latency_s": gen.latency_s}],
                **usage_delta(gen, is_master=True)}

    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "calls": [], **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        calls = final_state.get("calls", [])
        return final_state.get("answer", ""), {
            "rounds": final_state.get("rounds", 0),
            "converged": final_state.get("converged", False),
            "cluster_scores": final_state.get("scores", []),
            "agent_answers": final_state.get("agent_answers", []),
            "agent_costs": agent_costs_from_calls(calls),
        }
