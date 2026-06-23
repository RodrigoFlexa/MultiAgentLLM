"""
Protocolo — Mixture-of-Agents (MoA).

Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities"
(2024, arXiv:2406.04692).

Camada de "proposers": N cópias do SLM (frota = hub.n_minions) resolvem o
problema de forma independente. Um "agregador" (o mestre) lê todas as propostas
e sintetiza a resposta final — sem rodadas adversariais entre os proposers.

Grafo:  START → propose → aggregate → END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, solve_messages,
    usage_delta,
)

_AGGREGATOR_SYSTEM = (
    "You have received independent responses from several models to the same "
    "math problem. Some may be wrong or contain reasoning errors. Evaluate each "
    "carefully, do not blindly copy any of them, and produce the correct, most "
    "complete and coherent final answer. Think step by step and ALWAYS end with "
    "a line '#### <final number>'."
)


class State(UsageState, total=False):
    question: str
    proposals: list[str]
    answer: str


@register
class MixtureOfAgents(Protocol):
    name = "mixture_of_agents"

    def build_graph(self):
        self.n_proposers = self.hub.n_minions    # tamanho da frota (parâmetro de rodada)

        g = StateGraph(State)
        g.add_node("propose", self._propose)
        g.add_node("aggregate", self._aggregate)
        g.add_edge(START, "propose")
        g.add_edge("propose", "aggregate")
        g.add_edge("aggregate", END)
        return g.compile()

    def _propose(self, state: State) -> dict[str, Any]:
        question = state["question"]
        gens = [
            self.hub.minion_agent().chat(solve_messages(question), gen_cfg=cfg.CREATIVE)
            for _ in range(self.n_proposers)
        ]
        return {"proposals": [g.text for g in gens],
                **combine_usage(gens, is_master=False)}

    def _aggregate(self, state: State) -> dict[str, Any]:
        question = state["question"]
        bloco = "\n\n".join(
            f"[Proposal {i+1}]\n{p}" for i, p in enumerate(state.get("proposals", []))
        )
        msgs = [
            {"role": "system", "content": _AGGREGATOR_SYSTEM},
            {"role": "user", "content":
                f"Problem:\n{question}\n\nModels' proposals:\n{bloco}"},
        ]
        gen = self.hub.master.chat(msgs)
        return {"answer": gen.text, **usage_delta(gen, is_master=True)}

    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "proposals": [], **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "proposals": final_state.get("proposals", []),
        }
