"""
Protocolo 1 — Agente Sozinho (baseline).

"O melhor agente sozinho": o LLM mestre (modelo grande) resolve o problema
diretamente, com raciocínio passo a passo. É a referência de qualidade máxima
e de custo máximo contra a qual os outros protocolos são comparados.

Grafo:  START → solve → END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, empty_usage, register, solve_messages, usage_delta,
)


class State(UsageState, total=False):
    question: str
    answer: str


@register
class SingleAgent(Protocol):
    name = "single_agent"

    def build_graph(self):
        g = StateGraph(State)
        g.add_node("solve", self._solve)
        g.add_edge(START, "solve")
        g.add_edge("solve", END)
        return g.compile()

    # ── nó ──────────────────────────────────────────────────────────
    def _solve(self, state: State) -> dict[str, Any]:
        gen = self.hub.master.chat(solve_messages(state["question"]))
        return {"answer": gen.text, **usage_delta(gen, is_master=True)}

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {}
