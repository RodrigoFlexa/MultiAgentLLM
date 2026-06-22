"""
Protocolo 1 — Agente Sozinho (baseline).

Um único modelo resolve o problema diretamente, passo a passo. Dois baselines
saem do MESMO grafo, mudando apenas qual modelo do hub é usado:

  * `single_agent`  → o LLM mestre sozinho = teto de qualidade e custo máximo.
  * `single_minion` → o SLM minion sozinho = piso de qualidade e custo mínimo.

Ter os dois extremos torna fácil ler os protocolos do meio (Minions, Debate):
o quanto eles se aproximam do teto pagando perto do piso.

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
    solver = "master"   # atributo do LLMHub a ser usado ("master" ou "minion")

    def build_graph(self):
        g = StateGraph(State)
        g.add_node("solve", self._solve)
        g.add_edge(START, "solve")
        g.add_edge("solve", END)
        return g.compile()

    # ── nó ──────────────────────────────────────────────────────────
    def _solve(self, state: State) -> dict[str, Any]:
        llm = getattr(self.hub, self.solver)
        gen = llm.chat(solve_messages(state["question"]))
        return {"answer": gen.text, **usage_delta(gen, is_master=self.solver == "master")}

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {}


@register
class SingleMinion(SingleAgent):
    """O SLM resolvendo sozinho — mesmo grafo, modelo pequeno."""
    name = "single_minion"
    solver = "minion"
