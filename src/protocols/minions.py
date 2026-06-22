"""
Protocolo 2 — Minions (delegação SLM → LLM).

A ideia central do "Minion": o modelo pequeno (SLM) fica na linha de frente e
resolve o que consegue. Quando não tem confiança, ele delega ao modelo grande.
A regra de roteamento usa auto-avaliação: o minion é instruído a responder
apenas o token de delegação quando estiver inseguro.

Grafo:
    START → minion → (delegou?) ─sim→ master → END
                              └────não─────────→ END

Objetivo: manter a acurácia perto da do "Agente Sozinho" grande, mas chamando o
LLM caro só numa fração das perguntas (menor custo e latência média).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, empty_usage, register, solve_messages, usage_delta,
)

_DELEGATE = cfg.MINIONS.delegate_token

MINION_SYSTEM = (
    "Você é um assistente pequeno e rápido que resolve problemas de matemática.\n"
    "Resolva passo a passo APENAS se tiver alta confiança e os passos forem "
    "claros, terminando com uma linha '#### <número final>'.\n"
    f"Se houver qualquer dúvida, ou o problema for complexo demais, responda "
    f"ESTRITAMENTE com '{_DELEGATE}' e mais nada."
)


class State(UsageState, total=False):
    question: str
    minion_answer: str
    answer: str
    delegated: bool


@register
class Minions(Protocol):
    name = "minions"

    def build_graph(self):
        g = StateGraph(State)
        g.add_node("minion", self._minion)
        g.add_node("master", self._master)
        g.add_edge(START, "minion")
        g.add_conditional_edges(
            "minion", self._route, {"delegate": "master", "accept": END}
        )
        g.add_edge("master", END)
        return g.compile()

    # ── nós ────────────────────────────────────────────────────────────
    def _minion(self, state: State) -> dict[str, Any]:
        gen = self.hub.minion.chat(
            solve_messages(state["question"], system=MINION_SYSTEM)
        )
        delegated = _DELEGATE in gen.text
        return {
            "minion_answer": gen.text,
            "answer": "" if delegated else gen.text,
            "delegated": delegated,
            **usage_delta(gen, is_master=False),
        }

    def _master(self, state: State) -> dict[str, Any]:
        gen = self.hub.master.chat(solve_messages(state["question"]))
        return {"answer": gen.text, **usage_delta(gen, is_master=True)}

    # ── roteamento condicional ─────────────────────────────────────────
    @staticmethod
    def _route(state: State) -> str:
        return "delegate" if state.get("delegated") else "accept"

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "delegated": final_state.get("delegated", False),
            "minion_answer": final_state.get("minion_answer", ""),
        }
