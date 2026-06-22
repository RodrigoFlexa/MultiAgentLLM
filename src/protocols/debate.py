"""
Protocolo 3 — Debate Multiagente.

Vários SLMs (com personas diferentes) resolvem o problema, depois criticam as
respostas uns dos outros por algumas rodadas e, por fim, um agente "juiz" (o
LLM mestre) lê todo o debate e decide a resposta final.

Grafo (loop controlado por contador de rodadas):
    START → debate ──(round < N)──┐
              ▲                    │
              └────────────────────┘
            debate ──(round == N)──→ judge → END

Objetivo: avaliar se modelos pequenos debatendo superam o agente sozinho — e
medir o quanto o debate aumenta a latência (várias chamadas por pergunta).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, usage_delta,
)

_SOLVE_TAIL = " Pense passo a passo e termine com uma linha '#### <número final>'."


class State(UsageState, total=False):
    question: str
    answers: list[str]   # resposta atual de cada debatedor
    round: int
    answer: str          # veredito final do juiz


@register
class Debate(Protocol):
    name = "debate"

    def build_graph(self):
        self.n_debaters = cfg.DEBATE.n_debaters
        self.n_rounds = cfg.DEBATE.n_rounds
        self.personas = cfg.DEBATE.personas

        g = StateGraph(State)
        g.add_node("debate", self._debate_round)
        g.add_node("judge", self._judge)
        g.add_edge(START, "debate")
        g.add_conditional_edges(
            "debate", self._keep_debating, {"continue": "debate", "judge": "judge"}
        )
        g.add_edge("judge", END)
        return g.compile()

    # ── helpers de prompt ──────────────────────────────────────────────
    def _persona(self, i: int) -> str:
        return self.personas[i % len(self.personas)]

    def _solve_msgs(self, question: str, i: int) -> list[dict]:
        return [
            {"role": "system", "content": self._persona(i) + _SOLVE_TAIL},
            {"role": "user", "content": question},
        ]

    def _critique_msgs(self, question: str, i: int, others: list[str]) -> list[dict]:
        bloco = "\n\n".join(
            f"[Agente {j+1}]\n{ans}" for j, ans in enumerate(others) if j != i
        )
        user = (
            f"Problema:\n{question}\n\n"
            f"Respostas dos outros agentes na rodada anterior:\n{bloco}\n\n"
            "Aponte falhas no raciocínio deles (e no seu, se houver) e então dê "
            "sua resposta revisada e final, terminando com '#### <número final>'."
        )
        return [
            {"role": "system", "content": self._persona(i) + _SOLVE_TAIL},
            {"role": "user", "content": user},
        ]

    # ── nós ────────────────────────────────────────────────────────────
    def _debate_round(self, state: State) -> dict[str, Any]:
        question = state["question"]
        round_idx = state.get("round", 0)
        prev = state.get("answers", [])

        gens = []
        for i in range(self.n_debaters):
            msgs = (
                self._solve_msgs(question, i)
                if round_idx == 0
                else self._critique_msgs(question, i, prev)
            )
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        return {
            "answers": [g.text for g in gens],
            "round": round_idx + 1,
            **combine_usage(gens, is_master=False),
        }

    def _judge(self, state: State) -> dict[str, Any]:
        question = state["question"]
        bloco = "\n\n".join(
            f"[Agente {i+1}]\n{ans}" for i, ans in enumerate(state.get("answers", []))
        )
        msgs = [
            {"role": "system", "content":
                "Você é um juiz imparcial e rigoroso. Dadas as soluções de "
                "vários agentes, determine a resposta final correta."},
            {"role": "user", "content":
                f"Problema:\n{question}\n\nSoluções dos agentes:\n{bloco}\n\n"
                "Analise as soluções, identifique o consenso ou a que está "
                "correta, e termine com '#### <número final>'."},
        ]
        gen = self.hub.master.chat(msgs)
        return {"answer": gen.text, **usage_delta(gen, is_master=True)}

    # ── roteamento condicional ─────────────────────────────────────────
    def _keep_debating(self, state: State) -> str:
        return "continue" if state.get("round", 0) < self.n_rounds else "judge"

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "answers": [], "round": 0,
                **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "rounds": final_state.get("round", 0),
            "debater_answers": final_state.get("answers", []),
        }
