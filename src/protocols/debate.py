"""
Protocolo — Debate Multiagente clássico ("society of minds").

Du, Li, Torralba, Tenenbaum, Mordatch — "Improving Factuality and Reasoning in
Language Models through Multiagent Debate" (2023, arXiv:2305.14325).

  * AGENTES HOMOGÊNEOS: N cópias do MESMO SLM (frota = hub.n_minions). Sem juiz.
  * Rodada 0: cada agente responde INDEPENDENTE.
  * Rodadas seguintes: cada agente vê as respostas dos OUTROS e revisa a sua.
  * Resposta final por VOTO MAJORITÁRIO entre os agentes.

Grafo:  START → debate ──(round < N)──┐
                  ▲                    │
                  └────────────────────┘
                debate ──(round == N)──→ tally → END
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.metrics import extract_final_number
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, solve_messages,
)


class State(UsageState, total=False):
    question: str
    answers: list[str]
    round: int
    answer: str
    vote_distribution: dict


@register
class Debate(Protocol):
    name = "debate"

    def build_graph(self):
        self.n_agents = self.hub.n_minions       # tamanho da frota (parâmetro de rodada)
        self.n_rounds = cfg.DEBATE.n_rounds

        g = StateGraph(State)
        g.add_node("debate", self._debate_round)
        g.add_node("tally", self._tally)
        g.add_edge(START, "debate")
        g.add_conditional_edges(
            "debate", self._keep_debating, {"continue": "debate", "tally": "tally"}
        )
        g.add_edge("tally", END)
        return g.compile()

    def _revise_msgs(self, question: str, i: int, others: list[str]) -> list[dict]:
        bloco = "\n\n".join(
            f"[Agent {j+1}]\n{ans}" for j, ans in enumerate(others) if j != i
        )
        user = (
            f"Problem:\n{question}\n\n"
            f"These are other agents' solutions to the same problem:\n{bloco}\n\n"
            "Use the other agents' solutions as additional information, revise "
            "your reasoning and give your updated answer, ending with a line "
            "'#### <final number>'."
        )
        return solve_messages(user)

    def _debate_round(self, state: State) -> dict[str, Any]:
        question = state["question"]
        round_idx = state.get("round", 0)
        prev = state.get("answers", [])

        gens = []
        for i in range(self.n_agents):
            msgs = (solve_messages(question) if round_idx == 0
                    else self._revise_msgs(question, i, prev))
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        return {
            "answers": [g.text for g in gens],
            "round": round_idx + 1,
            **combine_usage(gens, is_master=False),
        }

    def _tally(self, state: State) -> dict[str, Any]:
        answers = state.get("answers", [])
        nums = [extract_final_number(a) for a in answers]
        valid = [(i, n) for i, n in enumerate(nums) if n is not None]
        if not valid:
            return {"answer": answers[0] if answers else "", "vote_distribution": {}}

        counts = Counter(n for _, n in valid)
        max_votes = max(counts.values())
        empatados = [n for n, c in counts.items() if c == max_votes]
        first_idx = {n: next(i for i, m in valid if m == n) for n in empatados}
        winner = min(empatados, key=lambda n: first_idx[n])
        return {
            "answer": answers[first_idx[winner]],
            "vote_distribution": {str(n): c for n, c in counts.items()},
        }

    def _keep_debating(self, state: State) -> str:
        return "continue" if state.get("round", 0) < self.n_rounds else "tally"

    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "answers": [], "round": 0, **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "rounds": final_state.get("round", 0),
            "agent_answers": final_state.get("answers", []),
            "vote_distribution": final_state.get("vote_distribution", {}),
        }
