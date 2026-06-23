"""
Protocolo — Debate Multiagente clássico ("society of minds").

Implementação fiel ao primeiro artigo de debate entre LLMs:
Du, Li, Torralba, Tenenbaum, Mordatch — "Improving Factuality and Reasoning in
Language Models through Multiagent Debate" (2023, arXiv:2305.14325).

Características do método original (e que diferenciam de uma versão com juiz):
  * AGENTES HOMOGÊNEOS: N cópias do MESMO modelo (aqui, o minion). Sem personas.
  * Rodada 0: cada agente responde de forma INDEPENDENTE.
  * Rodadas seguintes: cada agente recebe as respostas dos OUTROS agentes da
    rodada anterior e revisa a sua, usando-as como informação adicional.
  * SEM JUIZ: a resposta final sai por VOTO MAJORITÁRIO entre os agentes.

Grafo (loop de rodadas + apuração do voto):
    START → debate ──(round < N)──┐
              ▲                    │
              └────────────────────┘
            debate ──(round == N)──→ tally → END

A diversidade entre cópias idênticas vem da amostragem (temperatura > 0).
Nenhuma chamada usa o modelo grande, então toda a contabilização é de minion.
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
    answers: list[str]          # resposta atual de cada agente
    round: int
    answer: str                 # consenso final (texto do agente representante)
    vote_distribution: dict     # {valor_numérico(str): nº de votos}


@register
class Debate(Protocol):
    name = "debate"

    def build_graph(self):
        self.n_agents = cfg.DEBATE.n_agents
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

    # ── prompts ────────────────────────────────────────────────────────
    def _revise_msgs(self, question: str, i: int, others: list[str]) -> list[dict]:
        bloco = "\n\n".join(
            f"[Agent {j+1}]\n{ans}" for j, ans in enumerate(others) if j != i
        )
        user = (
            f"Problem:\n{question}\n\n"
            f"These are other agents' solutions to the same problem:\n"
            f"{bloco}\n\n"
            "Use the other agents' solutions as additional information, "
            "revise your reasoning and give your updated answer, ending "
            "with a line '#### <final number>'."
        )
        return solve_messages(user)

    # ── nós ────────────────────────────────────────────────────────────
    def _debate_round(self, state: State) -> dict[str, Any]:
        question = state["question"]
        round_idx = state.get("round", 0)
        prev = state.get("answers", [])

        gens = []
        for i in range(self.n_agents):
            msgs = (
                solve_messages(question)                 # rodada 0: independente
                if round_idx == 0
                else self._revise_msgs(question, i, prev)  # rodadas: revisão
            )
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        return {
            "answers": [g.text for g in gens],
            "round": round_idx + 1,
            **combine_usage(gens, is_master=False),
        }

    def _tally(self, state: State) -> dict[str, Any]:
        """Voto majoritário sobre o número final de cada agente. Sem modelo:
        nó puramente determinístico (não altera a contabilização)."""
        answers = state.get("answers", [])
        nums = [extract_final_number(a) for a in answers]
        valid = [(i, n) for i, n in enumerate(nums) if n is not None]

        if not valid:
            return {"answer": answers[0] if answers else "", "vote_distribution": {}}

        counts = Counter(n for _, n in valid)
        max_votes = max(counts.values())
        empatados = [n for n, c in counts.items() if c == max_votes]
        # Desempate determinístico: número que aparece no agente de menor índice.
        first_idx = {n: next(i for i, m in valid if m == n) for n in empatados}
        winner = min(empatados, key=lambda n: first_idx[n])
        rep_idx = first_idx[winner]

        return {
            "answer": answers[rep_idx],
            "vote_distribution": {str(n): c for n, c in counts.items()},
        }

    # ── roteamento condicional ─────────────────────────────────────────
    def _keep_debating(self, state: State) -> str:
        return "continue" if state.get("round", 0) < self.n_rounds else "tally"

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "answers": [], "round": 0,
                **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "rounds": final_state.get("round", 0),
            "agent_answers": final_state.get("answers", []),
            "vote_distribution": final_state.get("vote_distribution", {}),
        }
