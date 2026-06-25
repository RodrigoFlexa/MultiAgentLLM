"""
Protocolo — Debate Multiagente (estilo gold, Du et al. 2023, arXiv:2305.14325).

N cópias do MESMO SLM debatem. A diferença para uma versão "fraca" (onde todos
simplesmente aceitam a resposta alheia) é que aqui cada agente faz um DEBATE de
verdade: examina criticamente o raciocínio dos outros, reexamina o seu com a
mesma severidade, e só então atualiza — mudando de ideia apenas quando a lógica
justifica, sem se curvar à maioria. A resposta final sai por VOTO MAJORITÁRIO.

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
    Protocol, UsageState, combine_usage, empty_usage, register,
)

_SOLVE_SYSTEM = (
    "You are a sharp, independent problem-solver. Reason from first principles, "
    "check every arithmetic step, and do not cut corners. Solve the problem step "
    "by step and end with a line '#### <final number>'."
)

_DEBATE_SYSTEM = (
    "You are a careful debater in a panel solving a math problem. You do NOT "
    "defer to the majority — agreement matters only when it is actually correct. "
    "Scrutinize others' reasoning and your own with equal rigor, then commit to "
    "the answer the evidence supports. Always end with a line '#### <final number>'."
)


def _solve_msgs(question: str) -> list[dict]:
    return [{"role": "system", "content": _SOLVE_SYSTEM},
            {"role": "user", "content": f"Problem:\n{question}"}]


def _debate_msgs(question: str, own: str, others: list[str]) -> list[dict]:
    bloco = "\n\n".join(f"[Solver {j+1}]\n{a}" for j, a in enumerate(others))
    user = (
        f"Problem:\n{question}\n\n"
        f"Your previous solution:\n{own}\n\n"
        f"Other solvers' solutions:\n{bloco}\n\n"
        "Debate now, in this order:\n"
        "1) Examine each other solver's reasoning and arithmetic, and say "
        "specifically where each is right or wrong.\n"
        "2) Re-examine YOUR OWN previous solution with the same scrutiny and "
        "point out any mistake you find in it.\n"
        "3) Give your updated solution. Change your answer only if the reasoning "
        "justifies it — do not just follow the majority; if you still believe "
        "your answer, defend it.\n"
        "End with a line '#### <final number>'."
    )
    return [{"role": "system", "content": _DEBATE_SYSTEM},
            {"role": "user", "content": user}]


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
        self.n_agents = self.hub.n_minions
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

    def _debate_round(self, state: State) -> dict[str, Any]:
        question = state["question"]
        round_idx = state.get("round", 0)
        prev = state.get("answers", [])

        gens = []
        for i in range(self.n_agents):
            if round_idx == 0:
                msgs = _solve_msgs(question)
            else:
                others = [a for j, a in enumerate(prev) if j != i]
                msgs = _debate_msgs(question, prev[i], others)
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        return {"answers": [g.text for g in gens], "round": round_idx + 1,
                **combine_usage(gens, is_master=False)}

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
        return {"answer": answers[first_idx[winner]],
                "vote_distribution": {str(n): c for n, c in counts.items()}}

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
