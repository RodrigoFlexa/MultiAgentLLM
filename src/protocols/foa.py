"""
Protocolo — Federation of Agents (FoA), forma de RESOLVER a tarefa.

Giusti et al., "Federation of Agents: A Semantics-Aware Communication Fabric
for Large-Scale Agentic AI" (2025, arXiv:2509.20175).

Implementamos apenas o mecanismo de SOLUÇÃO do paper (não o roteamento
semântico/VCV nem o DAG de subtarefas, que não fazem sentido para uma tarefa
única como o GSM8K). O que fica é o coração colaborativo do FoA:

  1. FROTA (cluster): N SLMs homogêneos (frota = hub.n_minions).
  2. RASCUNHO: cada agente produz um rascunho inicial independente.
  3. REFINAMENTO (peer review): por k rodadas, cada agente vê os rascunhos dos
     pares e revisa o seu. Para cedo se a frota atingir CONSENSO numérico.
  4. SÍNTESE: um único ORQUESTRADOR (o mestre, "Agent-0") lê os rascunhos
     finais e sintetiza a resposta (operador SYNTH via meta-prompting).

Diferença para os vizinhos:
  * vs Debate: o Debate decide por voto majoritário e NÃO tem orquestrador;
    o FoA tem um mestre que SINTETIZA.
  * vs MoA: o MoA faz 1 camada de propostas independentes; o FoA REFINA em
    rodadas (os agentes veem uns aos outros) antes da síntese.

Grafo:  START → refine ──(round<N e sem consenso)──┐
                   ▲                                │
                   └────────────────────────────────┘
                 refine ──(consenso ou round==N)──→ synth → END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.metrics import extract_final_number
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, solve_messages,
    usage_delta,
)

_SYNTH_SYSTEM = (
    "You are the orchestrator of a fleet of agents (Agent-0). The agents "
    "independently drafted and refined solutions to the same math problem. "
    "Read their final solutions, resolve any disagreements between them, and "
    "produce the single correct final answer. Think step by step and ALWAYS "
    "end with a line '#### <final number>'."
)


class State(UsageState, total=False):
    question: str
    drafts: list[str]        # rascunho atual de cada agente da frota
    round: int
    consensus: bool          # a frota convergiu para o mesmo número?
    answer: str              # síntese do orquestrador


@register
class FoA(Protocol):
    name = "foa"

    def build_graph(self):
        self.n_agents = self.hub.n_minions       # tamanho da frota (parâmetro de rodada)
        self.n_rounds = cfg.FOA.n_rounds         # passes da frota (1 rascunho + refinos)

        g = StateGraph(State)
        g.add_node("refine", self._refine)
        g.add_node("synth", self._synth)
        g.add_edge(START, "refine")
        g.add_conditional_edges(
            "refine", self._route, {"continue": "refine", "synth": "synth"}
        )
        g.add_edge("synth", END)
        return g.compile()

    # ── prompts ────────────────────────────────────────────────────────
    def _refine_msgs(self, question: str, i: int, peers: list[str]) -> list[dict]:
        bloco = "\n\n".join(
            f"[Agent {j+1}]\n{ans}" for j, ans in enumerate(peers) if j != i
        )
        user = (
            f"Problem:\n{question}\n\n"
            f"Here are the current drafts from the other agents in your cluster:\n"
            f"{bloco}\n\n"
            "Critique their reasoning and yours, integrate any correct insights, "
            "and post your refined solution, ending with '#### <final number>'."
        )
        return solve_messages(user)

    # ── nós ────────────────────────────────────────────────────────────
    def _refine(self, state: State) -> dict[str, Any]:
        question = state["question"]
        round_idx = state.get("round", 0)
        prev = state.get("drafts", [])

        gens = []
        for i in range(self.n_agents):
            msgs = (solve_messages(question) if round_idx == 0      # rascunho inicial
                    else self._refine_msgs(question, i, prev))      # peer review
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        drafts = [g.text for g in gens]
        nums = [extract_final_number(d) for d in drafts]
        valid = [n for n in nums if n is not None]
        consensus = len(valid) >= 2 and len(set(valid)) == 1

        return {
            "drafts": drafts,
            "round": round_idx + 1,
            "consensus": consensus,
            **combine_usage(gens, is_master=False),
        }

    def _synth(self, state: State) -> dict[str, Any]:
        question = state["question"]
        bloco = "\n\n".join(
            f"[Agent {i+1}]\n{d}" for i, d in enumerate(state.get("drafts", []))
        )
        msgs = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content":
                f"Problem:\n{question}\n\nFleet's final solutions:\n{bloco}"},
        ]
        gen = self.hub.master.chat(msgs)
        return {"answer": gen.text, **usage_delta(gen, is_master=True)}

    # ── roteamento condicional ─────────────────────────────────────────
    def _route(self, state: State) -> str:
        if state.get("consensus"):
            return "synth"
        return "continue" if state.get("round", 0) < self.n_rounds else "synth"

    # ── plumbing ──────────────────────────────────────────────────────
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "drafts": [], "round": 0,
                "consensus": False, **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        return final_state.get("answer", ""), {
            "rounds": final_state.get("round", 0),
            "consensus": final_state.get("consensus", False),
            "fleet_drafts": final_state.get("drafts", []),
        }
