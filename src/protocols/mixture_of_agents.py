"""
hehe
Protocolo — Mixture-of-Agents (MoA), multi-camada.

Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities"
(2024, arXiv:2406.04692).

  * Camada 1: N proposers (cópias do SLM) respondem de forma independente.
  * Camadas 2..L: cada proposer RE-PROPÕE sintetizando TODAS as respostas da
    camada anterior (operador "aggregate-and-synthesize": avaliar criticamente,
    não copiar, e produzir uma resposta melhor).
  * Final: o MESTRE agrega a última camada na resposta final.

`n_layers` (config) é a profundidade do MoA — o análogo das rodadas do debate /
passos do FoA para o estudo de custo. n_layers=1 reproduz o MoA "raso"
(1 camada de propostas + agregação do mestre).

Grafo:  START → layer ──(layer < L)──┐
                  ▲                   │
                  └───────────────────┘
                layer ──(layer == L)──→ aggregate → END
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph

import config as cfg
from src.dataset import Sample
from src.protocols.base import (
    Protocol, UsageState, combine_usage, empty_usage, register, solve_messages,
    usage_delta,
)
from src.protocols.cluster import agent_costs_from_calls

# Prompt "aggregate-and-synthesize" do paper (usado nas camadas >1 e na
# agregação final do mestre).
_AGG_SYSTEM = (
    "You have been provided with responses from several models to the same math "
    "problem. Some of them may be biased, incomplete or simply wrong. Evaluate "
    "each one critically — do NOT copy any of them verbatim and do NOT assume the "
    "majority is right — and SYNTHESIZE a single, higher-quality, correct "
    "solution. Think step by step and ALWAYS end with a line '#### <final number>'."
)


def _agg_user(question: str, responses: list[str]) -> str:
    bloco = "\n\n".join(f"[Response {i+1}]\n{r}" for i, r in enumerate(responses))
    return f"Problem:\n{question}\n\nResponses to synthesize:\n{bloco}"


class State(UsageState, total=False):
    question: str
    responses: list[str]      # respostas da camada atual
    layer: int
    answer: str
    calls: Annotated[list, operator.add]


@register
class MixtureOfAgents(Protocol):
    name = "mixture_of_agents"

    def build_graph(self):
        self.n_proposers = self.hub.n_minions
        self.n_layers = cfg.MOA.n_layers

        g = StateGraph(State)
        g.add_node("layer", self._layer)
        g.add_node("aggregate", self._aggregate)
        g.add_edge(START, "layer")
        g.add_conditional_edges(
            "layer", self._more_layers, {"continue": "layer", "aggregate": "aggregate"}
        )
        g.add_edge("aggregate", END)
        return g.compile()

    def _layer(self, state: State) -> dict[str, Any]:
        q = state["question"]
        layer_idx = state.get("layer", 0)
        prev = state.get("responses", [])

        gens = []
        for i in range(self.n_proposers):
            if layer_idx == 0:                       # camada 1: independente
                msgs = solve_messages(q)
            else:                                    # camadas >1: sintetiza o conjunto
                msgs = [{"role": "system", "content": _AGG_SYSTEM},
                        {"role": "user", "content": _agg_user(q, prev)}]
            gens.append(self.hub.minion_agent().chat(msgs, gen_cfg=cfg.CREATIVE))

        calls = [self._rec(g, i, f"layer{layer_idx+1}") for i, g in enumerate(gens)]
        return {"responses": [g.text for g in gens], "layer": layer_idx + 1,
                "calls": calls, **combine_usage(gens, is_master=False)}

    def _aggregate(self, state: State) -> dict[str, Any]:
        q = state["question"]
        msgs = [{"role": "system", "content": _AGG_SYSTEM},
                {"role": "user", "content": _agg_user(q, state.get("responses", []))}]
        gen = self.hub.master.chat(msgs)
        return {"answer": gen.text, "calls": [self._rec(gen, "master", "aggregate")],
                **usage_delta(gen, is_master=True)}

    def _more_layers(self, state: State) -> str:
        return "continue" if state.get("layer", 0) < self.n_layers else "aggregate"

    @staticmethod
    def _rec(gen, agent, phase) -> dict:
        return {"agent": agent, "phase": phase,
                "completion_tokens": gen.completion_tokens,
                "prompt_tokens": gen.prompt_tokens,
                "compute_cost": gen.compute_cost,
                "latency_s": gen.latency_s}

    def initial_state(self, sample: Sample) -> dict[str, Any]:
        return {"question": sample.question, "responses": [], "layer": 0,
                "calls": [], **empty_usage()}

    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        calls = final_state.get("calls", [])
        return final_state.get("answer", ""), {
            "n_layers": final_state.get("layer", 0),
            "final_layer_responses": final_state.get("responses", []),
            "agent_costs": agent_costs_from_calls(calls),
        }
