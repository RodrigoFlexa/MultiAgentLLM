"""
Fundação dos protocolos.

Tudo que os três protocolos compartilham mora aqui:
  * o estado de "contabilização" do LangGraph (tokens/latência que se acumulam
    automaticamente conforme o grafo avança, via reducers);
  * a classe-base `Protocol` (constrói o grafo, roda uma pergunta, devolve um
    `QueryResult` padronizado);
  * um registry (`@register`) para que adicionar um 4º protocolo seja só criar
    um arquivo novo e decorá-lo — nada além disso precisa mudar.
"""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, TypedDict

from src.dataset import Sample
from src.llm import GenerationResult, LLMHub
from src.metrics import QueryResult, is_correct


# ──────────────────────────────────────────────────────────────────────
# Estado de contabilização (campos com reducer "soma" => acumulam sozinhos)
# ──────────────────────────────────────────────────────────────────────
class UsageState(TypedDict, total=False):
    latency_s: Annotated[float, operator.add]
    total_tokens: Annotated[int, operator.add]
    master_tokens: Annotated[int, operator.add]
    minion_tokens: Annotated[int, operator.add]
    n_model_calls: Annotated[int, operator.add]
    used_master: Annotated[bool, operator.or_]


_USAGE_KEYS = ("latency_s", "total_tokens", "master_tokens",
               "minion_tokens", "n_model_calls", "used_master")


def empty_usage() -> dict[str, Any]:
    """Valores iniciais; os reducers somam as deltas em cima destes."""
    return {"latency_s": 0.0, "total_tokens": 0, "master_tokens": 0,
            "minion_tokens": 0, "n_model_calls": 0, "used_master": False}


def usage_delta(gen: GenerationResult, *, is_master: bool) -> dict[str, Any]:
    """Converte um `GenerationResult` em deltas para o estado do grafo."""
    return {
        "latency_s": gen.latency_s,
        "total_tokens": gen.total_tokens,
        "master_tokens": gen.completion_tokens if is_master else 0,
        "minion_tokens": 0 if is_master else gen.completion_tokens,
        "n_model_calls": 1,
        "used_master": is_master,
    }


def combine_usage(gens: list[GenerationResult], *, is_master: bool) -> dict[str, Any]:
    """Soma as deltas de várias gerações feitas dentro de um único nó
    (ex.: uma rodada de debate com vários agentes)."""
    out = empty_usage()
    for g in gens:
        d = usage_delta(g, is_master=is_master)
        for k in ("latency_s", "total_tokens", "master_tokens",
                  "minion_tokens", "n_model_calls"):
            out[k] += d[k]
        out["used_master"] = out["used_master"] or d["used_master"]
    return out


# ──────────────────────────────────────────────────────────────────────
# Prompts compartilhados
# ──────────────────────────────────────────────────────────────────────
SOLVE_SYSTEM = (
    "Você resolve problemas de matemática com cuidado. Pense passo a passo "
    "e termine SEMPRE com uma linha no formato:\n#### <número final>"
)


def solve_messages(question: str, system: str = SOLVE_SYSTEM) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


# ──────────────────────────────────────────────────────────────────────
# Classe-base dos protocolos
# ──────────────────────────────────────────────────────────────────────
class Protocol(ABC):
    name: str = "base"

    def __init__(self, hub: LLMHub):
        self.hub = hub
        self.graph = self.build_graph()

    @abstractmethod
    def build_graph(self):
        """Monta e compila o StateGraph do LangGraph."""

    @abstractmethod
    def initial_state(self, sample: Sample) -> dict[str, Any]:
        """Estado inicial passado ao grafo para uma pergunta."""

    @abstractmethod
    def extract(self, final_state: dict[str, Any]) -> tuple[str, dict]:
        """Devolve (resposta_final_em_texto, infos_extras) do estado final."""

    def run(self, sample: Sample) -> QueryResult:
        final = self.graph.invoke(self.initial_state(sample))
        prediction, extra = self.extract(final)
        return QueryResult(
            protocol=self.name,
            question=sample.question,
            gold=sample.gold,
            prediction=prediction,
            correct=is_correct(prediction, sample.gold),
            latency_s=final.get("latency_s", 0.0),
            total_tokens=final.get("total_tokens", 0),
            master_tokens=final.get("master_tokens", 0),
            minion_tokens=final.get("minion_tokens", 0),
            used_master=final.get("used_master", False),
            n_model_calls=final.get("n_model_calls", 0),
            extra=extra,
        )


# ──────────────────────────────────────────────────────────────────────
# Registry: adicionar um protocolo = criar arquivo + @register
# ──────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[Protocol]] = {}


def register(cls: type[Protocol]) -> type[Protocol]:
    _REGISTRY[cls.name] = cls
    return cls


def get_protocol(name: str, hub: LLMHub) -> Protocol:
    if name not in _REGISTRY:
        raise KeyError(f"Protocolo {name!r} não registrado. "
                       f"Disponíveis: {sorted(_REGISTRY)}")
    return _REGISTRY[name](hub)


def available() -> list[str]:
    return sorted(_REGISTRY)
