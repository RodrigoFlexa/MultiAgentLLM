"""
Extração de resposta e métricas de avaliação.

  1) tira o número final da saída do modelo e diz se acertou;
  2) agrega por rodada (acurácia × latência × tokens × CUSTO computacional),
     guardando também QUAIS modelos e quantos SLMs aquela rodada usou — é o
     material para o estudo de custo × latência × acurácia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from statistics import mean


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> float | None:
    """Número final da resposta: (1) após '####'; (2) após 'answer is/...';
    (3) último número do texto."""
    if not text:
        return None

    hashed = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if hashed:
        return _to_float(hashed.group(1))

    phrase = re.search(
        r"(?:answer is|resposta\s+é|resposta:|answer:)\s*\$?(-?\d[\d,]*\.?\d*)",
        text, flags=re.IGNORECASE,
    )
    if phrase:
        return _to_float(phrase.group(1))

    nums = _NUMBER_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def is_correct(prediction: str, gold: str, tol: float = 1e-4) -> bool:
    pred = extract_final_number(prediction)
    gold_val = _to_float(gold)
    if pred is None or gold_val is None:
        return False
    return abs(pred - gold_val) <= tol


# ──────────────────────────────────────────────────────────────────────
# Registro por pergunta e agregação por rodada
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    protocol: str
    question: str
    gold: str
    prediction: str
    correct: bool = False
    latency_s: float = 0.0
    total_tokens: int = 0
    master_tokens: int = 0       # tokens do mestre/orquestrador (caros)
    minion_tokens: int = 0       # tokens dos SLMs (baratos)
    compute_cost: float = 0.0    # Σ params_b × tokens gerados (proxy de custo)
    used_master: bool = False
    n_model_calls: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class Aggregate:
    protocol: str
    minion: str                  # SLM usado na rodada
    master: str                  # mestre/orquestrador usado na rodada
    n_minions: int               # tamanho da frota
    n: int
    accuracy: float
    avg_latency_s: float
    avg_total_tokens: float
    avg_compute_cost: float      # custo computacional médio (params × tokens)
    avg_master_tokens: float
    master_usage_rate: float
    avg_model_calls: float

    def as_row(self) -> dict:
        return asdict(self)


def aggregate(protocol: str, results: list[QueryResult], *,
              minion: str = "", master: str = "", n_minions: int = 0) -> Aggregate:
    return Aggregate(
        protocol=protocol,
        minion=minion,
        master=master,
        n_minions=n_minions,
        n=len(results),
        accuracy=mean(r.correct for r in results) if results else 0.0,
        avg_latency_s=mean(r.latency_s for r in results) if results else 0.0,
        avg_total_tokens=mean(r.total_tokens for r in results) if results else 0.0,
        avg_compute_cost=mean(r.compute_cost for r in results) if results else 0.0,
        avg_master_tokens=mean(r.master_tokens for r in results) if results else 0.0,
        master_usage_rate=mean(r.used_master for r in results) if results else 0.0,
        avg_model_calls=mean(r.n_model_calls for r in results) if results else 0.0,
    )
