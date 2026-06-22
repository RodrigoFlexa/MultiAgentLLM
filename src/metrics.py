"""
Extração de resposta e métricas de avaliação.

Duas responsabilidades:
  1) tirar o número final da saída livre do modelo e dizer se acertou;
  2) agregar os resultados por protocolo (acurácia, latência, tokens, etc.),
     que é o material para comparar vantagens e desvantagens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from statistics import mean


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> float | None:
    """Extrai a resposta numérica final do texto do modelo.

    Prioridade: (1) número após '####'; (2) número após 'answer is/resposta é';
    (3) último número que aparecer no texto.
    """
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
# Registro por pergunta e agregação por protocolo
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    """O que cada protocolo devolve por pergunta."""
    protocol: str
    question: str
    gold: str
    prediction: str             # resposta final em texto
    correct: bool = False
    latency_s: float = 0.0
    total_tokens: int = 0
    master_tokens: int = 0      # tokens gerados pelo LLM grande (custo "caro")
    minion_tokens: int = 0      # tokens gerados pelo SLM (custo "barato")
    used_master: bool = False   # o LLM grande chegou a ser chamado?
    n_model_calls: int = 0
    extra: dict = field(default_factory=dict)  # campos específicos do protocolo


@dataclass
class Aggregate:
    protocol: str
    n: int
    accuracy: float
    avg_latency_s: float
    avg_total_tokens: float
    avg_master_tokens: float
    master_usage_rate: float    # fração de perguntas que acionaram o LLM grande
    avg_model_calls: float

    def as_row(self) -> dict:
        return asdict(self)


def aggregate(protocol: str, results: list[QueryResult]) -> Aggregate:
    n = len(results) or 1
    return Aggregate(
        protocol=protocol,
        n=len(results),
        accuracy=mean(r.correct for r in results) if results else 0.0,
        avg_latency_s=mean(r.latency_s for r in results) if results else 0.0,
        avg_total_tokens=mean(r.total_tokens for r in results) if results else 0.0,
        avg_master_tokens=mean(r.master_tokens for r in results) if results else 0.0,
        master_usage_rate=mean(r.used_master for r in results) if results else 0.0,
        avg_model_calls=mean(r.n_model_calls for r in results) if results else 0.0,
    )
