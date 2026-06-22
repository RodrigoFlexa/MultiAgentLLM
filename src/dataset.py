"""
Carregamento do dataset GSM8K.

GSM8K (Grade School Math 8K) é ideal para este experimento: a resposta final é
sempre um número, então a verificação de acerto é 100% automática (sem juiz
humano). No dataset, o campo `answer` traz a solução passo a passo terminando
em uma linha `#### <número>`.

Se o `datasets` do HuggingFace não estiver disponível (ex.: máquina offline),
caímos num pequeno conjunto embutido para que o pipeline rode mesmo assim.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    question: str
    gold: str          # resposta-ouro (string numérica, ex.: "18")
    raw_answer: str = ""  # solução completa original (opcional)


def _parse_gold(answer_field: str) -> str:
    """Extrai o número que vem depois de '####' na resposta do GSM8K."""
    match = re.search(r"####\s*(.+)", answer_field)
    raw = match.group(1) if match else answer_field
    return raw.replace(",", "").replace("$", "").strip()


def load_gsm8k(n_samples: int, seed: int, split: str = "test") -> list[Sample]:
    """Carrega uma amostra aleatória (seed fixa => reprodutível) do GSM8K."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split)
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n_samples]
        return [
            Sample(
                question=ds[i]["question"],
                gold=_parse_gold(ds[i]["answer"]),
                raw_answer=ds[i]["answer"],
            )
            for i in idxs
        ]
    except Exception as exc:  # offline ou sem a lib: usa o fallback embutido
        print(f"[dataset] Falha ao carregar GSM8K ({exc}). Usando conjunto embutido.")
        return _fallback_samples(n_samples, seed)


# ──────────────────────────────────────────────────────────────────────
# Conjunto mínimo embutido (apenas para validar o pipeline offline)
# ──────────────────────────────────────────────────────────────────────
_FALLBACK = [
    Sample(
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did she sell altogether in April and May?",
        "72",
    ),
    Sample(
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "3",
    ),
    Sample(
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "10",
    ),
    Sample(
        "Betty is saving for a $100 wallet. She has half of the money she needs. "
        "Her parents give her $15 and her grandparents twice as much as her "
        "parents. How much more money does Betty need to buy the wallet?",
        "5",
    ),
    Sample(
        "James writes a 3-page letter to 2 different friends twice a week. How "
        "many pages does he write a year?",
        "624",
    ),
]


def _fallback_samples(n_samples: int, seed: int) -> list[Sample]:
    pool = list(_FALLBACK)
    random.Random(seed).shuffle(pool)
    # repete o pool se pedirem mais do que existe (suficiente p/ teste de fumaça)
    out = (pool * ((n_samples // len(pool)) + 1))[:n_samples]
    return out


def load_samples(n_samples: int, seed: int, dataset: str = "gsm8k",
                 split: str = "test") -> list[Sample]:
    if dataset == "gsm8k":
        return load_gsm8k(n_samples, seed, split)
    raise ValueError(f"Dataset não suportado: {dataset!r}")
