#!/usr/bin/env python
"""
Ponto de entrada do experimento.

Uso:
    # teste de fumaça (sem GPU, sem baixar modelos):
    MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --n 10

    # experimento real na A100 (HuggingFace Transformers):
    python scripts/run_experiment.py --n 200

Flags úteis:
    --n N            quantidade de perguntas do GSM8K
    --seed S         seed da amostragem
    --protocols ...  subconjunto de protocolos (default: todos os registrados)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

# Permite rodar a partir da raiz do projeto sem instalar como pacote.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.protocols import available
from src.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar protocolos multiagente no GSM8K")
    parser.add_argument("--n", type=int, default=cfg.EXPERIMENT.n_samples,
                        help="número de perguntas do GSM8K")
    parser.add_argument("--seed", type=int, default=cfg.EXPERIMENT.seed)
    parser.add_argument("--protocols", nargs="+", default=None,
                        help=f"subconjunto de {available()}")
    args = parser.parse_args()

    exp = replace(
        cfg.EXPERIMENT,
        n_samples=args.n,
        seed=args.seed,
        protocols=tuple(args.protocols) if args.protocols else cfg.EXPERIMENT.protocols,
    )
    run_experiment(exp)


if __name__ == "__main__":
    main()
