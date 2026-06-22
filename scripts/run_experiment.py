#!/usr/bin/env python
"""
Ponto de entrada do experimento.

Uso:
    # teste de fumaça (sem GPU, sem baixar modelos):
    MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --n 10

    # experimento real, escolhendo GPU, protocolos e nº de perguntas:
    python scripts/run_experiment.py --gpu 0 --n 200 --protocols debate minions

Flags úteis:
    --gpu G          qual(is) GPU(s) usar (índice físico do `nvidia-smi`,
                      ex.: "0" ou "0,1"). Default: o que estiver em
                      CUDA_VISIBLE_DEVICES (.env) ou todas as GPUs visíveis.
    --n N            quantidade de perguntas do GSM8K (split de teste tem
                      1319 perguntas; pedir mais que isso usa todas elas)
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
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU(s) a usar, ex.: '0' ou '0,1' (default: CUDA_VISIBLE_DEVICES do .env)")
    parser.add_argument("--n", type=int, default=cfg.EXPERIMENT.n_samples,
                        help="número de perguntas do GSM8K (split de teste tem 1319)")
    parser.add_argument("--seed", type=int, default=cfg.EXPERIMENT.seed)
    parser.add_argument("--protocols", nargs="+", default=None,
                        help=f"subconjunto de {available()}")
    args = parser.parse_args()

    if args.gpu is not None:
        # Precisa ser setado antes do primeiro `import torch`, que só acontece
        # sob demanda (lazy) dentro de src/llm.py — a tempo, portanto.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    exp = replace(
        cfg.EXPERIMENT,
        n_samples=args.n,
        seed=args.seed,
        protocols=tuple(args.protocols) if args.protocols else cfg.EXPERIMENT.protocols,
    )
    run_experiment(exp)


if __name__ == "__main__":
    main()
