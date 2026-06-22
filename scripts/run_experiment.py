#!/usr/bin/env python
"""
Ponto de entrada do experimento.

Uso:
    # rodar TODOS os protocolos de uma vez (inclui o debate clássico):
    python scripts/run_experiment.py --all --gpu 0 --n 200

    # teste de fumaça (sem GPU, sem baixar modelos):
    MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --all --n 10

    # subconjunto específico:
    python scripts/run_experiment.py --gpu 0 --n 200 --protocols debate minions

Flags úteis:
    --all            roda todos os protocolos registrados, em ordem coerente
                      (piso → teto → meio → debate). Ignora o conjunto padrão.
    --gpu G          qual(is) GPU(s) usar (índice físico do `nvidia-smi`,
                      ex.: "0" ou "0,1"). Default: o que estiver em
                      CUDA_VISIBLE_DEVICES (.env) ou todas as GPUs visíveis.
    --n N            quantidade de perguntas do GSM8K (split de teste tem
                      1319 perguntas; pedir mais que isso usa todas elas)
    --seed S         seed da amostragem
    --protocols ...  subconjunto de protocolos (default: o conjunto padrão)
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


def all_protocols() -> tuple[str, ...]:
    """Todos os protocolos registrados, numa ordem didática (piso → teto →
    meio → debate). Protocolos novos não previstos entram no fim."""
    preferida = ["single_minion", "single_agent", "minions",
                 "mixture_of_agents", "debate"]
    registrados = available()
    ordenados = [p for p in preferida if p in registrados]
    ordenados += [p for p in registrados if p not in preferida]
    return tuple(ordenados)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar protocolos multiagente no GSM8K")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="roda TODOS os protocolos registrados (inclui o debate clássico)")
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

    if args.run_all:
        protocols = all_protocols()
    elif args.protocols:
        protocols = tuple(args.protocols)
    else:
        protocols = cfg.EXPERIMENT.protocols

    # Import tardio: só depois de definir a GPU acima.
    from src.runner import run_experiment

    exp = replace(cfg.EXPERIMENT, n_samples=args.n, seed=args.seed,
                  protocols=protocols)
    run_experiment(exp)


if __name__ == "__main__":
    main()
