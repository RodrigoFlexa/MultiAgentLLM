#!/usr/bin/env python
"""
Ponto de entrada de UMA rodada (uma combinação de modelos/frota).

Exemplos:
    # todos os protocolos com Phi-4-mini (frota de 3) e mestre Qwen2.5-32B:
    python scripts/run_experiment.py --all --gpu 0 --n 200 \
        --minion phi4-mini --master qwen2.5-32b --n-minions 3

    # baseline do modelo grande sozinho:
    python scripts/run_experiment.py --protocols single_agent --master qwen2.5-32b

    # teste de fumaça sem GPU:
    MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --all --n 10

Para varrer a GRADE inteira de experimentos de uma vez, use scripts/run_sweep.py.

Flags:
    --all            roda todos os protocolos registrados (piso→teto→meio→foa)
    --minion KEY     SLM da frota / single_minion (chave do MODEL_CATALOG)
    --master KEY     mestre/orquestrador (chave do MODEL_CATALOG)
    --n-minions N    tamanho da frota (2..4) para debate/MoA/FoA
    --gpu G          GPU(s), ex.: "0" ou "0,1"
    --n N            nº de perguntas do GSM8K
    --seed S         seed da amostragem
    --protocols ...  subconjunto de protocolos
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.protocols import available


def all_protocols() -> tuple[str, ...]:
    preferida = ["single_minion", "single_agent", "minions",
                 "mixture_of_agents", "foa", "debate"]
    regs = available()
    return tuple([p for p in preferida if p in regs]
                 + [p for p in regs if p not in preferida])


def main() -> None:
    p = argparse.ArgumentParser(description="Rodada de protocolos multiagente no GSM8K")
    p.add_argument("--all", action="store_true", dest="run_all",
                   help="roda TODOS os protocolos registrados")
    p.add_argument("--minion", default=cfg.EXPERIMENT.minion,
                   help=f"SLM da frota — chaves: {sorted(cfg.MODEL_CATALOG)}")
    p.add_argument("--master", default=cfg.EXPERIMENT.master,
                   help=f"mestre/orquestrador — chaves: {sorted(cfg.MODEL_CATALOG)}")
    p.add_argument("--n-minions", type=int, default=cfg.EXPERIMENT.n_minions,
                   dest="n_minions", help="tamanho da frota (2..4)")
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--n", type=int, default=cfg.EXPERIMENT.n_samples)
    p.add_argument("--seed", type=int, default=cfg.EXPERIMENT.seed)
    p.add_argument("--protocols", nargs="+", default=None)
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # valida as chaves de modelo cedo (mensagem clara)
    cfg.get_model(args.minion); cfg.get_model(args.master)

    if args.run_all:
        protocols = all_protocols()
    elif args.protocols:
        protocols = tuple(args.protocols)
    else:
        protocols = cfg.EXPERIMENT.protocols

    from src.runner import run_experiment

    exp = replace(cfg.EXPERIMENT, n_samples=args.n, seed=args.seed,
                  minion=args.minion, master=args.master,
                  n_minions=args.n_minions, protocols=protocols)
    run_experiment(exp)


if __name__ == "__main__":
    main()
