#!/usr/bin/env python
"""
Sweep: roda VÁRIOS experimentos num único processo, reaproveitando os modelos
já carregados na VRAM entre as configurações (o backend é compartilhado — o
Qwen-32B só é carregado uma vez na sessão toda).

Cada VARIAÇÃO é um experimento (um protocolo + uma combinação de modelos/frota).
Os eixos variados por protocolo estão em src/experiments.py.

Dois usos:
  * GRADE COMPLETA (todos os protocolos):
        python scripts/run_sweep.py --gpu 0 --n 200
  * UM PROTOCOLO isolado, varrendo só as variações dele:
        python scripts/run_sweep.py --gpu 0 --n 200 --protocols foa
        python scripts/run_sweep.py --gpu 0 --n 200 --protocols debate mixture_of_agents

Restringir eixos / ver o plano:
        python scripts/run_sweep.py --protocols foa --slms phi4-mini --counts 2 3
        python scripts/run_sweep.py --dry-run            # lista o plano e sai
        MULTIAGENT_BACKEND=mock python scripts/run_sweep.py --n 8   # teste sem GPU

Saída: um arquivo por variação (results/raw_<slug>.json) + um CSV mestre com
todas as linhas (results/sweep_summary.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.experiments import DEFAULT_SWEEP_PROTOCOLS, build_runs

# ── Eixos padrão da grade (edite à vontade) ───────────────────────────
SLMS = ["qwen3-4b", "phi4-mini"]
MASTERS = ["qwen2.5-32b", "qwen2.5-14b"]   # corte de custo: 4 -> 2 mestres
COUNTS = [2, 3, 4]


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep de experimentos multiagente")
    p.add_argument("--protocols", nargs="+", default=list(DEFAULT_SWEEP_PROTOCOLS),
                   help="quais protocolos varrer (default: grade completa). "
                        "Passe um só para rodar um protocolo isolado.")
    p.add_argument("--slms", nargs="+", default=SLMS)
    p.add_argument("--masters", nargs="+", default=MASTERS)
    p.add_argument("--counts", nargs="+", type=int, default=COUNTS)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--n", type=int, default=cfg.EXPERIMENT.n_samples)
    p.add_argument("--seed", type=int, default=cfg.EXPERIMENT.seed)
    p.add_argument("--out", default="sweep_summary",
                   help="nome-base do CSV/JSON mestre em results/")
    p.add_argument("--dry-run", action="store_true", help="lista o plano e sai")
    p.add_argument("--no-share", action="store_true",
                   help="backend novo por rodada (não reaproveita VRAM)")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    base = replace(cfg.EXPERIMENT, n_samples=args.n, seed=args.seed)
    plan = build_runs(base, args.protocols, args.slms, args.masters, args.counts)

    print(f"Plano: {len(plan)} experimentos | n={args.n} cada\n")
    for i, e in enumerate(plan, 1):
        proto = e.protocols[0]
        print(f"  {i:>2}. {proto:<18} SLM={e.minion:<11} "
              f"mestre={e.master:<11} N={e.n_minions}")
    if args.dry_run:
        print("\n(dry-run) nada foi executado.")
        return

    from src.llm import make_backend
    from src.runner import run_experiment

    backend = None if args.no_share else make_backend()  # compartilhado => reusa VRAM

    all_aggs = []
    for i, exp in enumerate(plan, 1):
        print(f"\n===== experimento {i}/{len(plan)} =====")
        all_aggs.extend(run_experiment(exp, backend=backend))

    os.makedirs(base.results_dir, exist_ok=True)
    rows = [a.as_row() for a in all_aggs]
    with open(os.path.join(base.results_dir, f"{args.out}.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(base.results_dir, f"{args.out}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n✔ Sweep concluído: {len(rows)} linhas em "
          f"{base.results_dir}/{args.out}.csv")


if __name__ == "__main__":
    main()
