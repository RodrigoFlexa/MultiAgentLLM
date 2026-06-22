"""
Runner do experimento.

Roda cada protocolo sobre a mesma amostra do GSM8K, agrega as métricas e salva:
  * results/raw_<protocolo>.json  → resultado pergunta a pergunta;
  * results/summary.json / .csv   → tabela comparativa entre os protocolos.

Também imprime no console uma tabela com acurácia, latência, tokens e uso do
LLM mestre — exatamente o material para discutir vantagens e desvantagens.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict

from tqdm import tqdm

import config as cfg
from src.dataset import load_samples
from src.llm import LLMHub
from src.metrics import Aggregate, QueryResult, aggregate
from src.protocols import available, get_protocol


def run_protocol(name: str, hub: LLMHub, samples) -> list[QueryResult]:
    proto = get_protocol(name, hub)
    results: list[QueryResult] = []
    t0 = time.perf_counter()
    bar = tqdm(samples, desc=f"[{name}]", unit="q")
    for sample in bar:
        results.append(proto.run(sample))
        acc = sum(r.correct for r in results) / len(results)
        bar.set_postfix(acc=f"{acc:.1%}")
    bar.close()
    print(f"  [{name}] concluído em {time.perf_counter() - t0:.1f}s")
    return results


def _save_raw(results_dir: str, name: str, results: list[QueryResult]) -> None:
    path = os.path.join(results_dir, f"raw_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def _save_summary(results_dir: str, aggs: list[Aggregate]) -> None:
    rows = [a.as_row() for a in aggs]
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        with open(os.path.join(results_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _print_table(aggs: list[Aggregate]) -> None:
    print("\n" + "=" * 92)
    print(f"{'Protocolo':<16}{'Acurácia':>10}{'Latência(s)':>14}"
          f"{'Tokens':>12}{'Tok.Mestre':>12}{'Uso Mestre':>12}{'Chamadas':>10}")
    print("-" * 92)
    for a in aggs:
        print(f"{a.protocol:<16}{a.accuracy:>9.1%}{a.avg_latency_s:>14.2f}"
              f"{a.avg_total_tokens:>12.0f}{a.avg_master_tokens:>12.0f}"
              f"{a.master_usage_rate:>11.1%}{a.avg_model_calls:>10.2f}")
    print("=" * 92 + "\n")


def run_experiment(exp: cfg.ExperimentConfig = cfg.EXPERIMENT) -> list[Aggregate]:
    os.makedirs(exp.results_dir, exist_ok=True)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or "(todas visíveis)"
    print(f"GPU(s): {gpu} | Backend: {cfg.BACKEND} | "
          f"Minion: {cfg.MINION_MODEL.label} | Mestre: {cfg.MASTER_MODEL.label}")
    print(f"Carregando {exp.n_samples} amostras de {exp.dataset} (seed={exp.seed})...")
    samples = load_samples(exp.n_samples, exp.seed, exp.dataset, exp.split)
    print(f"{len(samples)} perguntas carregadas.\n")

    hub = LLMHub()
    aggs: list[Aggregate] = []
    for name in exp.protocols:
        if name not in available():
            print(f"[aviso] protocolo {name!r} não registrado — pulando.")
            continue
        print(f"▶ Rodando protocolo: {name}")
        results = run_protocol(name, hub, samples)
        _save_raw(exp.results_dir, name, results)
        aggs.append(aggregate(name, results))

    _save_summary(exp.results_dir, aggs)
    _print_table(aggs)
    print(f"Resultados salvos em: {exp.results_dir}/")
    return aggs
