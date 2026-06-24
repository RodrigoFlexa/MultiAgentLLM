"""
Runner do experimento.

Roda os protocolos de UMA configuração (modelos + tamanho de frota fixos) sobre
a amostra do GSM8K e salva, para CADA protocolo, uma pasta própria e isolada:

    results/runs/<slug>/
        raw.json         # resultado pergunta a pergunta
        summary.json     # métricas agregadas deste experimento (1 linha)
        meta.json        # config completa: modelos+repos+params, n, seed, rounds...
        agent_costs.json # custo médio POR AGENTE (só protocolos que expõem isso)

O `slug` inclui o protocolo, então experimentos distintos (ex.: MoA e FoA com os
mesmos modelos/N) NUNCA se sobrescrevem. O `backend` pode ser compartilhado
entre rodadas (sweep) para reaproveitar os modelos já carregados na VRAM.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

from tqdm import tqdm

import config as cfg
from src.dataset import load_samples
from src.llm import Backend, LLMHub
from src.metrics import Aggregate, QueryResult, aggregate
from src.protocols import available, get_protocol


def slug(exp: cfg.ExperimentConfig, protocol: str) -> str:
    s = f"{protocol}__min-{exp.minion}__mas-{exp.master}__n{exp.n_minions}"
    if exp.version:
        s += f"__v{exp.version}"
    return s.replace("/", "-")


def _meta(exp: cfg.ExperimentConfig, protocol: str) -> dict:
    mc, ma = cfg.get_model(exp.minion), cfg.get_model(exp.master)
    return {
        "protocol": protocol,
        "dataset": exp.dataset, "split": exp.split,
        "n_samples": exp.n_samples, "seed": exp.seed,
        "n_minions": exp.n_minions,
        "minion": {"key": exp.minion, "repo": mc.name, "params_b": mc.params_b},
        "master": {"key": exp.master, "repo": ma.name, "params_b": ma.params_b},
        "debate_n_rounds": cfg.DEBATE.n_rounds,
        "foa_n_rounds": cfg.FOA.n_rounds,
        "backend": cfg.BACKEND, "version": exp.version,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _agent_costs(results: list[QueryResult]) -> Optional[list[dict]]:
    """Custo médio POR AGENTE ao longo das perguntas — responde 'quem trabalhou
    mais'. Só vale para protocolos que exponham extra['agent_costs'] (ex.: foa_dag)."""
    if not any(r.extra.get("agent_costs") for r in results):
        return None
    keys = ("completion_tokens", "prompt_tokens", "compute_cost", "latency_s", "n_calls")
    sums: dict = defaultdict(lambda: {k: 0.0 for k in keys})
    nq: dict = defaultdict(int)
    for r in results:
        for ac in r.extra.get("agent_costs", []):
            a = ac["agent"]
            for k in keys:
                sums[a][k] += ac.get(k, 0)
            nq[a] += 1
    total_cc = sum(v["compute_cost"] for v in sums.values()) or 1.0
    rows = []
    for a in sorted(sums):
        n = nq[a] or 1
        rows.append({
            "agent": a,
            "avg_completion_tokens": round(sums[a]["completion_tokens"] / n, 2),
            "avg_prompt_tokens": round(sums[a]["prompt_tokens"] / n, 2),
            "avg_compute_cost": round(sums[a]["compute_cost"] / n, 2),
            "avg_latency_s": round(sums[a]["latency_s"] / n, 4),
            "avg_n_calls": round(sums[a]["n_calls"] / n, 2),
            "share_compute_pct": round(100.0 * sums[a]["compute_cost"] / total_cc, 1),
        })
    return rows


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


def _save_run(exp: cfg.ExperimentConfig, protocol: str,
              results: list[QueryResult], agg: Aggregate) -> tuple[str, Optional[list]]:
    run_dir = os.path.join(exp.results_dir, "runs", slug(exp, protocol))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "raw.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg.as_row(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(_meta(exp, protocol), f, ensure_ascii=False, indent=2)
    agent_rows = _agent_costs(results)
    if agent_rows is not None:
        with open(os.path.join(run_dir, "agent_costs.json"), "w", encoding="utf-8") as f:
            json.dump(agent_rows, f, ensure_ascii=False, indent=2)
    return run_dir, agent_rows


def print_table(aggs: list[Aggregate]) -> None:
    print("\n" + "=" * 116)
    print(f"{'Protocolo':<16}{'SLM':>12}{'Mestre':>12}{'N':>3}"
          f"{'Acur.':>8}{'Lat.(s)':>10}{'Tokens':>10}{'Custo':>12}"
          f"{'Uso Mestre':>12}{'Chamadas':>10}")
    print("-" * 116)
    for a in aggs:
        print(f"{a.protocol:<16}{a.minion:>12}{a.master:>12}{a.n_minions:>3}"
              f"{a.accuracy:>7.1%}{a.avg_latency_s:>10.2f}{a.avg_total_tokens:>10.0f}"
              f"{a.avg_compute_cost:>12.0f}{a.master_usage_rate:>11.1%}"
              f"{a.avg_model_calls:>10.2f}")
    print("=" * 116 + "\n")


def run_experiment(exp: cfg.ExperimentConfig = cfg.EXPERIMENT,
                   backend: Optional[Backend] = None) -> list[Aggregate]:
    """Roda os protocolos de `exp`. Cada protocolo é salvo em sua própria pasta
    em results/runs/. Se `backend` for passado, é reutilizado (sweep)."""
    os.makedirs(exp.results_dir, exist_ok=True)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or "(todas visíveis)"
    print(f"GPU(s): {gpu} | Backend: {cfg.BACKEND} | SLM: {exp.minion} | "
          f"Mestre: {exp.master} | frota N={exp.n_minions} | n={exp.n_samples}")
    samples = load_samples(exp.n_samples, exp.seed, exp.dataset, exp.split)
    print(f"{len(samples)} perguntas carregadas.\n")

    hub = LLMHub(cfg.get_model(exp.minion), cfg.get_model(exp.master),
                 n_minions=exp.n_minions, backend=backend)

    aggs: list[Aggregate] = []
    for name in exp.protocols:
        if name not in available():
            print(f"[aviso] protocolo {name!r} não registrado — pulando.")
            continue
        print(f"▶ {name} (SLM={exp.minion}, mestre={exp.master}, N={exp.n_minions})")
        results = run_protocol(name, hub, samples)
        agg = aggregate(name, results, minion=exp.minion, master=exp.master,
                        n_minions=exp.n_minions)
        run_dir, agent_rows = _save_run(exp, name, results, agg)
        print(f"  → {run_dir}/")
        if agent_rows:
            print("    custo por agente (fatia do total):")
            for r in agent_rows:
                print(f"      agente {r['agent']} (subtarefa {r['agent']}): "
                      f"{r['share_compute_pct']:.1f}% | custo méd={r['avg_compute_cost']:.0f}"
                      f" | tokens méd={r['avg_completion_tokens']:.0f}")
        aggs.append(agg)

    print_table(aggs)
    return aggs
