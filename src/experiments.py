"""
Geração da grade de experimentos.

Cada protocolo varia apenas nos eixos que fazem sentido para ele; cada variação
vira UM experimento (um `ExperimentConfig` com um único protocolo). Isto serve
tanto para rodar a grade completa quanto para rodar UM protocolo isoladamente
varrendo só as suas variações.

Eixos:
  * slms    — SLMs candidatos (frota / modelo pequeno sozinho);
  * masters — modelos candidatos a mestre/orquestrador (e baseline sozinho);
  * counts  — tamanhos de frota (ex.: 2,3,4).

Quais eixos cada protocolo usa:
  single_agent       → 1 modelo sozinho            → varia: todos os modelos
  single_minion      → 1 SLM sozinho               → varia: slms
  minions            → 1 SLM + mestre              → varia: slms × masters
  debate             → N SLMs, sem mestre          → varia: slms × counts
  mixture_of_agents  → N SLMs + mestre agregador   → varia: slms × masters × counts
  foa                → N SLMs + mestre orquestrador→ varia: slms × masters × counts
  foa_dag            → decompõe em N subtarefas (DAG)→ varia: slms × masters × counts
"""
from __future__ import annotations

from dataclasses import replace

import config as cfg

# Protocolos incluídos por padrão na grade COMPLETA (single_minion é opt-in,
# pois o single_agent já cobre cada modelo sozinho).
DEFAULT_SWEEP_PROTOCOLS = (
    "single_agent", "minions", "debate", "mixture_of_agents", "foa", "foa_dag",
)


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def variations_for(protocol: str, slms, masters, counts) -> list[tuple[str, str, int]]:
    """Lista de (minion, master, n_minions) para um protocolo. Eixos que não se
    aplicam são neutralizados (ex.: master=minion quando não há mestre)."""
    if protocol == "single_agent":
        return [(m, m, 1) for m in _unique(list(masters) + list(slms))]
    if protocol == "single_minion":
        return [(s, s, 1) for s in slms]
    if protocol == "minions":
        return [(s, m, 1) for s in slms for m in masters]
    if protocol == "debate":
        return [(s, s, n) for s in slms for n in counts]
    if protocol in ("mixture_of_agents", "foa", "foa_dag"):
        return [(s, m, n) for s in slms for m in masters for n in counts]
    # protocolo desconhecido: varia todos os eixos
    return [(s, m, n) for s in slms for m in masters for n in counts]


def build_runs(base: cfg.ExperimentConfig, protocols, slms, masters,
               counts) -> list[cfg.ExperimentConfig]:
    """Uma `ExperimentConfig` por variação (um único protocolo cada)."""
    runs: list[cfg.ExperimentConfig] = []
    for proto in protocols:
        for minion, master, n in variations_for(proto, slms, masters, counts):
            runs.append(replace(base, protocols=(proto,), minion=minion,
                                master=master, n_minions=n))
    return runs
