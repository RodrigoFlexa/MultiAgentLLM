"""
Configuração central do experimento.

Tudo que você normalmente mexeria (modelos, tamanho da amostra, número de
rodadas de debate, etc.) está concentrado aqui. Os módulos em `src/` apenas
leem deste arquivo, então não há "números mágicos" espalhados pelo código.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────
# Modelos
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """Identifica um modelo no HuggingFace Hub e seus limites de geração."""
    name: str                 # repo_id no HuggingFace Hub
    label: str                # nome curto usado em logs/relatórios
    max_new_tokens: int = 512


# O "minion": SLM rápido e barato, fica na linha de frente.
MINION_MODEL = ModelConfig(
    name="Qwen/Qwen2.5-3B-Instruct",
    label="Qwen2.5-3B",
    max_new_tokens=512,
)

# O "mestre": LLM mais capaz (<= 20B params, conforme pedido). Resolve o que
# o minion não dá conta e atua como juiz no protocolo de debate.
# Qwen2.5-14B é forte em raciocínio matemático (GSM8K) e cabe folgado numa A100.
MASTER_MODEL = ModelConfig(
    name="Qwen/Qwen2.5-14B-Instruct",
    label="Qwen2.5-14B",
    max_new_tokens=768,
)


# ──────────────────────────────────────────────────────────────────────
# Backend de inferência
# ──────────────────────────────────────────────────────────────────────
# "hf"   -> HuggingFace Transformers de verdade (precisa de GPU).
# "mock" -> backend falso e determinístico, sem baixar nada. Serve para
#           validar o encanamento do LangGraph e dos protocolos numa máquina
#           sem GPU. Selecione com a variável de ambiente MULTIAGENT_BACKEND.
BACKEND: str = os.environ.get("MULTIAGENT_BACKEND", "hf")

# dtype para o Transformers. "bfloat16" é o ideal na A100.
TORCH_DTYPE: str = os.environ.get("MULTIAGENT_DTYPE", "bfloat16")

# Carregar o master em 4-bit (bitsandbytes). Desnecessário numa A100 80GB,
# útil se a VRAM apertar.
MASTER_LOAD_IN_4BIT: bool = os.environ.get("MASTER_4BIT", "0") == "1"


# ──────────────────────────────────────────────────────────────────────
# Geração (decodificação)
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0   # 0.0 => greedy/determinístico (bom p/ baseline)
    top_p: float = 1.0


# Decodificação determinística para single-agent e minion (reprodutibilidade).
DETERMINISTIC = GenerationConfig(temperature=0.0, top_p=1.0)

# Um pouco de temperatura no debate, para que os agentes divirjam de fato.
CREATIVE = GenerationConfig(temperature=0.7, top_p=0.95)


# ──────────────────────────────────────────────────────────────────────
# Protocolos
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MinionsConfig:
    # Token que o minion deve emitir quando não tem confiança na resposta.
    delegate_token: str = "<DELEGAR>"


@dataclass(frozen=True)
class DebateConfig:
    n_debaters: int = 2        # quantos SLMs debatem
    n_rounds: int = 2          # rodadas de crítica antes do juiz
    # Personas dão pontos de vista diferentes aos debatedores.
    personas: tuple[str, ...] = (
        "Você é um matemático rigoroso: resolve passo a passo, "
        "conferindo cada conta aritmética.",
        "Você é um pensador cético: procura ativamente erros de "
        "raciocínio e armadilhas no enunciado.",
    )


# ──────────────────────────────────────────────────────────────────────
# Experimento
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "gsm8k"
    split: str = "test"
    n_samples: int = 200       # amostra aleatória do GSM8K
    seed: int = 42
    results_dir: str = "results"
    # Quais protocolos rodar (nomes registrados no registry).
    # Ordem: piso (SLM sozinho) → teto (LLM sozinho) → meio (Minions, Debate).
    protocols: tuple[str, ...] = (
        "single_minion", "single_agent", "minions", "debate",
    )


EXPERIMENT = ExperimentConfig()
MINIONS = MinionsConfig()
DEBATE = DebateConfig()
