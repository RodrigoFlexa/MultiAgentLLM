"""
Configuração central do experimento.

Pontos que você normalmente ajusta:
  * MODEL_CATALOG  — quais modelos existem (repo no HF, nº de params, etc.);
  * minion / master / n_minions — quais modelos e quantos SLMs uma rodada usa;
  * DEBATE / FOA / MOA — nº de rodadas de refinamento de cada protocolo.

A escolha de modelos é POR RODADA (não há mais modelo "fixo" hardcoded): o
runner monta um LLMHub com os modelos pedidos. Isso é o que permite varrer a
grade de experimentos (protocolo × SLM × mestre × nº de SLMs) por script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Carrega o .env (se existir) antes de ler variáveis de ambiente. Precisa rodar
# antes do primeiro `import torch` (lazy em src/llm.py), pois decide a GPU.
load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# Modelos: catálogo
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """Um modelo do HuggingFace Hub + metadados usados em custo/relatórios."""
    name: str                  # repo_id no HuggingFace Hub
    label: str                 # nome curto usado em logs/relatórios/arquivos
    params_b: float            # nº de parâmetros em BILHÕES (base do custo)
    max_new_tokens: int = 512
    load_in_4bit: bool = False  # quantizar em 4-bit ao carregar (modelos grandes)


# Catálogo de modelos disponíveis. Para usar outros pesos, é só adicionar uma
# entrada aqui e referenciá-la pela chave em --minion / --master / no sweep.
# params_b alimenta a métrica de custo computacional (≈ params × tokens).
MODEL_CATALOG: dict[str, ModelConfig] = {
    # Mestres "grandes/médios"
    "qwen2.5-32b": ModelConfig(
        "Qwen/Qwen2.5-32B-Instruct", "Qwen2.5-32B", params_b=32.0,
        max_new_tokens=768, load_in_4bit=True,   # 4-bit p/ caber folgado na A100
    ),
    "qwen2.5-14b": ModelConfig(
        "Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B", params_b=14.0,
        max_new_tokens=768,
    ),
    # Modelos pequenos (servem tanto de SLM/minion quanto de mestre "pequeno")
    "qwen3-4b": ModelConfig(
        "Qwen/Qwen3-4B", "Qwen3-4B", params_b=4.0, max_new_tokens=512,
    ),
    "phi4-mini": ModelConfig(
        # Phi-4-mini-instruct tem ~3.8B params; ajuste params_b se preferir
        # outra contagem (ela só afeta a métrica de custo).
        "microsoft/Phi-4-mini-instruct", "Phi-4-mini", params_b=3.8,
        max_new_tokens=512,
    ),
}


def get_model(key: str) -> ModelConfig:
    if key not in MODEL_CATALOG:
        raise KeyError(f"Modelo {key!r} não está no catálogo. "
                       f"Disponíveis: {sorted(MODEL_CATALOG)}")
    return MODEL_CATALOG[key]


# Defaults de uma rodada avulsa (o sweep sobrescreve estes).
DEFAULT_MINION = "qwen3-4b"     # SLM da frota (debate/MoA/FoA) e do single_minion
DEFAULT_MASTER = "qwen2.5-14b"  # orquestrador/mestre (single_agent, MoA, FoA, minions)


# ──────────────────────────────────────────────────────────────────────
# GPU / backend / decodificação
# ──────────────────────────────────────────────────────────────────────
GPU_DEVICE: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")

# "hf" -> Transformers de verdade (GPU). "mock" -> backend falso (sem GPU).
BACKEND: str = os.environ.get("MULTIAGENT_BACKEND", "hf")
TORCH_DTYPE: str = os.environ.get("MULTIAGENT_DTYPE", "bfloat16")


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0   # 0.0 => greedy/determinístico
    top_p: float = 1.0


DETERMINISTIC = GenerationConfig(temperature=0.0, top_p=1.0)
CREATIVE = GenerationConfig(temperature=0.7, top_p=0.95)  # diversidade na frota


# ──────────────────────────────────────────────────────────────────────
# Protocolos
# ──────────────────────────────────────────────────────────────────────
# O NÚMERO de SLMs (2..4) NÃO mora aqui: é parâmetro de rodada (n_minions),
# carregado no LLMHub e lido pelos protocolos multi-agente (debate/MoA/FoA).
@dataclass(frozen=True)
class MinionsConfig:
    delegate_token: str = "<DELEGATE>"


@dataclass(frozen=True)
class DebateConfig:
    """Debate "society of minds" (Du et al. 2023): N cópias do SLM revisam-se
    por rodadas; resposta final por voto majoritário, sem orquestrador."""
    n_rounds: int = 2          # rodadas no total (1 inicial + revisões)


@dataclass(frozen=True)
class MoAConfig:
    """Mixture-of-Agents (Wang et al. 2024): N SLMs propõem em paralelo e o
    mestre agrega. 1 camada de propostas, sem rodadas adversariais."""
    pass                        # nº de proposers = n_minions (parâmetro de rodada)


@dataclass(frozen=True)
class ClusterConfig:
    """Mecanismo de cluster do FoA (usado pelo `foa` e por subtarefas complexas
    do `foa_dag`). Reflexão crítica + nota: ao ver as respostas dos pares, cada
    agente faz uma análise crítica/autocrítica (sem tratar nenhuma resposta como
    verdade) E dá uma NOTA a cada par; depois re-resolve usando os insights. Para
    cedo em caso de consenso; sem consenso, vence a maior nota acumulada."""
    max_size: int = 3          # nº máximo de agentes no cluster
    max_steps: int = 3         # nº máximo de passos (1 rascunho + refinos)


@dataclass(frozen=True)
class FoADagConfig:
    """foa_dag: o orquestrador decide quebrar (ou não) o problema em subtarefas,
    atribui um agente a cada uma (reuso permitido) e marca quais são complexas
    (resolvidas por um cluster). Execução em sequência com síntese progressiva."""
    max_subtasks: int = 6      # teto de subtarefas que o orquestrador pode criar


@dataclass(frozen=True)
class FoAConfig:
    """Federation of Agents (Giusti et al. 2025, arXiv:2509.20175) — apenas a
    forma de RESOLVER (sem roteamento semântico/DAG): uma frota de N SLMs faz um
    rascunho, refina por k rodadas vendo os rascunhos dos pares (peer review) e
    um único orquestrador (o mestre) sintetiza a resposta final."""
    n_rounds: int = 2          # rodadas de refinamento da frota antes da síntese


# ──────────────────────────────────────────────────────────────────────
# Experimento (uma rodada totalmente especificada)
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "gsm8k"
    split: str = "test"
    n_samples: int = 200
    seed: int = 42
    results_dir: str = "results"
    version: str | None = None     # sufixo opcional nos arquivos de saída

    # Modelos e tamanho da frota desta rodada:
    minion: str = DEFAULT_MINION   # chave no MODEL_CATALOG
    master: str = DEFAULT_MASTER   # chave no MODEL_CATALOG
    n_minions: int = 3             # nº de SLMs na frota (debate/MoA/FoA): 2..4

    protocols: tuple[str, ...] = (
        "single_minion", "single_agent", "minions", "mixture_of_agents", "foa",
    )


EXPERIMENT = ExperimentConfig()
MINIONS = MinionsConfig()
DEBATE = DebateConfig()
MOA = MoAConfig()
FOA = FoAConfig()
CLUSTER = ClusterConfig()
DAG = FoADagConfig()
