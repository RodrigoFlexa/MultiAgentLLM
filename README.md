# MultiAgentLLM — protocolos multiagente (SLM × LLM) no GSM8K

Bancada de testes que compara formas de organizar modelos de linguagem para
resolver o mesmo problema, medindo o trade-off **acurácia × latência × custo
computacional**. A tarefa é o **GSM8K** (matemática escolar): a resposta final é
sempre um número, então o acerto é verificado automaticamente por script.

A orquestração de cada protocolo é um grafo de estados em **LangGraph**; os
modelos rodam localmente via **HuggingFace Transformers**. As duas camadas são
independentes: o protocolo não sabe qual modelo está por baixo, e vice-versa.

---

## 1. Ambiente (conda)

```bash
conda create -n multiagent python=3.11 -y
conda activate multiagent
pip install -r requirements.txt
```

Isso instala LangGraph, Transformers, PyTorch (com CUDA), `datasets`, etc. Para
rodar de verdade é preciso uma GPU; o Qwen2.5-32B usa carregamento em 4-bit
(via `bitsandbytes`) e cabe numa A100.

---

## 2. Modelos

Ficam no catálogo em `config.py` (`MODEL_CATALOG`); cada rodada escolhe o SLM e o
mestre **por chave**:

| Chave | Modelo | Papel típico |
|-------|--------|--------------|
| `qwen2.5-32b` | Qwen2.5-32B-Instruct (4-bit) | mestre forte |
| `qwen2.5-14b` | Qwen2.5-14B-Instruct | mestre médio |
| `qwen3-4b`    | Qwen3-4B | SLM / mestre pequeno |
| `phi4-mini`   | Phi-4-mini-instruct | SLM / mestre pequeno |

Para usar outros pesos, basta adicionar uma entrada no catálogo. O campo
`params_b` (nº de parâmetros em bilhões) alimenta a métrica de custo.

---

## 3. Métodos implementados

Cada protocolo tem um nome (usado em `--protocols`):

- **`single_minion`** — um SLM resolve sozinho. É o piso de qualidade e custo.
- **`single_agent`** — um único modelo (qualquer um do catálogo) resolve
  sozinho, passo a passo. Serve de baseline/teto para cada modelo.
- **`minions`** — delegação: o SLM tenta resolver e, se não tiver confiança,
  delega ao mestre (LLM grande). *(Em redesenho para a forma do paper, com N
  SLMs e decomposição em subtarefas.)*
- **`debate`** — "society of minds" (Du et al. 2023): N cópias do **mesmo** SLM
  respondem e revisam suas respostas vendo as dos outros por algumas rodadas; a
  resposta final sai por **voto majoritário**, sem mestre.
- **`mixture_of_agents`** — MoA (Wang et al. 2024): N SLMs propõem soluções de
  forma independente e o **mestre agrega/sintetiza** a final (1 camada, sem
  rodadas adversariais).
- **`foa`** — Federation of Agents (Giusti et al. 2025), só a parte de resolver:
  uma frota de N SLMs faz um rascunho, **refina por k rodadas vendo os pares**
  (peer review, com parada antecipada por consenso) e um **orquestrador (o
  mestre) sintetiza** a resposta final. Difere do debate (que vota, sem mestre)
  e do MoA (que não refina entre rodadas).

O **número de SLMs da frota** (`--n-minions`, normalmente 2–4) vale para
`debate`, `mixture_of_agents` e `foa`. Os modelos "sozinhos" usam 1; o `minions`
usa 1 SLM + mestre.

---

## 4. Como rodar

Três níveis, do mais específico ao mais amplo.

### 4.1 Uma rodada (uma configuração exata)

`scripts/run_experiment.py` roda os protocolos que você listar, com modelos e
frota fixos:

```bash
# FoA com frota de 3 Phi-4-mini e mestre Qwen2.5-32B, 200 perguntas, GPU 0:
python scripts/run_experiment.py --protocols foa \
    --minion phi4-mini --master qwen2.5-32b --n-minions 3 --gpu 0 --n 200

# baseline: o modelo grande sozinho
python scripts/run_experiment.py --protocols single_agent --master qwen2.5-32b --gpu 0

# debate com 4 cópias do Qwen3-4B (debate não usa mestre)
python scripts/run_experiment.py --protocols debate --minion qwen3-4b --n-minions 4 --gpu 0

# todos os protocolos de uma vez, com a config dada
python scripts/run_experiment.py --all --minion qwen3-4b --master qwen2.5-14b --n-minions 3 --gpu 0
```

### 4.2 Um protocolo, varrendo suas variações

`scripts/run_sweep.py --protocols <nome>` roda **um experimento por variação**
do protocolo (cada protocolo varia só nos eixos que fazem sentido — veja
`src/experiments.py`):

```bash
python scripts/run_sweep.py --protocols foa --gpu 0 --n 200      # 24 experimentos
python scripts/run_sweep.py --protocols debate --gpu 0 --n 200   # 6 experimentos
python scripts/run_sweep.py --protocols mixture_of_agents foa --gpu 0 --n 200
```

Eixos restringíveis: `--slms`, `--masters`, `--counts`. Use `--dry-run` para só
listar o plano:

```bash
python scripts/run_sweep.py --protocols foa --slms phi4-mini --counts 2 3 --dry-run
```

### 4.3 Grade completa (todos os protocolos × variações)

```bash
python scripts/run_sweep.py --gpu 0 --n 200          # 38 experimentos
# ou o atalho:
bash scripts/run_sweep.sh 0 200
```

O sweep compartilha o backend entre as rodadas, então cada modelo é carregado
**uma vez** na VRAM e reutilizado. Numa A100 de 40 GB, se faltar memória, use
`--no-share` (recarrega por rodada).

---

## 5. Saídas e métricas

Cada experimento (1 protocolo numa configuração) ganha **sua própria pasta**,
nomeada pela config — assim nada se sobrescreve e dá pra saber na hora o que
rodou:

```
results/
  runs/
    foa__min-qwen3-4b__mas-qwen2.5-32b__n2/
      raw.json       # resultado pergunta a pergunta (com a contabilização)
      summary.json   # métricas agregadas deste experimento
      meta.json      # config completa: repos dos modelos, params, n, seed, rounds, timestamp
    mixture_of_agents__min-phi4-mini__mas-qwen2.5-14b__n3/
      ...
  sweep_summary.csv  # 1 linha por experimento — a tabela mestre p/ análise
  sweep_summary.json
```

O `slug` da pasta inclui o protocolo, então MoA e FoA com os mesmos modelos/N
ficam separados. O `sweep_summary.csv` junta todas as rodadas; as colunas:

| Métrica | O que é |
|---------|---------|
| `accuracy` | fração de respostas numéricas corretas |
| `avg_latency_s` | tempo médio por pergunta |
| `avg_total_tokens` | tokens médios (prompt + geração) |
| `avg_compute_cost` | **custo computacional**: Σ (params_bi × tokens gerados) |
| `master_usage_rate` | fração de perguntas que acionaram o mestre |
| `avg_model_calls` | nº médio de chamadas a modelos por pergunta |

O `compute_cost` é o eixo de custo do estudo: captura que gerar tokens num 32B
"pesa" ~8× gerar no 4B, indo além da simples contagem de tokens.

---

## 6. Estrutura

```
config.py                # catálogo de modelos + parâmetros padrão de rodada
src/
  llm.py                 # backends (HF | mock), LLM e LLMHub (modelos da rodada)
  dataset.py             # GSM8K (com fallback offline)
  metrics.py             # extração da resposta, acerto e agregação (+ custo)
  experiments.py         # geração da grade: variações por protocolo
  runner.py              # roda uma rodada, agrega e salva
  protocols/
    base.py              # Protocol (classe-base) + registry + contabilização
    single_agent.py      # single_agent / single_minion
    minions.py           # delegação SLM → mestre
    debate.py            # debate (voto majoritário)
    mixture_of_agents.py # propostas + agregador
    foa.py               # frota + refinamento + síntese
scripts/
  run_experiment.py      # uma rodada (uma config)
  run_sweep.py           # varredura (um protocolo ou a grade inteira)
  run_sweep.sh           # atalho da grade/protocolo
results/
  runs/<slug>/           # 1 pasta por experimento: raw.json, summary.json, meta.json
  sweep_summary.csv      # tabela mestre (1 linha por experimento)
```

Adicionar um protocolo novo: crie `src/protocols/<nome>.py` com uma classe
decorada com `@register`, importe-a em `src/protocols/__init__.py` e ele já fica
disponível em `--protocols` e no sweep.
