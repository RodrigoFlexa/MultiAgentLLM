# MultiAgentLLM — Comparando protocolos de agentes (Single vs. Minions vs. Debate)

Bancada de testes para comparar três formas de organizar LLMs/SLMs no mesmo
problema e medir, de forma automática, **qualidade × custo × latência**.

Os três protocolos comparados:

| Protocolo | Ideia | Modelos envolvidos |
|-----------|-------|--------------------|
| **Single Agent** (baseline) | O melhor modelo resolve sozinho, com raciocínio passo a passo. | só o **mestre** (LLM grande) |
| **Minions** (delegação) | O SLM resolve o que consegue; quando não tem confiança, delega ao LLM. | **minion** (SLM) + **mestre** sob demanda |
| **Debate** | Vários SLMs com personas debatem e criticam-se; um juiz (LLM) decide. | N× **minion** + **mestre** como juiz |

A tarefa de avaliação é o **GSM8K** (matemática escolar): a resposta final é
sempre um número, então o acerto é verificado por script, sem juiz humano.

---

## Arquitetura

Orquestração em **LangGraph** (cada protocolo é um grafo de estados) e modelos
locais via **HuggingFace Transformers**. As duas camadas são independentes: o
grafo nunca sabe qual modelo está por baixo, e o modelo nunca sabe em qual
protocolo está. Isso mantém o código modular e fácil de estender.

```
config.py            # modelos, dataset, nº de rodadas, seed — tudo num lugar só
src/
  llm.py             # Backend (HF Transformers | Mock) + wrapper LLM + LLMHub
  dataset.py         # carrega GSM8K (com fallback offline) e extrai a resposta-ouro
  metrics.py         # extrai o número final, verifica acerto, agrega métricas
  protocols/
    base.py          # Protocol (classe-base) + registry + estado de contabilização
    single_agent.py  # baseline
    minions.py       # delegação com nó condicional
    debate.py        # debate em loop + juiz
runner.py            # roda os protocolos, agrega e salva os resultados
scripts/
  run_experiment.py  # ponto de entrada (CLI)
results/             # saídas: raw_<protocolo>.json, summary.json/.csv
```

### Como cada grafo funciona

**Single Agent** — `START → solve → END`. Uma chamada ao mestre.

**Minions** — o minion responde; um nó condicional lê a saída:

```
START → minion ──(delegou?)──┬── sim → master → END
                             └── não ───────────→ END
```

A decisão de delegar usa **auto-avaliação**: o minion é instruído a responder
apenas `<DELEGAR>` quando estiver inseguro (token configurável em `config.py`).

**Debate** — loop controlado por contador de rodadas, depois o juiz:

```
START → debate ──(round < N)──┐
          ▲                    │
          └────────────────────┘
        debate ──(round == N)──→ judge → END
```

Na rodada 0 os debatedores resolvem de forma independente; nas seguintes, cada
um vê as respostas dos outros e revisa a sua. O juiz (mestre) lê tudo e decide.

---

## Modelos e hardware

Configurados em `config.py` (troque o `name` para usar outros pesos):

- **Minion (SLM):** `Qwen/Qwen2.5-3B-Instruct` — rápido e leve.
- **Mestre (LLM ≤ 20B):** `Qwen/Qwen2.5-14B-Instruct` — forte em raciocínio
  matemático e cabe folgado numa **A100** (≈ 28 GB em bfloat16).

Na A100 de 80 GB os dois modelos ficam residentes ao mesmo tempo, sem
recarregamento entre protocolos (o `LLMHub` mantém cada modelo em cache). Se a
VRAM apertar, ative `MASTER_4BIT=1` para carregar o mestre em 4-bit.

> Família Qwen2.5 escolhida por ser forte em GSM8K e ter SLM e LLM ≤ 20B na
> mesma família. Para reproduzir o setup original da conversa, é só trocar por
> `meta-llama/Meta-Llama-3-8B-Instruct` (minion) e um modelo da casa dos 14B
> como mestre.

---

## Como rodar

```bash
pip install -r requirements.txt

# Experimento real na GPU (200 perguntas, os 3 protocolos):
python scripts/run_experiment.py --n 200

# Subconjunto de protocolos:
python scripts/run_experiment.py --n 200 --protocols single_agent minions

# Teste de fumaça SEM GPU e SEM baixar modelos (valida todo o encanamento):
MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --n 10
```

O **backend mock** é determinístico e não usa modelo nenhum — serve para
verificar, numa máquina sem GPU, que os grafos, a delegação e a contabilização
funcionam de ponta a ponta. Ele não mede qualidade real.

---

## O que é medido

Por pergunta (`results/raw_<protocolo>.json`) e agregado
(`results/summary.csv`):

- **accuracy** — fração de respostas numéricas corretas.
- **avg_latency_s** — tempo médio por pergunta (soma de todas as chamadas).
- **avg_total_tokens** — tokens médios (prompt + geração).
- **avg_master_tokens** — tokens gerados pelo LLM grande, o custo "caro".
- **master_usage_rate** — fração de perguntas que acionaram o mestre.
- **avg_model_calls** — número médio de chamadas a modelos por pergunta.

A contabilização é acumulada automaticamente pelo grafo: os campos de uso no
estado do LangGraph usam *reducers* de soma, então cada nó só precisa devolver
sua própria delta.

---

## Vantagens e desvantagens esperadas

Análise qualitativa do que cada protocolo tende a entregar (a confirmar com a
rodada real na A100):

**Single Agent (mestre sozinho)** — teto de qualidade e a referência de custo
máximo. Vantagem: melhor acurácia bruta e implementação trivial. Desvantagem:
todo problema, fácil ou difícil, paga o preço do modelo grande (latência e
tokens caros em 100% dos casos).

**Minions (delegação)** — a aposta de custo-benefício. Vantagem: se o SLM
resolve sozinho a maioria dos problemas fáceis, a `master_usage_rate` cai bem
abaixo de 100%, derrubando latência média e custo enquanto a acurácia fica
próxima da do mestre sozinho. Desvantagens: depende da **calibração da
delegação** — se o minion for confiante demais, erra sem delegar (perde
acurácia); se for cauteloso demais, delega tudo e vira o baseline com overhead
extra. A auto-avaliação por token é simples, mas imperfeita.

**Debate** — a aposta de qualidade via redundância. Vantagem: múltiplas
perspectivas podem corrigir erros que um único SLM cometeria, às vezes
superando o mestre sozinho em problemas que se beneficiam de verificação
cruzada. Desvantagem clara: **latência e custo explodem** — são N debatedores ×
R rodadas + 1 chamada de juiz por pergunta (no default, 5 chamadas). É o
protocolo mais caro e mais lento; só compensa se o ganho de acurácia justificar.

O experimento existe justamente para colocar números nesse trade-off:
quanto de acurácia o **Minions** preserva gastando uma fração do mestre, e se o
**Debate** paga o próprio custo em acurácia.

---

## Estendendo (adicionar um 4º protocolo)

1. Crie `src/protocols/meu_protocolo.py` com uma classe que herde de `Protocol`,
   implemente `build_graph`, `initial_state` e `extract`, e decore com
   `@register`.
2. Importe-a em `src/protocols/__init__.py`.
3. Adicione o nome em `EXPERIMENT.protocols` (em `config.py`) ou passe via
   `--protocols`.

Nada mais muda: runner, métricas e relatórios já funcionam para qualquer
protocolo registrado. Candidatos naturais: *self-consistency* (votação por
maioria de N amostras do SLM) e *self-reflection* (o modelo revisa a própria
resposta uma vez).
