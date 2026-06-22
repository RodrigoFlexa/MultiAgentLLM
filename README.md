# MultiAgentLLM — Comparando protocolos de agentes (Single vs. Minions vs. MoA vs. Debate)

Bancada de testes para comparar diferentes formas de organizar LLMs/SLMs no mesmo
problema e medir, de forma automática, **qualidade × custo × latência**.

Os protocolos comparados:

| Protocolo | Ideia | Modelos envolvidos |
|-----------|-------|--------------------|
| **Single Minion** (piso) | O SLM resolve sozinho. Referência de qualidade mínima e custo mínimo. | só o **minion** (SLM) |
| **Single Agent** (teto) | O melhor modelo resolve sozinho, com raciocínio passo a passo. | só o **mestre** (LLM grande) |
| **Minions** (delegação) | O SLM resolve o que consegue; quando não tem confiança, delega ao LLM. | **minion** (SLM) + **mestre** sob demanda |
| **Mixture-of-Agents** | N minions propõem de forma independente; o mestre agrega e sintetiza a final. | N× **minion** + **mestre** agregador |
| **Debate** (clássico) | N cópias do *mesmo* SLM respondem e revisam vendo as respostas das outras; a final sai por **voto majoritário**, sem juiz. | N× **minion** |

Os dois "agente sozinho" saem do mesmo grafo, mudando só qual modelo é usado —
servem de piso e teto para ler os protocolos do meio.

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
    base.py             # Protocol (classe-base) + registry + estado de contabilização
    single_agent.py     # baselines (piso/teto): SLM ou LLM sozinho
    minions.py          # delegação com nó condicional
    mixture_of_agents.py# N propostas independentes + agregação pelo mestre
    debate.py           # debate clássico: N cópias do SLM + voto majoritário
runner.py            # roda os protocolos, agrega e salva os resultados
scripts/
  run_experiment.py  # ponto de entrada (CLI)
results/             # saídas: raw_<protocolo>.json, summary.json/.csv
```

### Como cada grafo funciona

**Single Agent / Single Minion** — `START → solve → END`. Uma chamada ao mestre
(ou ao minion, no caso do piso).

**Minions** — o minion responde; um nó condicional lê a saída:

```
START → minion ──(delegou?)──┬── sim → master → END
                             └── não ───────────→ END
```

A decisão de delegar usa **auto-avaliação**: o minion é instruído a responder
apenas `<DELEGAR>` quando estiver inseguro (token configurável em `config.py`).

**Mixture-of-Agents** — `START → propose → aggregate → END`. N minions resolvem
em paralelo (diversidade por temperatura); o mestre lê todas as propostas e
sintetiza a resposta final, sem rodadas adversariais entre os proposers.

**Debate** (clássico "society of minds", Du et al. 2023, arXiv:2305.14325) —
loop de rodadas seguido da apuração do voto:

```
START → debate ──(round < N)──┐
          ▲                    │
          └────────────────────┘
        debate ──(round == N)──→ tally → END
```

São N cópias do **mesmo** modelo (sem personas, sem juiz). Na rodada 0 cada
agente responde de forma independente; nas seguintes, cada um revisa a própria
resposta vendo as respostas dos outros. No fim, o nó `tally` decide a resposta
final por **voto majoritário** sobre o número de cada agente (desempate
determinístico pelo agente de menor índice).

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

# Experimento real na GPU (200 perguntas, conjunto padrão de protocolos):
python scripts/run_experiment.py --gpu 0 --n 200

# Subconjunto de protocolos (inclui o debate clássico, fora do padrão):
python scripts/run_experiment.py --n 200 --protocols single_agent minions debate

# Teste de fumaça SEM GPU e SEM baixar modelos (valida todo o encanamento):
MULTIAGENT_BACKEND=mock python scripts/run_experiment.py --n 10
```

O **backend mock** é determinístico e não usa modelo nenhum — serve para
verificar, numa máquina sem GPU, que os grafos, a delegação, o voto e a
contabilização funcionam de ponta a ponta. Ele não mede qualidade real.

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
extra.

**Mixture-of-Agents** — qualidade via síntese. Vantagem: o mestre agrega várias
propostas baratas do SLM e pode corrigir erros pontuais, sem o custo de rodadas
adversariais. Desvantagem: ainda paga uma chamada do modelo grande por pergunta
(o agregador), então custa mais que o Minions quando este não delega.

**Debate (voto majoritário, sem juiz)** — qualidade via redundância **barata**:
não usa o modelo grande, só N cópias do SLM. Vantagem: várias tentativas
independentes que se revisam podem corrigir erros que uma só cometeria, a custo
de tokens baratos. Custo: N agentes × R rodadas de chamadas ao SLM por pergunta
(no default, 3 × 2 = 6 chamadas) — mais lento que os baselines. Risco: cópias
idênticas podem convergir para o **mesmo erro** (diversidade real limitada), e o
voto majoritário não cria conhecimento novo se a maioria já erra.

O experimento existe justamente para colocar números nesse trade-off: quanto de
acurácia o **Minions** preserva gastando uma fração do mestre, se o **MoA**
compensa a chamada extra do agregador, e se o **Debate** entre SLMs baratos
chega perto do mestre sozinho.

---

## Estendendo (adicionar um novo protocolo)

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
