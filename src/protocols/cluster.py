"""
Mecanismo de cluster do FoA (reutilizado pelo `foa` e por subtarefas complexas
do `foa_dag`).

Ideia (reflexão crítica + nota):
  * Rascunho: cada membro resolve de forma independente.
  * A cada passo de refinamento, DUAS partes:
      1) CRÍTICA/AUTOCRÍTICA: o membro vê os rascunhos dos pares E o seu, e os
         analisa com honestidade — sem tratar nenhuma resposta (nem a própria)
         como verdade. Dá uma NOTA (0–10) a cada par e extrai insights.
      2) RE-RESOLUÇÃO: usando os insights, resolve a questão de novo do zero.
  * Para cedo se a frota atingir CONSENSO numérico.
  * Sem consenso, vence a resposta do membro com a MAIOR NOTA ACUMULADA.

Limites (config): cluster de até `max_size` membros e até `max_steps` passos.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import config as cfg
from src.llm import GenerationResult, LLMHub
from src.metrics import extract_final_number
from src.protocols.base import combine_usage


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
def _ctx_block(context: str) -> str:
    return f"\n\nRelevant context / prior results:\n{context}" if context else ""


def _draft_msgs(problem: str, context: str) -> list[dict]:
    system = (
        "You are a member of a small expert team solving a problem. Work it out "
        "independently and carefully, showing each reasoning step, and end with "
        "a line '#### <final number>'."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Problem:\n{problem}{_ctx_block(context)}"}]


def _critique_msgs(problem: str, context: str, i: int, answers: list[str]) -> list[dict]:
    own = answers[i]
    peers = "\n\n".join(f"[Agent {j+1}]\n{a}" for j, a in enumerate(answers) if j != i)
    system = (
        "You are a rigorous, intellectually honest reviewer in a problem-solving "
        "team. Do NOT assume your own draft is correct, and do NOT assume any "
        "teammate is correct. Treat every draft — including yours — as a "
        "hypothesis that may contain errors. Check each one's logic and "
        "arithmetic step by step, expose mistakes, hidden assumptions and "
        "disagreements, and be candid."
    )
    user = (
        f"Problem:\n{problem}{_ctx_block(context)}\n\n"
        f"Your own current draft:\n{own}\n\n"
        f"Your teammates' drafts:\n{peers}\n\n"
        "Do two things:\n"
        "1) Give each teammate an integer score from 0 to 10 for how correct and "
        "well-reasoned their solution is (0 = clearly wrong, 10 = clearly correct "
        "and rigorous).\n"
        "2) Summarize the key insights from your review: what looks right, what "
        "looks wrong, and what must be fixed.\n"
        "Output JSON ONLY, e.g.: "
        '{"scores": {"agent_2": 7, "agent_3": 3}, "insights": "..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _resolve_msgs(problem: str, context: str, own: str, insights: str) -> list[dict]:
    system = (
        "You are solving the problem again, now using the critical insights you "
        "gathered while reviewing your team. Do not blindly trust any earlier "
        "draft (including your own). Re-derive the answer from scratch, fixing "
        "the mistakes you identified. Think step by step and ALWAYS end with a "
        "line '#### <final number>'."
    )
    user = (
        f"Problem:\n{problem}{_ctx_block(context)}\n\n"
        f"Your previous draft:\n{own}\n\n"
        f"Critical insights from the team discussion:\n{insights}\n\n"
        "Now produce your improved, carefully reasoned solution."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ──────────────────────────────────────────────────────────────────────
# Parsing das notas / convergência
# ──────────────────────────────────────────────────────────────────────
def _parse_critique(text: str, size: int, self_idx: int) -> tuple[str, dict[int, float]]:
    """Devolve (insights, {member_idx: nota}). Robusto: tenta JSON e cai para
    regex caso o modelo não devolva JSON limpo."""
    insights, scores = "", {}
    try:
        a, b = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[a:b])
        insights = str(data.get("insights", "")).strip()
        raw = data.get("scores", {}) or {}
        for k, v in raw.items():
            m = re.search(r"(\d+)", str(k))
            if m and isinstance(v, (int, float)):
                idx = int(m.group(1)) - 1            # "agent_2" -> índice 1
                if 0 <= idx < size and idx != self_idx:
                    scores[idx] = float(v)
    except Exception:
        pass
    if not scores:  # regex de resgate: "agent_2": 7  /  agent 2 = 7
        for k, v in re.findall(r"agent[_ ]?(\d+)\D{0,6}?(\d{1,2})", text, flags=re.I):
            idx = int(k) - 1
            if 0 <= idx < size and idx != self_idx:
                scores[idx] = float(v)
    if not insights:
        insights = text.strip()[:600]
    return insights, scores


def _all_agree(answers: list[str]) -> bool:
    nums = [extract_final_number(a) for a in answers]
    valid = [n for n in nums if n is not None]
    return len(valid) >= 2 and len(set(valid)) == 1


# ──────────────────────────────────────────────────────────────────────
# Resultado + execução
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ClusterResult:
    answer: str
    agent_answers: list[str]
    scores: list[float]
    converged: bool
    rounds: int
    usage: dict
    calls: list = field(default_factory=list)


def run_cluster(hub: LLMHub, problem: str, context: str = "",
                max_size: Optional[int] = None, max_steps: Optional[int] = None,
                agent_ids: Optional[list[int]] = None) -> ClusterResult:
    max_size = max_size or cfg.CLUSTER.max_size
    max_steps = max_steps or cfg.CLUSTER.max_steps
    size = max(1, min(max_size, hub.n_minions))
    if agent_ids is None:
        agent_ids = list(range(size))

    gens: list[GenerationResult] = []
    calls: list[dict] = []

    def rec(gen, member, phase):
        gens.append(gen)
        calls.append({"agent": agent_ids[member], "phase": phase,
                      "completion_tokens": gen.completion_tokens,
                      "prompt_tokens": gen.prompt_tokens,
                      "compute_cost": gen.compute_cost,
                      "latency_s": gen.latency_s})

    # passo 1 — rascunho independente
    answers = []
    for i in range(size):
        g = hub.minion_agent().chat(_draft_msgs(problem, context), gen_cfg=cfg.CREATIVE)
        answers.append(g.text); rec(g, i, "draft")

    scores = [0.0] * size
    converged = _all_agree(answers)
    rounds, step = 1, 1

    while step < max_steps and not converged and size >= 2:
        # parte 1 — crítica + nota
        insights = [""] * size
        for i in range(size):
            g = hub.minion_agent().chat(_critique_msgs(problem, context, i, answers),
                                        gen_cfg=cfg.CREATIVE)
            rec(g, i, "critique")
            ins, sc = _parse_critique(g.text, size, i)
            insights[i] = ins
            for j, val in sc.items():
                scores[j] += val
        # parte 2 — re-resolver com os insights
        new = []
        for i in range(size):
            g = hub.minion_agent().chat(_resolve_msgs(problem, context, answers[i], insights[i]),
                                        gen_cfg=cfg.CREATIVE)
            new.append(g.text); rec(g, i, "resolve")
        answers = new
        rounds += 1
        converged = _all_agree(answers)
        step += 1

    # seleção da resposta do cluster
    nums = [extract_final_number(a) for a in answers]
    valid = [(i, n) for i, n in enumerate(nums) if n is not None]
    if converged and valid:
        win = valid[0][0]
    elif any(s > 0 for s in scores):
        win = max(range(size), key=lambda i: scores[i])      # maior nota acumulada
    elif valid:
        c = Counter(n for _, n in valid)
        top = max(c.values())
        topnums = [n for n, k in c.items() if k == top]
        first = {n: next(i for i, m in valid if m == n) for n in topnums}
        win = first[min(topnums, key=lambda n: first[n])]
    else:
        win = 0

    return ClusterResult(
        answer=answers[win], agent_answers=answers, scores=scores,
        converged=converged, rounds=rounds,
        usage=combine_usage(gens, is_master=False), calls=calls,
    )


def agent_costs_from_calls(calls: list[dict]) -> list[dict]:
    """Agrega os registros de chamada por agente (só índices inteiros, i.e. os
    SLMs da frota) — material do `agent_costs.json`."""
    ids = sorted({c["agent"] for c in calls if isinstance(c["agent"], int)})
    rows = []
    for a in ids:
        cs = [c for c in calls if c["agent"] == a]
        rows.append({
            "agent": a,
            "n_calls": len(cs),
            "completion_tokens": sum(c["completion_tokens"] for c in cs),
            "prompt_tokens": sum(c["prompt_tokens"] for c in cs),
            "compute_cost": sum(c["compute_cost"] for c in cs),
            "latency_s": sum(c["latency_s"] for c in cs),
        })
    return rows
