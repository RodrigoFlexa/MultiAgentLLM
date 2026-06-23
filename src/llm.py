"""
Camada de modelos.

Interface única `LLM.chat(messages)`, sem o protocolo saber se por baixo roda
HuggingFace Transformers numa GPU ou um backend falso (mock) sem GPU.

Cada chamada devolve um `GenerationResult` com texto + contabilização (tokens,
latência e `params_b` do modelo que gerou) — base para comparar custo entre
protocolos e entre escolhas de modelo.
"""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import config as cfg

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


# ──────────────────────────────────────────────────────────────────────
# Resultado de uma geração
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GenerationResult:
    text: str
    model_label: str
    params_b: float = 0.0          # params (bi) do modelo que gerou — p/ custo
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def compute_cost(self) -> float:
        """Proxy de custo computacional desta geração: params (bi) × tokens
        gerados. Rodar um 32B por 100 tokens "pesa" ~8x um 4B por 100 tokens."""
        return self.params_b * self.completion_tokens


# ──────────────────────────────────────────────────────────────────────
# Backends de inferência
# ──────────────────────────────────────────────────────────────────────
class Backend(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        model_cfg: cfg.ModelConfig,
        gen_cfg: cfg.GenerationConfig,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        ...


class HFBackend(Backend):
    """Backend real (`transformers`). Carrega modelos sob demanda e os mantém
    em cache — crucial no sweep: o mesmo modelo nunca é recarregado, mesmo
    entre experimentos diferentes que o reutilizem."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple] = {}  # name -> (tokenizer, model)

    def _load(self, model_cfg: cfg.ModelConfig):
        if model_cfg.name in self._cache:
            return self._cache[model_cfg.name]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = dict(device_map="auto", torch_dtype=getattr(torch, cfg.TORCH_DTYPE))
        if model_cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, cfg.TORCH_DTYPE),
                bnb_4bit_quant_type="nf4",
            )
            kwargs.pop("torch_dtype")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
        model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **kwargs)
        model.eval()
        self._cache[model_cfg.name] = (tokenizer, model)
        return tokenizer, model

    def generate(self, messages, model_cfg, gen_cfg, max_new_tokens=None):
        import torch

        tokenizer, model = self._load(model_cfg)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = gen_cfg.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or model_cfg.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=gen_cfg.temperature, top_p=gen_cfg.top_p)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = out[0][prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            model_label=model_cfg.label,
            params_b=model_cfg.params_b,
            prompt_tokens=int(prompt_len),
            completion_tokens=int(completion_ids.shape[0]),
            latency_s=latency,
        )


class MockBackend(Backend):
    """Backend falso e determinístico (sem GPU, sem download). Heurística de
    brinquedo só para validar o encanamento: delega quando o system-prompt pede
    o token de delegação e o enunciado é longo; resposta numérica vem de um hash
    estável do enunciado. Independente de qual modelo está configurado."""

    def generate(self, messages, model_cfg, gen_cfg, max_new_tokens=None):
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        user = " ".join(m["content"] for m in messages if m["role"] == "user")

        digest = int(hashlib.sha1(user.encode()).hexdigest(), 16)
        pseudo_answer = digest % 100

        delegate = cfg.MINIONS.delegate_token in system and len(user) > 200
        if delegate:
            text = cfg.MINIONS.delegate_token
        else:
            text = f"Solving step by step (mock).\n#### {pseudo_answer}"

        time.sleep(0.001)
        return GenerationResult(
            text=text,
            model_label=model_cfg.label,
            params_b=model_cfg.params_b,
            prompt_tokens=len((system + user).split()),
            completion_tokens=len(text.split()),
            latency_s=0.001,
        )


# ──────────────────────────────────────────────────────────────────────
# LLM: um modelo + um backend
# ──────────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self, model_cfg: cfg.ModelConfig, backend: Backend):
        self.model_cfg = model_cfg
        self.backend = backend

    @property
    def label(self) -> str:
        return self.model_cfg.label

    def chat(
        self,
        messages: list[Message],
        gen_cfg: cfg.GenerationConfig = cfg.DETERMINISTIC,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        return self.backend.generate(messages, self.model_cfg, gen_cfg, max_new_tokens)


# ──────────────────────────────────────────────────────────────────────
# Hub: contexto de execução de UMA rodada (modelos + tamanho da frota)
# ──────────────────────────────────────────────────────────────────────
def make_backend() -> Backend:
    if cfg.BACKEND == "mock":
        return MockBackend()
    if cfg.BACKEND == "hf":
        return HFBackend()
    raise ValueError(f"Backend desconhecido: {cfg.BACKEND!r} (use 'hf' ou 'mock')")


class LLMHub:
    """Reúne, para uma rodada, o SLM (minion), o mestre (master) e o tamanho da
    frota (n_minions). Os protocolos pedem `hub.minion`, `hub.master`,
    `hub.minion_agent()` e `hub.n_minions` — sem conhecer os modelos concretos.

    O `backend` pode (e deve, no sweep) ser compartilhado entre vários hubs,
    para reaproveitar os modelos já carregados na VRAM."""

    def __init__(
        self,
        minion_model: Optional[cfg.ModelConfig] = None,
        master_model: Optional[cfg.ModelConfig] = None,
        n_minions: int = cfg.EXPERIMENT.n_minions,
        backend: Optional[Backend] = None,
    ):
        self.backend = backend or make_backend()
        self.minion_model = minion_model or cfg.get_model(cfg.DEFAULT_MINION)
        self.master_model = master_model or cfg.get_model(cfg.DEFAULT_MASTER)
        self.n_minions = n_minions
        self.minion = LLM(self.minion_model, self.backend)
        self.master = LLM(self.master_model, self.backend)

    def minion_agent(self) -> LLM:
        """Uma instância de SLM da frota (mesmo modelo/backend; protocolos rodam
        N delas em paralelo com prompts diferentes)."""
        return LLM(self.minion_model, self.backend)
