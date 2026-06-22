"""
Camada de modelos.

Expõe uma interface única e simples — `LLM.chat(messages)` — que os protocolos
usam sem saber se por baixo está rodando HuggingFace Transformers numa GPU ou
um backend falso (mock) numa máquina sem GPU.

Toda chamada devolve um `GenerationResult` com o texto E a contabilização
(tokens e latência), que é o que permite comparar custo entre os protocolos.
"""
from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ──────────────────────────────────────────────────────────────────────
# Backends de inferência
# ──────────────────────────────────────────────────────────────────────
class Backend(ABC):
    """Contrato que qualquer motor de inferência precisa cumprir."""

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
    """Backend real, baseado em `transformers`. Carrega modelos sob demanda
    e os mantém em cache (um mesmo modelo nunca é carregado duas vezes)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple] = {}  # name -> (tokenizer, model)

    def _load(self, model_cfg: cfg.ModelConfig):
        if model_cfg.name in self._cache:
            return self._cache[model_cfg.name]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = dict(device_map="auto", torch_dtype=getattr(torch, cfg.TORCH_DTYPE))
        if cfg.MASTER_LOAD_IN_4BIT and model_cfg.name == cfg.MASTER_MODEL.name:
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
            prompt_tokens=int(prompt_len),
            completion_tokens=int(completion_ids.shape[0]),
            latency_s=latency,
        )


class MockBackend(Backend):
    """Backend falso e determinístico. Não baixa nem roda nenhum modelo.

    Serve só para validar o encanamento (LangGraph + protocolos) numa máquina
    sem GPU. Heurística de brinquedo:
      * o minion "tem confiança" em problemas curtos e delega os longos;
      * a resposta numérica é derivada de um hash do enunciado (estável).
    Não mede qualidade de verdade — apenas faz o sistema rodar de ponta a ponta.
    """

    def generate(self, messages, model_cfg, gen_cfg, max_new_tokens=None):
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        user = " ".join(m["content"] for m in messages if m["role"] == "user")
        is_minion = model_cfg.name == cfg.MINION_MODEL.name

        # Pseudo-resposta estável a partir do enunciado.
        digest = int(hashlib.sha1(user.encode()).hexdigest(), 16)
        pseudo_answer = digest % 100

        # Minion delega problemas "difíceis" (heurística: enunciado longo).
        delegate = (
            is_minion
            and cfg.MINIONS.delegate_token in system
            and len(user) > 200
        )
        if delegate:
            text = cfg.MINIONS.delegate_token
        else:
            text = (
                f"Resolvendo passo a passo (mock).\n"
                f"#### {pseudo_answer}"
            )

        time.sleep(0.001)  # latência simbólica
        return GenerationResult(
            text=text,
            model_label=model_cfg.label,
            prompt_tokens=len((system + user).split()),
            completion_tokens=len(text.split()),
            latency_s=0.001,
        )


# ──────────────────────────────────────────────────────────────────────
# LLM: um modelo + um backend, com interface de chat
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
# Hub: cria/serve os LLMs compartilhando um único backend
# ──────────────────────────────────────────────────────────────────────
def _make_backend() -> Backend:
    if cfg.BACKEND == "mock":
        return MockBackend()
    if cfg.BACKEND == "hf":
        return HFBackend()
    raise ValueError(f"Backend desconhecido: {cfg.BACKEND!r} (use 'hf' ou 'mock')")


class LLMHub:
    """Ponto único de acesso aos modelos. Os protocolos pedem `hub.minion`
    e `hub.master`; o backend (real ou mock) é compartilhado entre eles."""

    def __init__(self, backend: Optional[Backend] = None):
        self.backend = backend or _make_backend()
        self.minion = LLM(cfg.MINION_MODEL, self.backend)
        self.master = LLM(cfg.MASTER_MODEL, self.backend)

    def minion_agent(self) -> LLM:
        """Nova instância do minion, para protocolos que rodam várias cópias
        em paralelo (debate, mixture-of-agents) com prompts diferentes sobre
        o mesmo modelo e backend compartilhado."""
        return LLM(cfg.MINION_MODEL, self.backend)
