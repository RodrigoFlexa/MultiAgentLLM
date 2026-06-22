"""
Pacote de protocolos.

Importar este pacote registra todos os protocolos no registry. Para adicionar
um novo (ex.: self-consistency), basta criar `meu_protocolo.py` com uma classe
decorada com `@register` e importá-la aqui.
"""
from src.protocols.base import Protocol, available, get_protocol, register

# Importações que disparam o @register de cada protocolo.
from src.protocols import single_agent, minions, debate, mixture_of_agents  # noqa: E402,F401

__all__ = ["Protocol", "available", "get_protocol", "register"]
