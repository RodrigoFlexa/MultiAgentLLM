"""
Pacote de protocolos.

Importar este pacote registra todos os protocolos no registry. Para adicionar
um novo, crie `meu_protocolo.py` com uma classe decorada com `@register` e
importe-a aqui.
"""
from src.protocols.base import Protocol, available, get_protocol, register

# Importações que disparam o @register de cada protocolo.
from src.protocols import (  # noqa: E402,F401
    single_agent, minions, debate, mixture_of_agents, foa, foa_dag,
)

__all__ = ["Protocol", "available", "get_protocol", "register"]
