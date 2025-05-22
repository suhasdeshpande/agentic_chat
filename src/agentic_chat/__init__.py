"""
AgenticChat - CopilotKit integration with CrewAI flows

This package provides integration between CopilotKit and CrewAI flows,
allowing for seamless tool usage in chat interactions.
"""

from agentic_chat.main import AgenticChatFlow
from agentic_chat.entrypoint import kickoff

__all__ = [
    'AgenticChatFlow',
    'kickoff'
]
