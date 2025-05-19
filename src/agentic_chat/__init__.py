"""
AgenticChat - CopilotKit integration with CrewAI flows

This package provides integration between CopilotKit and CrewAI flows,
allowing for seamless tool usage in chat interactions.
"""

from agentic_chat.main import AgenticChatFlow
from agentic_chat.entrypoint import kickoff
from agentic_chat.copilotkit_integration import (
    CopilotKitFlow,
    CopilotKitToolCallEvent,
    register_tool_call_listener,
    tool_calls_log,
    create_tool_proxy
)

__all__ = [
    'AgenticChatFlow',
    'CopilotKitFlow',
    'CopilotKitToolCallEvent',
    'register_tool_call_listener',
    'tool_calls_log',
    'create_tool_proxy',
    'kickoff'
]
