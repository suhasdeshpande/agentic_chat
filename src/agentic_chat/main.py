#!/usr/bin/env python
from dotenv import load_dotenv
from crewai.flow import start
from crewai import LLM
import sys
from pydantic import BaseModel
from typing import List, Dict, Any
from crewai.flow import persist
import json

# Import from copilotkit_integration
from agentic_chat.copilotkit_integration import (
    CopilotKitFlow,
    tool_calls_log,
)

# Load environment variables from .env file
load_dotenv()

class AgentInputState(BaseModel):
    """Defines the expected input state for the AgenticChatFlow."""
    messages: List[Dict[str, str]] = [] # Current message(s) from the user
    tools: List[Dict[str, Any]] = [] # CopilotKit tool format: name, description, parameters
    conversation_history: List[Dict[str, str]] = [] # Full conversation history (persisted between runs)


@persist()
class AgenticChatFlow(CopilotKitFlow[AgentInputState]): # Inherit from CopilotKitFlow and use AgentInputState
    """
    The main chat flow that utilizes the CopilotKit state and integration.
    """

    @start()
    def chat(self):
        # pre_chat is called by CopilotKitFlow's kickoff/run logic if needed,
        # or you can ensure it's called if your override kickoff.
        # For now, assuming CopilotKitFlow handles its lifecycle methods.

        # Initialize system prompt
        system_prompt = "You are a helpful assistant."

        # Initialize CrewAI LLM with streaming enabled
        # CrewAI's LLM class expects 'model' as the parameter name
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        # This should now correctly use self.state.messages from AgentInputState
        messages = self.get_message_history(system_prompt=system_prompt)

        # Get available tools using the base class method
        # This should now correctly use self.state.tools from AgentInputState
        tools_definitions = self.get_available_tools()

        # Format tools for OpenAI API using the base class method
        formatted_tools, available_functions = self.format_tools_for_llm(tools_definitions)

        try:
            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)

            response_content = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions if available_functions else None
            )

            # Handle tool responses using the base class method
            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_content, # Pass the text content of the response
                messages=messages, # Original messages sent to LLM
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # ---- Maintain conversation history ----
            # 1. Add the current user message(s) to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            # 2. Add the assistant's response to conversation history
            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)


            return json.dumps({
                "response": final_response,
                "id": self.state.id
            })

        except Exception as e:
            return f"\n\nAn error occurred: {str(e)}\n\n"


def kickoff():
    """Shim function that re-exports kickoff from entrypoint.py to avoid import errors"""
    from agentic_chat.entrypoint import kickoff as entrypoint_kickoff
    return entrypoint_kickoff()

if __name__ == "__main__":
    # Run kickoff for compatibility with crewai run
    sys.exit(kickoff())
