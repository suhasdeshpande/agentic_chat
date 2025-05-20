#!/usr/bin/env python
from dotenv import load_dotenv
from crewai.flow import start
from crewai import LLM
import sys
import logging
from agentic_chat.copilotkit_integration import register_tool_call_listener


logger = logging.getLogger(__name__)

# Import from copilotkit_integration
from agentic_chat.copilotkit_integration import (
    CopilotKitFlow,
    tool_calls_log,
)

# Load environment variables from .env file
load_dotenv()


class AgenticChatFlow(CopilotKitFlow):
    """
    The main chat flow that utilizes the CopilotKit state
    """

    @start()
    def chat(self):
        register_tool_call_listener()

        # Run pre_chat to ensure tools are set
        self.pre_chat()

        # Initialize system prompt
        system_prompt = "You are a helpful assistant."

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)
        logger.info(f"CHAT: Messages after get_message_history: {messages}")

        # Get available tools using the base class method
        tools = self.get_available_tools()
        logger.info(f"CHAT: Tools count: {len(tools)}")

        # Format tools for OpenAI API using the base class method
        formatted_tools, available_functions = self.format_tools_for_llm(tools)

        logger.info(f"CHAT: Formatted tools: {formatted_tools}")

        try:
            # Track if tools were called during this interaction
            tools_called_count = len(tool_calls_log)

            response = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions
            )

            logger.info(f"CHAT: LLM response received: {response}")

            # Handle tool responses using the base class method
            response = self.handle_tool_responses(
                llm=llm,
                response=response,
                messages=messages,
                tools_called_count=tools_called_count
            )

            # Initialize messages list if it doesn't exist
            if not hasattr(self.state, "messages"):
                self.state.messages = []

            # Append the new message to the messages in state
            self.state.messages.append(response)

            return response

        except Exception as e:
            logger.error(f"CHAT ERROR: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"


def kickoff():
    """Shim function that re-exports kickoff from entrypoint.py to avoid import errors"""
    from agentic_chat.entrypoint import kickoff as entrypoint_kickoff
    return entrypoint_kickoff()

if __name__ == "__main__":
    # Run kickoff for compatibility with crewai run
    sys.exit(kickoff())
