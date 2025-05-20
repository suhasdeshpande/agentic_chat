#!/usr/bin/env python
from dotenv import load_dotenv
from crewai.flow import start
from crewai import LLM
import sys
import logging

# Configure logging to print to console
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output to console
    ]
)

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
        # Add super verbose logging at the start to capture exactly what we get
        logger.info("============ ENTERPRISE DEBUG START ============")
        logger.debug(f"Flow ID: {getattr(self, 'id', 'unknown')}")
        logger.debug(f"self dict keys: {list(vars(self).keys())}")

        # Log ALL attributes we can find
        logger.debug(f"Input attribute exists: {hasattr(self, 'input')}")
        logger.debug(f"State attribute exists: {hasattr(self, 'state')}")
        logger.debug(f"Raw input attribute exists: {hasattr(self, '_raw_input')}")

        # Try to dump full raw objects
        if hasattr(self, "input"):
            try:
                logger.debug(f"Input raw dump: {str(self.input)[:1000]}")
            except:
                logger.warning("Failed to dump input")

        if hasattr(self, "state"):
            try:
                logger.debug(f"State raw dump: {str(self.state)[:1000]}")
            except:
                logger.warning("Failed to dump state")

        # Try to access direct attributes possibly available on Enterprise
        for possible_attr in ["threadId", "runId", "tools", "messages", "user_message", "input_message"]:
            if hasattr(self, possible_attr):
                logger.debug(f"Found attribute {possible_attr}: {getattr(self, possible_attr)}")

        # Try to access any messages directly
        if hasattr(self, "input") and isinstance(self.input, dict) and "messages" in self.input:
            logger.debug(f"Messages from input: {self.input['messages']}")

        # Full dump of all variables to see exactly what we're dealing with
        logger.debug("--- VARS DUMP ---")
        for k, v in vars(self).items():
            try:
                logger.debug(f"{k}: {str(v)[:300]}...")
            except:
                logger.debug(f"{k}: <failed to print>")
        logger.info("============ ENTERPRISE DEBUG END ============")

        # Run pre_chat to ensure tools are set
        self.pre_chat()

        # Debug the input when chat is called
        logger.debug(f"CHAT: Input available: {hasattr(self, 'input')}")
        if hasattr(self, "input"):
            if isinstance(self.input, dict):
                logger.debug(f"CHAT: Input keys: {list(self.input.keys())}")
                if "messages" in self.input:
                    logger.debug(f"CHAT: Input messages: {self.input['messages']}")

        # Initialize system prompt
        system_prompt = "You are a helpful assistant."

        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)
        logger.debug(f"CHAT: Messages after get_message_history: {messages}")

        # Get available tools using the base class method
        tools = self.get_available_tools()
        logger.debug(f"CHAT: Tools count: {len(tools)}")

        # Format tools for OpenAI API using the base class method
        formatted_tools, available_functions = self.format_tools_for_llm(tools)

        try:
            # Track if tools were called during this interaction
            tools_called_count = len(tool_calls_log)

            # Call LLM with tools
            logger.debug(f"CHAT: Calling LLM with {len(messages)} messages")
            response = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions
            )
            logger.debug(f"CHAT: LLM response received: {response}")

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
