#!/usr/bin/env python
from dotenv import load_dotenv
from crewai.flow import start
from crewai import LLM
import sys

# Import from copilotkit_integration
from agentic_chat.copilotkit_integration import (
    CopilotKitFlow,
    tool_calls_log,
)

# Load environment variables from .env file
load_dotenv()

# Re-export kickoff from entrypoint.py
def kickoff():
    """Shim function that re-exports kickoff from entrypoint.py to avoid import errors"""
    from agentic_chat.entrypoint import kickoff as entrypoint_kickoff
    return entrypoint_kickoff()

class AgenticChatFlow(CopilotKitFlow):
    """
    The main chat flow that utilizes the CopilotKit state
    """
    
    @start()
    def chat(self):
        # Run pre_chat to ensure tools are set
        self.pre_chat()
        
        # Initialize system prompt
        system_prompt = "You are a helpful assistant."
        
        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)
        
        # Prepare messages array
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add existing messages from state if available
        if hasattr(self.state, "messages") and self.state.messages:
            messages.extend(self.state.messages)
        
        # Get tools directly from state
        tools = []
        if hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            tools = self.state.copilotkit.actions
        
        # Format tools for OpenAI API using the imported method
        formatted_tools, available_functions = self.format_tools_for_llm(tools)
        
        try:
            # Track if tools were called during this interaction
            tools_called_count = len(tool_calls_log)
            
            # Call LLM with tools
            response = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions
            )
            
            # Handle tool responses using the imported method
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
            return f"\n\nAn error occurred: {str(e)}\n\n"


if __name__ == "__main__":
    # Run kickoff for compatibility with crewai run
    sys.exit(kickoff())
