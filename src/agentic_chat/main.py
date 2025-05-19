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


class AgenticChatFlow(CopilotKitFlow):
    """
    The main chat flow that utilizes the CopilotKit state
    """
    
    @start()
    def chat(self):
        # Run pre_chat to ensure tools are set
        self.pre_chat()
        
        # Debug the input when chat is called
        print(f"CHAT: Input available: {hasattr(self, 'input')}")
        if hasattr(self, "input"):
            if isinstance(self.input, dict):
                print(f"CHAT: Input keys: {list(self.input.keys())}")
                if "messages" in self.input:
                    print(f"CHAT: Input messages: {self.input['messages']}")
        
        # Initialize system prompt
        system_prompt = "You are a helpful assistant."
        
        # Initialize CrewAI LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)
        
        # Get message history using the base class method
        messages = self.get_message_history(system_prompt=system_prompt)
        print(f"CHAT: Messages after get_message_history: {messages}")
        
        # Get available tools using the base class method  
        tools = self.get_available_tools()
        print(f"CHAT: Tools count: {len(tools)}")
        
        # Format tools for OpenAI API using the base class method
        formatted_tools, available_functions = self.format_tools_for_llm(tools)
        
        try:
            # Track if tools were called during this interaction
            tools_called_count = len(tool_calls_log)
            
            # Call LLM with tools
            print(f"CHAT: Calling LLM with {len(messages)} messages")
            response = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions
            )
            print(f"CHAT: LLM response received: {response}")
            
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
            print(f"CHAT ERROR: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"


def kickoff():
    """Shim function that re-exports kickoff from entrypoint.py to avoid import errors"""
    from agentic_chat.entrypoint import kickoff as entrypoint_kickoff
    return entrypoint_kickoff()

if __name__ == "__main__":
    # Run kickoff for compatibility with crewai run
    sys.exit(kickoff())
