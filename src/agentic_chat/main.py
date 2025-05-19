#!/usr/bin/env python
import os
from random import randint
from dotenv import load_dotenv

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from crewai import LLM
from copilotkit.crewai import CopilotKitState

# Load environment variables from .env file
load_dotenv()

class AgenticChatFlow(Flow[CopilotKitState]):

    @start()
    def chat(self):
        system_prompt = "You are a helpful assistant."

        print("State: ", self.state)
        
        # Initialize CrewAI LLM with streaming enabled
        # Remove the provider parameter which is causing the error
        llm = LLM(model="gpt-4o", stream=True)
        
        # Prepare messages array
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add existing messages from state if available
        if hasattr(self.state, "messages") and self.state.messages:
            messages.extend(self.state.messages)
        
        # Prepare tools from copilotkit if available
        tools = []
        if hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            tools = self.state.copilotkit.actions
        
        # Use the correct CrewAI method to run the LLM
        # Keep only the essential supported parameters
        response = llm.call(
            messages=messages,
            tools=tools if tools and len(tools) > 0 else None
        )
        
        # Initialize messages list if it doesn't exist
        if not hasattr(self.state, "messages"):
            self.state.messages = []
            
        # Append the new message to the messages in state
        self.state.messages.append(response)
        
        return response


def kickoff():
    agentic_chat_flow = AgenticChatFlow()
    agentic_chat_flow.kickoff()


def plot():
    agentic_chat_flow = AgenticChatFlow()
    agentic_chat_flow.plot()


if __name__ == "__main__":
    kickoff()
