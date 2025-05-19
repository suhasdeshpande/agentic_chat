#!/usr/bin/env python
from dotenv import load_dotenv
from crewai.flow import Flow, start
from crewai import LLM
from copilotkit.crewai import CopilotKitState
from crewai.utilities.events import crewai_event_bus
from typing import Dict, Any
import datetime

# Load environment variables from .env file
load_dotenv()

# Custom event type for CopilotKit tool calls
class CopilotKitToolCallEvent:
    """Event emitted when a tool call is made through CopilotKit"""
    type = "COPILOTKIT_FRONTEND_TOOL_CALL"
    
    def __init__(self, tool_name: str, args: Dict[str, Any]):
        self.tool_name = tool_name
        self.args = args
        self.timestamp = datetime.datetime.now().isoformat()

# Tool calls log for tracking
tool_calls_log = []

# Tool proxy function generator
def create_tool_proxy(tool_name):
    """Creates a proxy function for a tool that emits an event when called"""
    def tool_proxy(**kwargs):
        # Log that the tool was called
        print(f"\n🛠️ TOOL CALL: '{tool_name}' with args: {kwargs}")
        
        # Create the event
        event = CopilotKitToolCallEvent(tool_name=tool_name, args=kwargs)
        
        # Add to tool calls log
        tool_calls_log.append({
            "tool_name": tool_name,
            "args": kwargs,
            "timestamp": event.timestamp
        })
        
        # Emit the event
        assert hasattr(crewai_event_bus, "emit")
        crewai_event_bus.emit(None, event=event)
        
        # Return a string response with newlines to ensure proper formatting
        return f"\n\nTool {tool_name} called successfully with parameters: {kwargs}\n\n"
    
    return tool_proxy

class EnhancedCopilotKitFlow(Flow[CopilotKitState]):
    """
    A base Flow class that properly extends CopilotKitState and ensures
    tools from the kickoff input are available in self.state.copilotkit.actions
    """
    
    # Store tools at the class level
    _tools_from_input = []
    
    def kickoff(self, state):
        # Store tools at the class level for use in pre_chat
        if isinstance(state, dict) and "tools" in state:
            EnhancedCopilotKitFlow._tools_from_input = state.get("tools", [])
            print(f"Stored {len(EnhancedCopilotKitFlow._tools_from_input)} tools at class level")
        
        return super().kickoff(state)
        

class AgenticChatFlow(EnhancedCopilotKitFlow):
    """
    The main chat flow that utilizes the enhanced CopilotKit state
    """
    
    # This method runs before chat
    def pre_chat(self):
        # Set tools on state just before chat runs
        if hasattr(self, "state") and hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            if EnhancedCopilotKitFlow._tools_from_input:
                print(f"Setting {len(EnhancedCopilotKitFlow._tools_from_input)} tools in pre_chat")
                try:
                    self.state.copilotkit.actions = EnhancedCopilotKitFlow._tools_from_input
                    print("Tools set successfully")
                except Exception as e:
                    print(f"Error setting tools in pre_chat: {e}")
    
    @start()
    def chat(self):
        # Run pre_chat to ensure tools are set
        self.pre_chat()
        
        # Log the tools available in the state
        print("\n--- Tools in self.state.copilotkit.actions ---")
        if hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            actions = self.state.copilotkit.actions
            print(f"Number of tools available: {len(actions)}")
            for i, tool in enumerate(actions):
                print(f"\nTool {i+1}:")
                print(f"  Name: {tool.get('name')}")
                print(f"  Description: {tool.get('description')}")
                print(f"  Parameters: {tool.get('parameters')}")
        else:
            print("No tools found in self.state.copilotkit.actions")
        print("--- End of tools ---\n")
            
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
            print(f"Using {len(self.state.messages)} messages from state")
            messages.extend(self.state.messages)
        
        # Get tools directly from state
        tools = []
        if hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            tools = self.state.copilotkit.actions
        
        # Format tools for OpenAI API
        formatted_tools = []
        
        # Create function handlers for each tool
        available_functions = {}
        
        for tool in tools:
            if "name" in tool and "parameters" in tool and "description" in tool:
                # Create a properly formatted tool for OpenAI
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                formatted_tools.append(formatted_tool)
                
                # Create a proxy function for this tool
                tool_name = tool["name"]
                available_functions[tool_name] = create_tool_proxy(tool_name)
        
        if formatted_tools:
            print(f"Passing {len(formatted_tools)} tools to LLM with {len(available_functions)} function handlers")
        
        try:
            # Call LLM with tools
            response = llm.call(
                messages=messages,
                tools=formatted_tools if formatted_tools else None,
                available_functions=available_functions
            )
            
            # Initialize messages list if it doesn't exist
            if not hasattr(self.state, "messages"):
                self.state.messages = []
                
            # Append the new message to the messages in state
            self.state.messages.append(response)
            
            return response
        except Exception as e:
            print(f"\n❌ LLM Error: {str(e)}")
            # Return a meaningful error message instead of propagating the exception
            return f"\n\nAn error occurred: {str(e)}\n\n"


def kickoff():
    try:
        kickoff_input = {
            "threadId": "ab3f70ea-5dff-4aa9-a3fa-eb2c1205b29a",
            "runId": "744172c5-edc2-430b-a73b-81c8ee42ad78",
            "tools": [
                {
                    "name": "get-details-before-writing-book",
                    "description": "Get the details of the book before writing it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The topic of the book"
                            }
                        },
                        "required": ["topic"]
                    }
                },
                {
                    "name": "change_background",
                    "description": "Change the background color of the chat. Can be anything that the CSS background attribute accepts. Regular colors, linear of radial gradients etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "background": {
                                "type": "string",
                                "description": "The background. Prefer gradients."
                            }
                        },
                        "required": ["background"]
                    }
                },
                {
                    "name": "flow_finished",
                    "description": "The flow has finished",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ],
            "context": [],
            "forwardedProps": {},
            "state": {},
            "messages": [
                {
                    "id": "ck-bde5d991-261a-46e9-a825-895adcc92103",
                    "role": "user",
                    "content": "Can you write a book about artificial intelligence? Also, please change the background to a nice blue gradient."
                }
            ]
        }
        
        # Print summary of what we're passing to kickoff
        print(f"Starting flow with {len(kickoff_input['tools'])} tools and {len(kickoff_input['messages'])} messages")
        
        # Register event listeners for tool calls
        @crewai_event_bus.on(CopilotKitToolCallEvent)
        def on_tool_call_event(source, event):
            print(f"\n📢 EVENT EMITTED: COPILOTKIT_FRONTEND_TOOL_CALL")
            print(f"  Tool: {event.tool_name}")
            print(f"  Args: {event.args}")
            print(f"  Time: {event.timestamp}")
            
        print("🔌 Event listener registered for CopilotKitToolCallEvent")
        
        # Start the flow with the input
        agentic_chat_flow = AgenticChatFlow()
        result = agentic_chat_flow.kickoff(kickoff_input)
        
        # Print summary of tool calls
        print("\n🔍 TOOL CALLS SUMMARY:")
        print(f"Total tool calls: {len(tool_calls_log)}")
        for i, call in enumerate(tool_calls_log):
            print(f"\n[{i+1}] Tool: {call['tool_name']}")
            print(f"    Args: {call['args']}")
            print(f"    Time: {call['timestamp']}")
        
        # Return success code to ensure proper exit status
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        # Return failure code for proper error handling
        return 1


def plot():
    agentic_chat_flow = AgenticChatFlow()
    agentic_chat_flow.plot()


if __name__ == "__main__":
    # Call kickoff and use the return value as exit code
    import sys
    sys.exit(kickoff())
