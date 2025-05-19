#!/usr/bin/env python
from typing import Dict, Any, List, Optional
import datetime
from crewai.flow import Flow
from crewai import LLM
from crewai.utilities.events import crewai_event_bus
from copilotkit.crewai import CopilotKitState

# Tool calls log for tracking
tool_calls_log = []

# Custom event type for CopilotKit tool calls
class CopilotKitToolCallEvent:
    """Event emitted when a tool call is made through CopilotKit"""
    type = "COPILOTKIT_FRONTEND_TOOL_CALL"
    
    def __init__(self, tool_name: str, args: Dict[str, Any]):
        self.tool_name = tool_name
        self.args = args
        self.timestamp = datetime.datetime.now().isoformat()

# Tool proxy function generator
def create_tool_proxy(tool_name):
    """Creates a proxy function for a tool that emits an event when called"""
    def tool_proxy(**kwargs):
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

class CopilotKitFlow(Flow[CopilotKitState]):
    """
    A base Flow class that extends CopilotKitState and ensures
    tools from the kickoff input are available in self.state.copilotkit.actions
    """
    
    # Store tools at the class level
    _tools_from_input = []
    
    def kickoff(self, state=None, inputs=None):
        """
        Start execution of the flow with the given input state
        
        Args:
            state: The input state (legacy parameter name)
            inputs: The input state (new parameter name)
            
        Returns:
            The result of the flow execution
        """
        # Use inputs parameter if provided, otherwise use state
        actual_input = inputs if inputs is not None else state
        
        # Store tools at the class level for use in pre_chat
        if isinstance(actual_input, dict) and "tools" in actual_input:
            CopilotKitFlow._tools_from_input = actual_input.get("tools", [])
        
        # Call parent's kickoff with the correct parameter
        return super().kickoff(actual_input)

    def pre_chat(self):
        """
        Set tools on state just before chat runs
        """
        if hasattr(self, "state") and hasattr(self.state, "copilotkit") and hasattr(self.state.copilotkit, "actions"):
            if CopilotKitFlow._tools_from_input:
                try:
                    self.state.copilotkit.actions = CopilotKitFlow._tools_from_input
                except Exception as e:
                    pass

    def format_tools_for_llm(self, tools: List[Dict[str, Any]]):
        """
        Format tools for the OpenAI API
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Tuple of (formatted_tools, available_functions)
        """
        formatted_tools = []
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
        
        return formatted_tools, available_functions

    def handle_tool_responses(self, llm: LLM, response: str, messages: List[Dict[str, str]], 
                             tools_called_count: int, follow_up_prompt: Optional[str] = None):
        """
        Handle tool responses and get a follow-up response if needed
        
        Args:
            llm: The LLM instance to use
            response: The initial response from the LLM
            messages: The message history
            tools_called_count: Number of tool calls before the last response
            follow_up_prompt: Custom prompt for follow-up (optional)
            
        Returns:
            The final response to return
        """
        # Check if new tool calls were made during this interaction
        new_tools_called = len(tool_calls_log) > tools_called_count
        
        # Check if a follow-up is needed (tools were called but no substantive content)
        need_followup = new_tools_called and (
            # Response is empty or very short
            not response.strip() or 
            # Or response consists entirely of tool call confirmation messages
            all(f"Tool {call['tool_name']}" in response for call in tool_calls_log[tools_called_count:])
        )
        
        if need_followup:
            # Create a new message array with the tool call responses
            follow_up_messages = messages.copy()
            follow_up_messages.append({"role": "assistant", "content": response})
            
            if follow_up_prompt:
                prompt = follow_up_prompt
            else:
                prompt = "Please provide your complete response now that the tools have been used."
                
            follow_up_messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Call LLM without tools for a final response
            final_response = llm.call(messages=follow_up_messages)
            
            # Combine responses for the state
            combined_response = response + "\n\n" + final_response
            
            return combined_response
        else:
            return response

    def get_tools_summary(self):
        """
        Get a summary of all tool calls made
        
        Returns:
            A string with the tool calls summary
        """
        summary = f"\nTotal tool calls: {len(tool_calls_log)}\n"
        
        for i, call in enumerate(tool_calls_log):
            summary += f"\n[{i+1}] Tool: {call['tool_name']}"
            summary += f"\n    Args: {call['args']}"
            summary += f"\n    Time: {call['timestamp']}\n"
            
        return summary

# Register event listener for tool calls
def register_tool_call_listener():
    """Register an event listener for CopilotKit tool calls"""
    @crewai_event_bus.on(CopilotKitToolCallEvent)
    def on_tool_call_event(source, event):
        print(f"Received CopilotKit tool call event: {event}")
        pass