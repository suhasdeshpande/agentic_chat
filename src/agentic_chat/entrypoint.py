#!/usr/bin/env python
"""
Entrypoint module for CrewAI CLI integration and other tools
that expect a kickoff function in a module.

This module intentionally separates imports to avoid circular dependencies.
"""

import sys
from agentic_chat.copilotkit_integration import register_tool_call_listener

def kickoff():
    """
    Main kickoff function that can be used as an entry point for crewai run
    """
    try:
        # Import AgenticChatFlow here to avoid circular imports
        from agentic_chat.main import AgenticChatFlow
        
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
                    "content": "Change background to a nice blue gradient, and tell me a joke."
                }
            ]
        }
        
        # Register event listeners for tool calls
        register_tool_call_listener()
        
        # Start the flow with the input
        agentic_chat_flow = AgenticChatFlow()
        agentic_chat_flow.kickoff(kickoff_input)
        
        # Print summary of tool calls
        print(agentic_chat_flow.get_tools_summary())

        # Return success code to ensure proper exit status
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        # Return failure code for proper error handling
        return 1


if __name__ == "__main__":
    sys.exit(kickoff()) 