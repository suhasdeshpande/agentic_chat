"""
A simple agentic chat flow.
"""

from crewai.flow.flow import Flow, start
from litellm import completion
from ag_ui_crewai import copilotkit_stream, CopilotKitState

import sys

class AgenticChatFlow(Flow[CopilotKitState]):

    @start()
    async def chat(self):
        system_prompt = "You are a helpful assistant."

        # 1. Run the model and stream the response
        #    Note: In order to stream the response, wrap the completion call in
        #    copilotkit_stream and set stream=True.
        response = await copilotkit_stream(
            completion(

                # 1.1 Specify the model to use
                model="openai/gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *self.state.messages
                ],

                # 1.2 Bind the available tools to the model
                tools=[
                    *self.state.copilotkit.actions,
                ],

                # 1.3 Disable parallel tool calls to avoid race conditions,
                #     enable this for faster performance if you want to manage
                #     the complexity of running tool calls in parallel.
                parallel_tool_calls=False,
                stream=True
            )
        )

        message = response.choices[0].message

        # 2. Append the message to the messages in state
        self.state.messages.append(message)
        return response.choices[0].message.content

def kickoff():
    """Shim function that re-exports kickoff from entrypoint.py to avoid import errors"""
    kickoff_input = {
        "tools": [
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
            }
        ],
        "state": {},
        "messages": [
            {
                "role": "user",
                "content": "Yes"
            }
        ]
    }

    # Start the flow with the input
    agentic_chat_flow = AgenticChatFlow()
    result = agentic_chat_flow.kickoff(kickoff_input)
    print("RESULT", result)

if __name__ == "__main__":
    # Run kickoff for compatibility with crewai run
    sys.exit(kickoff())