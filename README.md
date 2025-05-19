# {{crew_name}} Crew

Welcome to the {{crew_name}} Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:

```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/agentic_chat/config/agents.yaml` to define your agents
- Modify `src/agentic_chat/config/tasks.yaml` to define your tasks
- Modify `src/agentic_chat/crew.py` to add your own logic, tools and specific args
- Modify `src/agentic_chat/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your flow and begin execution, run this from the root folder of your project:

```bash
crewai run
```

This command initializes the agentic_chat Flow as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The agentic_chat Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the {{crew_name}} Crew or crewAI.

- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.

# AgenticChat

Integration between CopilotKit and CrewAI flows for seamless tool usage in chat interactions.

## Features

- Seamless integration of CopilotKit tools with CrewAI flows
- Automatic tool detection and formatting for LLM consumption
- Event-based handling of tool calls
- Improved response handling for tool usage

## Installation

```bash
pip install -e .
```

## Usage

### Basic Example

Create an instance of `AgenticChatFlow` and use it in your application:

```python
from agentic_chat import AgenticChatFlow, register_tool_call_listener

# Define your tools
tools = [
    {
        "name": "example-tool",
        "description": "An example tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "A parameter"
                }
            },
            "required": ["param"]
        }
    }
]

# Create input structure
kickoff_input = {
    "tools": tools,
    "messages": [
        {
            "role": "user",
            "content": "Hello, can you help me?"
        }
    ]
}

# Register listener for tool calls
register_tool_call_listener()

# Create flow and run
flow = AgenticChatFlow()
result = flow.kickoff(kickoff_input)
```

### Using the CrewAI CLI

You can run the example flows using the CrewAI CLI:

```bash
# Run the main example
crewai run

# Run alternative examples
crewai run agentic_chat.examples:simple_example
crewai run agentic_chat.examples:plot_example
```

## Development

### Project Structure

- `main.py`: Core `AgenticChatFlow` class implementation
- `copilotkit_integration.py`: Underlying integration with CopilotKit
- `entrypoint.py`: Main kickoff function for CLI and direct execution
- `examples.py`: Additional example functions and test cases

### Module Organization

The package is designed to avoid circular imports while maintaining compatibility with various entry points:

1. `copilotkit_integration.py` contains the base integration functionality
2. `main.py` imports from integration and defines the main flow class
3. `entrypoint.py` imports from main to provide the kickoff entry point
4. `examples.py` contains additional examples that import from main

### Running Tests

Test the integration using the provided examples:

```bash
# Run directly
python -m agentic_chat.examples

# Run with CrewAI CLI
crewai run
```

## License

MIT
