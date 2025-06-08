# Streamlit UI for Agents Built using AutoGen AgntChat (v0.4+)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AutoGen Version](https://img.shields.io/badge/AutoGen-0.4+-green.svg)](https://microsoft.github.io/autogen/stable/index.html)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)

This project demonstrates how to create a modern web interface for AutoGen AgentChat agents using Streamlit. Unlike most tutorials that focus on AutoGen 0.2, this implementation specifically targets version 0.4 and above (this one uses v0.6) and showcases agent interactions with responses in the Streamlit UI.

## Project Setup

### Prerequisites
- Python 3.10 or higher
- `uv` package manager

### Installation

1. Create a new project directory and set up a Python environment:
```sh
mkdir streamlit-autogen-04
cd streamlit-autogen-04
uv venv
```

2. Install dependencies using the pyproject.toml:
```sh
uv pip install -e .
```

## Project Structure

### Dependencies
The project relies on several key packages:
- `autogen-agentchat>=0.6.1` - Core AutoGen functionality
- `autogen-ext[azure,openai]>=0.6.1` - Extensions for Azure/OpenAI integration
- `streamlit>=1.45.1` - Web interface framework
- `aiohttp>=3.12.11` - Async HTTP client
- `dotenv>=0.9.9` - Environment variable management

### Creating the Streamlit Interface
The interface is built using Streamlit components:
- Header section with project title
- Sidebar for configuration options
- Chat interface with message history
- Input field for user queries

### Custom Agent Implementation
The `TrackableAssistantAgent` class extends AutoGen's `AssistantAgent` with:
- Real-time response tracking
- Message stream processing
- Tool call visualization
- Formatted message display

### Configuration and Authentication
Required credentials:
- GitHub Personal Access Token (PAT)
- Model selection (gpt-4o-mini, gpt-4o, o1-mini, o1)
- Optional: Serper API key for web search capability

### Initialization Components
The code handles:
- Chat history management using Streamlit's session state
- Windows-specific asyncio configuration
- Event loop initialization for async operations
- Message history persistence

### Web Search Tool Integration
Implements a Serper-powered web search tool:
```python
def serper_web_search(query: str) -> str:
    # Implementation details...
```

### Agent Configuration
The agent is initialized with:
- Azure AI model client setup
- Career mentor specialization
- Tool availability based on provided API keys
- Termination conditions for chat completion

### Chat Interaction Flow
1. Validates required credentials
2. Initializes agent if not present
3. Displays chat history
4. Processes user input
5. Streams agent responses
6. Updates UI in real-time

## Usage

1. Start the Streamlit application:
```sh
streamlit run career-mentor-agent.py
```

2. In the sidebar:
   - Enter your GitHub PAT
   - Select your preferred model
   - Optionally add Serper API key for web search capability

3. Start chatting with the career mentor agent in the main interface


## Conclusion

This implementation showcases a modern approach to building AI agent interfaces using Streamlit and AutoGen AgentChat. The architecture supports real-time interactions, tool integration, and extensibility for additional features. The modular design allows for easy modifications and enhancements to suit specific use cases.
