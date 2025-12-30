# 🤖 AgentD - AI Research Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.2.0-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit)

**An intelligent AI-powered research assistant that searches the web and answers your questions with context awareness**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 📖 Overview

**AgentD** is a cutting-edge conversational AI agent built with **LangGraph** and **LangChain** that combines the power of large language models with real-time web search capabilities. Ask any question, and AgentD will research the topic, synthesize information, and provide comprehensive answers while maintaining conversation context.

### ✨ Key Highlights

- 🧠 **Smart Orchestration**: Built on LangGraph for complex agent workflows
- 🔍 **Web Search Integration**: Real-time internet access via DuckDuckGo
- 💬 **Conversational Memory**: Remembers context across multiple questions
- 🎨 **Beautiful UI**: Clean Streamlit interface with chat history
- 🚀 **Powered by Gemini 2.0**: Uses Google's latest AI model
- 🔧 **Transparent Reasoning**: See the agent's thought process

---

## 🌟 Features

### Core Capabilities

- **🔎 Intelligent Research**: Automatically searches the web for accurate, up-to-date information
- **💭 Thought Process Visualization**: View tool calls and reasoning steps
- **🧵 Multi-turn Conversations**: Ask unlimited follow-up questions with full context
- **📊 Conversation Stats**: Track your interaction history
- **🔄 Session Management**: Start fresh conversations anytime
- **⚡ Real-time Streaming**: See responses as they're generated

### Technical Features

- **State Management**: LangGraph StateGraph for robust agent orchestration
- **Memory Persistence**: MemorySaver checkpointer for conversation continuity
- **Tool Binding**: Seamless integration of external tools (search)
- **Error Handling**: Graceful degradation and error recovery
- **Modular Architecture**: Clean separation of concerns

---

## 🎬 Demo

```bash
User: What is the latest news about OpenAI Sora?

🤖 Agent: 
🔍 Researching... 
🔧 Tool: DuckDuckGo Search
📊 Processing results...

The latest news about OpenAI Sora includes...
[Comprehensive answer with sources]
```

### 💭 Thought Process Example

```
[agent] → 🧠 Analyzing query
[action] → 🔧 Calling tool: search
[action] → ✅ Tool result received
[agent] → 📝 Synthesizing answer
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Google API Key (for Gemini)
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/AgentD.git
cd AgentD
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Run the application**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📚 Usage

### Basic Usage

1. **Launch the app**: Run `streamlit run app.py`
2. **Ask a question**: Type your query in the chat input
3. **View the response**: See the agent's answer with sources
4. **Ask follow-ups**: Continue the conversation naturally
5. **Check thought process**: Expand to see how the agent reasoned

### Example Questions

```
💡 "What are the latest developments in quantum computing?"
💡 "Explain the impact of AI on healthcare"
💡 "What happened in the tech world this week?"
💡 "Compare Python and JavaScript for web development"
```

### Advanced Features

- **View Thought Process**: Click the expander to see tool calls
- **Reset Conversation**: Use the "New Conversation" button in sidebar
- **Track Progress**: Monitor question count in the sidebar

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   Streamlit UI  │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LangGraph     │
│   Agent Core    │
├─────────────────┤
│ • State Mgmt    │
│ • Tool Binding  │
│ • Memory        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Gemini │ │ DuckDuck │
│  LLM   │ │   Go     │
└────────┘ └──────────┘
```

### Components

#### 1. **Agent Core** (`LangGraph`)
- **StateGraph**: Manages agent workflow
- **Nodes**: `agent` (LLM) and `action` (tools)
- **Edges**: Conditional routing logic

#### 2. **Memory System** (`MemorySaver`)
- Persistent conversation history
- Thread-based session management
- Context preservation across queries

#### 3. **Tools**
- **DuckDuckGo Search**: Web search capability
- Extensible tool framework for future additions

#### 4. **Frontend** (`Streamlit`)
- Chat interface with history
- Thought process visualization
- Session management UI

---

## 📁 Project Structure

```
AgentD/
├── app.py                 # Main Streamlit application
├── main.py               # CLI version (alternative interface)
├── streamlit_app.py      # Legacy Streamlit version
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph |
| **LLM Integration** | LangChain |
| **Language Model** | Google Gemini 2.0 Flash |
| **Web Search** | DuckDuckGo Search API |
| **Frontend** | Streamlit |
| **Memory** | LangGraph MemorySaver |
| **Language** | Python 3.12 |

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini | ✅ Yes |

### Model Configuration

The agent uses `gemini-2.0-flash` by default. To change:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Change model here
    temperature=0              # Adjust creativity
)
```
<div align="center">
</div>
