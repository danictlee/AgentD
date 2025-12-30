import streamlit as st
import os
from dotenv import load_dotenv
import operator
from typing import Annotated, TypedDict

# LangChain & LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load API Key
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'config' not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit_session"}}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Define the State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# Define tools
search_tool = DuckDuckGoSearchRun()

@tool
def search(query: str) -> str:
    """Search the internet for information"""
    return search_tool.run(query)

tools = [search]

# Setup agent (only once)
@st.cache_resource
def setup_agent():
    memory = MemorySaver()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", END: END}
    )
    workflow.add_edge("action", "agent")

    return workflow.compile(checkpointer=memory)

# Initialize agent
if st.session_state.agent is None:
    st.session_state.agent = setup_agent()

# Header
st.title("🤖 AI Research Agent")
st.markdown("*Ask questions and get intelligent answers with internet search*")
st.divider()

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        
        # Show thoughts if available
        if "thoughts" in chat and chat["thoughts"]:
            with st.expander("💭 Thought Process", expanded=False):
                for thought in chat["thoughts"]:
                    if thought["type"] == "tool_call":
                        st.markdown(f"🔧 **Calling tool:** `{thought['tool']}`")
                        st.code(thought["input"], language="json")
                    elif thought["type"] == "tool_result":
                        st.markdown(f"✅ **Tool result received**")

# Chat input
if prompt := st.chat_input("Ask your question..."):
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Researching..."):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            thought_steps = []
            final_result = None
            
            for event in st.session_state.agent.stream(inputs, config=st.session_state.config):
                for node, state in event.items():
                    if state.get("messages"):
                        last_msg = state["messages"][-1]
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            thought_steps.append({
                                "type": "tool_call",
                                "tool": last_msg.tool_calls[0]['name'],
                                "input": str(last_msg.tool_calls[0]['args'])
                            })
                        elif hasattr(last_msg, 'content') and last_msg.content:
                            if node == "action":
                                thought_steps.append({
                                    "type": "tool_result",
                                    "content": last_msg.content[:500]
                                })
                    final_result = state
            
            response_text = final_result["messages"][-1].content if final_result else "No response"
            
            # Display response
            st.markdown(response_text)
            
            # Show thought process
            if thought_steps:
                with st.expander("💭 Thought Process", expanded=False):
                    for thought in thought_steps:
                        if thought["type"] == "tool_call":
                            st.markdown(f"🔧 **Calling tool:** `{thought['tool']}`")
                            st.code(thought["input"], language="json")
                        elif thought["type"] == "tool_result":
                            st.markdown(f"✅ **Tool result received**")
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "thoughts": thought_steps
            })

# Sidebar with info and controls
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI Research Agent uses:
    - 🧠 **LangGraph** for agent orchestration
    - 🔍 **DuckDuckGo** for web search
    - 🤖 **Gemini 2.0 Flash** as the brain
    - 💾 **Memory** to remember context
    """)
    
    st.divider()
    
    # Show conversation stats
    user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    st.metric("Questions Asked", user_messages)
    
    st.divider()
    
    # Reset button
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.config = {"configurable": {"thread_id": f"session_{len(st.session_state.chat_history)}"}}
        st.rerun()
    
    st.divider()
    st.caption("🚀 Powered by LangChain & LangGraph")
