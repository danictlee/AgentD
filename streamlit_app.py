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

# Initialize Memory Saver
memory = MemorySaver()

# 1. Define the "State"
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 2. Define tools
search_tool = DuckDuckGoSearchRun()

@tool
def search(query: str) -> str:
    """Search the internet for information"""
    return search_tool.run(query)

tools = [search]

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 AgentD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .stTitle {
            text-align: center;
            color: #1f77b4;
        }
        .thought-process {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
        }
        .agent-response {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #0d7c66;
        }
        .tool-call {
            background-color: #fff4e6;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("# 🤖 AgentD")
st.markdown("*Ask questions and get intelligent answers with internet search capabilities*")
st.divider()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'config' not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit_session"}}
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'messages_count' not in st.session_state:
    st.session_state.messages_count = 0

# Setup the agent (only once)
def setup_agent():
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

# Setup agent
if st.session_state.agent is None:
    st.session_state.agent = setup_agent()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔍 Initial Question")
    initial_query = st.text_input(
        "What would you like to research?",
        placeholder="e.g., What is the latest news about AI?",
        key="initial_query_input"
    )

with col2:
    st.markdown("### ℹ️ Info")
    st.info("""
    **How it works:**
    1. Enter your question above
    2. The agent will search the web
    3. You'll see the thought process
    4. Ask follow-up questions below
    """)

st.divider()

# Process initial query
if initial_query and not st.session_state.conversation_started:
    st.session_state.conversation_started = True
    st.session_state.initial_query = initial_query
    st.rerun()

if st.session_state.conversation_started and 'initial_query' in st.session_state:
    query = st.session_state.initial_query
    
    with st.spinner(f"🔍 Researching: {query}..."):
        st.markdown("### 💭 Thought Process")
        
        thought_container = st.container()
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        final_result = None
        thought_steps = []
        
        for event in st.session_state.agent.stream(inputs, config=st.session_state.config):
            for node, state in event.items():
                if state.get("messages"):
                    last_msg = state["messages"][-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        thought_steps.append({
                            "type": "tool_call",
                            "tool": last_msg.tool_calls[0]['name'],
                            "input": last_msg.tool_calls[0]['args']
                        })
                    elif hasattr(last_msg, 'content') and last_msg.content:
                        thought_steps.append({
                            "type": "response",
                            "content": last_msg.content
                        })
                final_result = state
        
        # Display thought process
        with thought_container:
            for step in thought_steps:
                if step["type"] == "tool_call":
                    st.markdown(f'<div class="tool-call">🔧 <b>Calling tool:</b> {step["tool"]}<br/>📝 <b>Input:</b> {step["input"]}</div>', unsafe_allow_html=True)
                elif step["type"] == "response":
                    content = step["content"][:300] + "..." if len(step["content"]) > 300 else step["content"]
                    st.markdown(f'<div class="thought-process">💭 {content}</div>', unsafe_allow_html=True)
        
        # Display final response
        if final_result:
            st.markdown("### 🤖 AgentD's Response")
            st.markdown(f'<div class="agent-response">{final_result["messages"][-1].content}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Chat interface for follow-up questions
    st.markdown("### 💬 Follow-up Questions")
    st.markdown("Ask the agent more questions about the topic based on its research:")
    
    follow_up = st.text_input(
        "Your follow-up question:",
        placeholder="Ask a follow-up question...",
        key=f"follow_up_{st.session_state.messages_count}"
    )
    
    if follow_up:
        with st.spinner("⏳ Processing..."):
            inputs = {"messages": [HumanMessage(content=follow_up)]}
            
            final_result = None
            tool_used = False
            
            for event in st.session_state.agent.stream(inputs, config=st.session_state.config):
                for node, state in event.items():
                    final_result = state
                    if node == "action" and state.get("messages"):
                        last_msg = state["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            tool_used = True
            
            if tool_used:
                st.info("🔧 Used search tool to answer your question")
            
            if final_result:
                st.markdown(f'<div class="agent-response"><b>Agent:</b> {final_result["messages"][-1].content}</div>', unsafe_allow_html=True)
                st.session_state.messages_count += 1
    
    # Reset button
    st.divider()
    if st.button("🔄 Start New Research", use_container_width=True):
        st.session_state.conversation_started = False
        del st.session_state.initial_query
        st.session_state.messages_count = 0
        st.rerun()

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    🚀 Powered by LangChain, LangGraph & Gemini | 🔍 Search powered by DuckDuckGo
    </div>
""", unsafe_allow_html=True)
