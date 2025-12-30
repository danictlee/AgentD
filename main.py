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

# 1. Define the "State" (The Shared Notebook)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 2. Define tools as LangChain tools
search_tool = DuckDuckGoSearchRun()

@tool
def search(query: str) -> str:
    """Search the internet for information"""
    return search_tool.run(query)

tools = [search]

def run_research_agent(initial_query: str = None):
    # Setup the Brain
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # --- NODES (The Workers) ---
    
    # Node 1: The Agent calls the LLM
    def call_model(state):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Node 2: The Tool Executor
    tool_node = ToolNode(tools)

    # --- EDGES (The Logic) ---
    
    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        
        # If the LLM called a tool, continue to tool execution
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        
        # Otherwise, end the conversation
        return END

    # --- BUILD THE GRAPH ---
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional logic
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            END: END,
        }
    )

    # After tool execution, go back to agent
    workflow.add_edge("action", "agent")

    # Compile the graph with memory persistence
    app = workflow.compile(checkpointer=memory)

    # --- RUN IT ---
    config = {"configurable": {"thread_id": "1"}}
    
    # Initial query
    if initial_query:
        print(f"🕵️‍♂️ Researching: {initial_query}...")
        print("\n--- THOUGHT PROCESS ---\n")
        
        inputs = {"messages": [HumanMessage(content=initial_query)]}
        
        final_result = None
        for event in app.stream(inputs, config=config):
            for node, state in event.items():
                print(f"[{node}]")
                if state.get("messages"):
                    last_msg = state["messages"][-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"  🔧 Calling tool: {last_msg.tool_calls[0]['name']}")
                        print(f"     Input: {last_msg.tool_calls[0]['args']}")
                    elif hasattr(last_msg, 'content'):
                        content = last_msg.content[:200] + "..." if len(last_msg.content) > 200 else last_msg.content
                        print(f"  💭 {content}")
                print()
                final_result = state
        
        print("\n-----------------\nAGENT RESPONSE:\n" + final_result["messages"][-1].content)
    
    # Interactive chat loop for follow-up questions
    print("\n\n🤖 Agent Ready! Ask follow-up questions (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n--- PROCESSING ---\n")
        
        # Add the user's follow-up question to the conversation
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        final_result = None
        for event in app.stream(inputs, config=config):
            for node, state in event.items():
                if node == "action":
                    print(f"[{node}] Running tool...")
                # Don't print agent node for every step to reduce noise
                final_result = state
        
        # Print the final response
        print("\n🤖 Agent: " + final_result["messages"][-1].content + "\n")

if __name__ == "__main__":
    topic = "What is the meaning of being a Performative Male?"
    run_research_agent(initial_query=topic)