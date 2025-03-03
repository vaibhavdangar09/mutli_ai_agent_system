#!/usr/bin/env python3
"""
Enhanced Query Assistant - Uses langgraph for structured query processing.
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.graph.state import END

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import components
from src.core.query_router import StructuredQueryRouter
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()


class EnhancedQueryAssistant:
    """
    Assistant that processes queries using langgraph for structured execution.
    """
    
    def __init__(self):
        """Initialize the assistant."""
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Check for required API keys
        if not os.getenv("GROQ_API_KEY"):
            self.logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY is required")
        
        # Initialize the query router
        self.query_router = StructuredQueryRouter()
        self.logger.info("Query router initialized")

        # Create the graph
        self.graph = self.create_graph()

    def classify_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the query using the structured query router.
        """
        query = state["query"]
        classification = self.query_router.classify_query(query)

        self.logger.info(f"Classified query as {classification.query_type} with confidence {classification.confidence:.2f}")

        # Print classification details
        print("\n=== Query Classification ===")
        print(f"Type: {classification.query_type}")
        print(f"Confidence: {classification.confidence * 100:.2f}%")
        print(f"Reasoning: {classification.reasoning}")
        print("===========================\n")

        state["classification"] = classification
        return state

    async def route_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the query to the appropriate agent based on classification.
        """
        query = state["query"]
        result = await self.query_router.route_query(query)

        self.logger.info(f"Query processed by {result['agent_type']} agent")

        state["response"] = result
        return state

    def process_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and format the response before returning.
        """
        response = state["response"].get("response", {})
        answer = response.get("answer", "No answer provided")
        confidence = response.get("confidence", 0.0)

        print("\n=== Response ===")
        print(answer)
        print(f"\nResponse confidence: {confidence * 100:.2f}%")
        print("===============")

        # Ask if user wants to see sources
        show_sources = input("\nShow reference sources? (y/n): ").lower()
        
        if show_sources == "y":
            sources = response.get("sources", [])
            if not sources:
                print("No sources available")
            else:
                print("\n=== Reference Sources ===")
                for i, source in enumerate(sources, 1):
                    source_id = source.get("id", "N/A")
                    source_name = source.get("source", "Unknown")
                    distance = source.get("metadata", {}).get("distance", 0)
                    relevance = f"{(1 - distance) * 100:.1f}%" if isinstance(distance, (int, float)) else "N/A"
                    
                    print(f"Source {i}: {source_name} (ID: {source_id}, Relevance: {relevance})")
                    content = source.get("content", "")
                    print(f"Content Preview: {content[:100]}..." if len(content) > 100 else f"Content: {content}")
                    print()

        return state

    def create_graph(self):
        """Define the langgraph workflow."""
        # Create a new StateGraph with Dict[str, Any] as the state type
        workflow = StateGraph(Dict[str, Any])
        
        # Add nodes
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("process_response", self.process_response)

        # Define the edges
        workflow.add_edge("classify_query", "route_query")
        workflow.add_edge("route_query", "process_response")
        workflow.add_edge("process_response", END)
        
        # Set the entry point
        workflow.set_entry_point("classify_query")
        
        # Compile the graph
        return workflow.compile()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Run the langgraph workflow for the user's query."""
        return await self.graph.ainvoke({"query": query})

async def interactive_mode(assistant: EnhancedQueryAssistant):
    """
    Run the assistant in interactive mode.
    
    Args:
        assistant: EnhancedQueryAssistant instance
    """
    print("=== Enhanced Multi-Agent RAG System ===")
    print("Enter your queries below. Type 'exit', 'quit', or press Ctrl+C to exit.")
    print()
    
    try:
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() in ("exit", "quit"):
                break
            
            if not query.strip():
                continue
            
            print("\nProcessing query...")
            result = await assistant.process_query(query)
            
            if "error" in result and result["error"]:
                print(f"Error: {result['message']}")
    
    except KeyboardInterrupt:
        print("\nExiting...")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Query Assistant for Multi-Agent RAG System")
    parser.add_argument("--query", type=str, help="One-off query (if not provided, runs in interactive mode)")
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        assistant = EnhancedQueryAssistant()
        
        if args.query:
            print("Processing query...")
            result = await assistant.process_query(args.query)
            
            if "error" in result and result["error"]:
                print(f"Error: {result['message']}")
                return 1
            
            print(result.get("response", {}).get("answer", "No answer provided"))
            return 0
        else:
            await interactive_mode(assistant)
            return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

