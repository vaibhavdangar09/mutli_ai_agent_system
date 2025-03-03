#!/usr/bin/env python3
"""
Enhanced Query Assistant - Shows classification details and confidence scores
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import uuid

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the components directly
# from src.core.query_router import QueryRouter
from src.core.query_router import StructuredQueryRouter
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()

class EnhancedQueryAssistant:
    """
    Assistant that directly processes queries and shows detailed classification information.
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
        self.query_router =  StructuredQueryRouter()
        self.logger.info("Query router initialized")
    
    async def process_query(self, query: str, user_id: str = "direct_user") -> Dict[str, Any]:
        """
        Process a query directly.
        
        Args:
            query: The query text
            user_id: User identifier
            
        Returns:
            Response dictionary
        """
        self.logger.info(f"Processing query from {user_id}: {query}")
        
        try:
            # First, get the classification directly to display it
            classification = self.query_router.classify_query(query)
            
            # Print classification details
            print("\n=== Query Classification ===")
            print(f"Type: {classification.query_type}")
            print(f"Confidence: {classification.confidence * 100:.2f}%")
            print(f"Reasoning: {classification.reasoning}")
            print("===========================\n")
            
            # Now route the query to the appropriate agent
            result = await self.query_router.route_query(query)
            
            self.logger.info(f"Query processed by {result['agent_type']} agent")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "error": True,
                "message": f"Error processing query: {str(e)}"
            }

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
                continue
            
            # Display the answer
            response = result.get("response", {})
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
    
    except KeyboardInterrupt:
        print("\nExiting...")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Query Assistant for Multi-Agent RAG System")
    parser.add_argument(
        "--query", 
        type=str,
        help="One-off query (if not provided, runs in interactive mode)"
    )
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize the assistant
        assistant = EnhancedQueryAssistant()
        
        if args.query:
            # One-off query mode
            print("Processing query...")
            result = await assistant.process_query(args.query)
            
            if "error" in result and result["error"]:
                print(f"Error: {result['message']}")
                return 1
            
            # Display the answer
            response = result.get("response", {})
            answer = response.get("answer", "No answer provided")
            
            print(answer)
            return 0
        else:
            # Interactive mode
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