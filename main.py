#!/usr/bin/env python3
"""
Multi-Agent System with RAG-based Knowledge Base Agents
Main entry point for the application.
"""

import os
import logging
import yaml
from fastapi import FastAPI
from dotenv import load_dotenv

from src.core.query_router import StructuredQueryRouter
from src.interfaces.api import setup_api_routes
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config", "app_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_app():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Multi-Agent System")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not found in environment variables")
        raise ValueError("GROQ_API_KEY is required")
    
    # Initialize the query router
    query_router = StructuredQueryRouter()
    logger.info("Query router initialized")
    
    # Create FastAPI app
    app = FastAPI(
        title="Multi-Agent RAG System",
        description="A system of specialized RAG agents for handling various query types",
        version="1.0.0"
    )
    
    # Setup API routes
    setup_api_routes(app, query_router)
    logger.info("API routes configured")
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)