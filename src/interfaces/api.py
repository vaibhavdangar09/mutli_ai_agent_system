"""
API Interface - FastAPI routes for the multi-agent system
"""

import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Body, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.query_router import StructuredQueryRouter

logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Model for a query request"""
    query: str = Field(..., description="The user's natural language query")
    user_id: str = Field(None, description="Optional user identifier for tracking")
    session_id: str = Field(None, description="Optional session identifier for tracking")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the query")

class QueryResponse(BaseModel):
    """Model for a query response"""
    query: str = Field(..., description="The original query")
    agent_type: str = Field(..., description="The agent that processed the query")
    response: Dict[str, Any] = Field(..., description="The agent's response")
    classification: Dict[str, Any] = Field(..., description="Classification metadata")
    processing_time: float = Field(..., description="Time taken to process the query (in seconds)")

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    agents: List[str] = Field(..., description="Available agents")

def setup_api_routes(app: FastAPI, query_router: StructuredQueryRouter):
    """
    Set up API routes for the FastAPI application.
    
    Args:
        app: FastAPI application
        query_router: QueryRouter instance
    """
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Set specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/api/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest = Body(...)):
        """
        Process a user query through the multi-agent system.
        """
        import time
        
        logger.info(f"Received query: {request.query}")
        start_time = time.time()
        
        try:
            # Route the query to the appropriate agent
            result = await query_router.route_query(request.query)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare the response
            response = QueryResponse(
                query=request.query,
                agent_type=result["agent_type"],
                response=result["response"],
                classification=result["classification"],
                processing_time=processing_time
            )
            
            logger.info(f"Query processed by {result['agent_type']} agent in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing query: {str(e)}"
            )
    
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.
        """
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            agents=list(query_router.agents.keys())
        )
    
    logger.info("API routes configured")