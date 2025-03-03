"""
Order Agent - Handles order-related queries using a RAG approach
"""

import os
import logging
from typing import Dict, List, Any
import faiss
import numpy as np
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from src.database.faiss_db.faiss_utils import query_faiss_index
from src.utils.nlp_utils import get_embedding
from src.core.response_generator import generate_response, get_confidence_score

logger = logging.getLogger(__name__)

class OrderDocument(BaseModel):
    """Schema for an order document retrieved from the knowledge base"""
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the document")

class OrderAgent:
    """
    Agent responsible for handling order-related queries using RAG.
    """
    
    def __init__(self):
        """Initialize the order agent with its knowledge base."""
        # Load FAISS index
        self.index_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "database", 
            "faiss_db", 
            "order_index.faiss"
        )
        
        # Check if index exists
        if not os.path.exists(self.index_path):
            logger.warning(f"FAISS index not found at {self.index_path}")
            self.index = None
        else:
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = None
        
        # Load document lookup
        self.docs_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "database",
            "faiss_db",
            "order_documents.npy"
        )
        
        if os.path.exists(self.docs_path):
            self.documents = np.load(self.docs_path, allow_pickle=True).item()
            logger.info(f"Loaded {len(self.documents)} order documents")
        else:
            logger.warning(f"Document lookup not found at {self.docs_path}")
            self.documents = {}
        
        # Initialize LLM for response generation
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        
        logger.info("OrderAgent initialized")
    
    async def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for the query.
        
        Args:
            query: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        if self.index is None:
            logger.warning("FAISS index not available, returning empty results")
            return []
        
        # Get embedding for the query
        query_embedding = get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to get embedding for query")
            return []
        
        # Query the FAISS index
        doc_ids, distances = query_faiss_index(
            self.index,
            query_embedding,
            top_k=top_k
        )
        
        # Retrieve the actual documents
        results = []
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append({
                    "id": str(doc_id),
                    "content": doc["content"],
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", ""),
                    "metadata": {
                        "distance": float(distances[i]),
                        **doc.get("metadata", {})
                    }
                })
            else:
                logger.warning(f"Document with ID {doc_id} not found in document store")
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process an order-related query using RAG.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Dict containing the response and metadata
        """
        logger.info(f"Processing order query: {query}")
        
        # Retrieve relevant documents
        documents = await self.retrieve_relevant_documents(query)
        
        # Generate response
        response = generate_response(
            query=query,
            documents=documents,
            domain="order"
        )
        
        # Calculate confidence score
        confidence = get_confidence_score(query, response, documents)
        
        # Return the response with metadata
        return {
            "answer": response,
            "sources": [doc for doc in documents],
            "confidence": confidence
        }
    
    async def track_order(self, order_id: str) -> Dict[str, Any]:
        """
        Track a specific order by ID.
        
        Args:
            order_id: The order ID to track
            
        Returns:
            Dict containing order tracking information
        """
        logger.info(f"Tracking order: {order_id}")
        
        # Build a query based on the order ID
        query = f"What is the status of order {order_id}?"
        
        # Process the query to get tracking information
        result = await self.process_query(query)
        
        return {
            "order_id": order_id,
            "status": result.get("answer"),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.5)
        }