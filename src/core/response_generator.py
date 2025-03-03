"""
Response Generator - Combines retrieved data and generated response
"""

import logging
from typing import Dict, List, Any
import os
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

def generate_response(
    query: str,
    documents: List[Dict[str, Any]],
    domain: str = "general",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response based on query and retrieved documents.
    
    Args:
        query: User's original query
        documents: Retrieved documents from the knowledge base
        domain: Domain of the query (billing, technical, order)
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens for the response
        
    Returns:
        Generated response text
    """
    logger.info(f"Generating response for {domain} query with {len(documents)} documents")
    
    # Initialize the LLM
    llm = ChatGroq(
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    
    # Format the context from documents
    context = ""
    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        source = doc.get("source", "Unknown")
        context += f"Document {i+1} from {source}:\n{content}\n\n"
    
    # Create domain-specific prompts
    domain_prompts = {
        "billing": """You are a billing support specialist. Use the provided information to answer the customer's question about billing, payments, invoices, refunds, or other financial matters. If the information doesn't contain the answer, acknowledge what you don't know and suggest what they might do next.""",
        
        "technical": """You are a technical support specialist. Use the provided information to solve the customer's technical issue or explain how a feature works. Be precise and clear in your explanations. If the information doesn't contain the answer, acknowledge what you don't know and suggest what they might do next.""",
        
        "order": """You are an order management specialist. Use the provided information to answer the customer's question about their order status, shipping, delivery, returns, or product availability. If the information doesn't contain the answer, acknowledge what you don't know and suggest what they might do next.""",
        
        "general": """You are a helpful customer support agent. Use the provided information to answer the customer's question as accurately as possible. If the information doesn't contain the answer, acknowledge what you don't know and suggest what they might do next."""
    }
    
    system_prompt = domain_prompts.get(domain, domain_prompts["general"])
    
    # Build the full prompt
    prompt = f"""
    {system_prompt}
    
    CONTEXT INFORMATION:
    {context if context else "No specific information available for this query."}
    
    USER QUESTION:
    {query}
    
    YOUR RESPONSE:
    """
    
    # Generate the response
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again later or contact our support team for assistance."

def get_confidence_score(query: str, response: str, documents: List[Dict[str, Any]]) -> float:
    """
    Calculate a confidence score for the generated response.
    
    Args:
        query: Original user query
        response: Generated response
        documents: Retrieved documents
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Simple heuristic-based confidence scoring
    if not documents:
        return 0.2  # Low confidence if no documents were retrieved
    
    # Check if the response contains uncertainty markers
    uncertainty_phrases = ["I don't know", "I'm not sure", "I don't have enough information",
                          "I can't determine", "not specified", "unclear", "not mentioned"]
    
    for phrase in uncertainty_phrases:
        if phrase in response.lower():
            return 0.4  # Medium-low confidence for uncertain responses
    
    # Count how many documents seem relevant
    relevant_count = 0
    for doc in documents:
        content = doc.get("content", "").lower()
        # Simple relevance check - the document contains keywords from the query
        query_words = set(query.lower().split())
        if any(word in content for word in query_words):
            relevant_count += 1
    
    # Calculate relevance ratio
    relevance_ratio = relevant_count / len(documents) if documents else 0
    
    # Final confidence score (mix of document quality and relevance)
    base_confidence = 0.7  # Base confidence for responses with relevant documents
    confidence = base_confidence + (0.3 * relevance_ratio)  # Boost by relevance
    
    return min(confidence, 1.0)  # Cap at 1.0