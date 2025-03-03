# """
# Query Router - Routes incoming queries to the appropriate agent.
# """

# import logging
# import os
# from typing import Dict, List, Tuple, Any
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq

# from src.agents.billing_agent import BillingAgent
# from src.agents.technical_agent import TechnicalAgent
# from src.agents.order_agent import OrderAgent
# from src.utils.nlp_utils import get_embedding

# logger = logging.getLogger(__name__)

# class QueryClassification(BaseModel):
#     """Classification result for a user query"""
#     query_type: str = Field(..., description="The type of query: 'billing', 'technical', or 'order'")
#     confidence: float = Field(..., description="Confidence score for the classification")
#     reasoning: str = Field(..., description="Reasoning behind the classification")

# class QueryRouter:
#     """
#     Routes user queries to the appropriate agent based on query classification.
#     """
    
#     def __init__(self):
#         """Initialize the router with agent instances and classifier model."""
        
#         # Initialize the agents
#         self.agents = {
#             "billing": BillingAgent(),
#             "technical": TechnicalAgent(),
#             "order": OrderAgent()
#         }
        
#         # Initialize the LLM for query classification
#         self.classifier = ChatGroq(
#             temperature=0,
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             model_name="llama-3.3-70b-versatile"
#         )
        
#         logger.info("QueryRouter initialized with all agents")
    
#     def classify_query(self, query: str) -> QueryClassification:
#         """
#         Classify the query to determine which agent should handle it.
        
#         Args:
#             query: The user's natural language query
            
#         Returns:
#             QueryClassification object with query_type, confidence, and reasoning
#         """
#         logger.info(f"Classifying query: {query}")
        
#         # Prompt for the LLM to classify the query
#         prompt = f"""
#         You are a query classifier for a customer support system. Your job is to determine whether 
#         the following query is related to billing issues, technical issues, or order-related issues.
        
#         Query: {query}
        
#         Analyze the query carefully and classify it as one of:
#         - billing: Questions about invoices, payments, refunds, pricing, subscription plans, costs, fees,
#           or anything related to money, charges, financial matters, or what's included in a plan
#         - technical: Questions about product functionality, errors, bugs, usage instructions, features,
#           technical specifications, compatibility, or troubleshooting
#         - order: Questions about order status, shipping, delivery, returns, product availability,
#           tracking packages, or order modifications
        
#         Examples of billing queries:
#         - "How much does the premium plan cost?"
#         - "What's included in my plan?"
#         - "When will I be charged?"
#         - "How do I update my payment method?"
#         - "What's included in the $99 plan?"
        

#         Respond with ONLY a JSON object (no markdown code blocks, no backticks) containing:
#         - query_type: The type of query (billing, technical, or order)
#         - confidence: A confidence score between 0 and 1
#         - reasoning: Your reasoning behind this classification
        
#         IMPORTANT: Return only the raw JSON with no markdown formatting or code blocks.
#         """
        
#         # Call the LLM to get classification
#         response = self.classifier.invoke(prompt)
        
#         try:
#             # Extract JSON from the response
#             json_str = response.content
            
#             # Check if the response is wrapped in markdown code blocks and extract it
#             if "```json" in json_str:
#                 # Extract the JSON from between the code blocks
#                 import re
#                 json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
#                 if json_match:
#                     json_str = json_match.group(1)
            
#             # Parse the result into the Pydantic model
#             result = QueryClassification.model_validate_json(json_str)
#             logger.info(f"Query classified as {result.query_type} with confidence {result.confidence}")
#             return result
#         except Exception as e:
#             logger.error(f"Error parsing classification result: {e}")
            
#             # Try to extract the classification directly if JSON parsing failed
#             lower_content = response.content.lower()
#             if "billing" in lower_content and ("pricing" in lower_content or "plan" in lower_content):
#                 query_type = "billing"
#                 confidence = 0.8
#                 reasoning = "Classification based on keyword matching after JSON parsing failed"
#             elif "technical" in lower_content:
#                 query_type = "technical"
#                 confidence = 0.7
#                 reasoning = "Classification based on keyword matching after JSON parsing failed"
#             elif "order" in lower_content or "shipping" in lower_content:
#                 query_type = "order" 
#                 confidence = 0.7
#                 reasoning = "Classification based on keyword matching after JSON parsing failed"
#             else:
#                 # Default to billing for any plan-related questions
#                 if "$" in query or "plan" in query.lower() or "cost" in query.lower() or "price" in query.lower():
#                     query_type = "billing"
#                     confidence = 0.8
#                     reasoning = "Classification based on query keywords (plan/price related)"
#                 else:
#                     # Default to technical if all else fails
#                     query_type = "technical"
#                     confidence = 0.5
#                     reasoning = "Default classification due to parsing error"
            
#             return QueryClassification(
#                 query_type=query_type,
#                 confidence=confidence,
#                 reasoning=reasoning
#             )
    
#     async def route_query(self, query: str) -> Dict[str, Any]:
#         """
#         Route the query to the appropriate agent and return its response.
        
#         Args:
#             query: The user's natural language query
            
#         Returns:
#             Dict containing the agent's response and metadata
#         """
#         # Classify the query
#         classification = self.classify_query(query)
        
#         # Get the appropriate agent
#         agent_type = classification.query_type
#         if agent_type not in self.agents:
#             logger.warning(f"Unknown agent type: {agent_type}, defaulting to technical")
#             agent_type = "technical"
        
#         agent = self.agents[agent_type]
        
#         # Let the agent process the query
#         response = await agent.process_query(query)
        
#         # Return the result with metadata
#         return {
#             "query": query,
#             "classification": classification.model_dump(),
#             "agent_type": agent_type,
#             "response": response
#         }

"""
Structured Query Router - Routes incoming queries to the appropriate agent
using a more robust structured classification approach.
"""

import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from src.agents.billing_agent import BillingAgent
from src.agents.technical_agent import TechnicalAgent
from src.agents.order_agent import OrderAgent
from src.utils.nlp_utils import get_embedding

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    """Enumeration of possible query types"""
    BILLING = "billing"
    TECHNICAL = "technical"
    ORDER = "order"

class QueryClassification(BaseModel):
    """
    Enhanced structured classification result for a user query
    using typed fields and proper validation
    """
    query_type: QueryType = Field(..., description="The type of query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the classification")
    reasoning: str = Field(..., description="Reasoning behind the classification")
    key_terms: List[str] = Field(default_factory=list, description="Key terms identified in the query")
    suggested_agent: Optional[str] = Field(None, description="Suggested agent for handling the query")

class StructuredQueryRouter:
    """
    Routes user queries to the appropriate agent based on structured query classification.
    """
    
    def __init__(self):
        """Initialize the router with agent instances and classifier model."""
        
        # Initialize the agents
        self.agents = {
            "billing": BillingAgent(),
            "technical": TechnicalAgent(),
            "order": OrderAgent()
        }
        
        # Initialize the LLM for query classification
        self.classifier = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
        
        logger.info("StructuredQueryRouter initialized with all agents")
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify the query to determine which agent should handle it.
        
        Args:
            query: The user's natural language query
            
        Returns:
            QueryClassification object with structured fields
        """
        logger.info(f"Classifying query: {query}")
        
        # Enhanced prompt for the LLM to classify the query with structured output
        prompt = f"""
        You are a query classifier for a customer support system. Your job is to determine 
        the most appropriate category for the following query.
        
        Query: {query}
        
        Analyze the query carefully and categorize it as one of the following types:
        
        1. BILLING: Questions about invoices, payments, refunds, pricing, subscription plans, 
           costs, fees, or anything related to money, charges, financial matters, 
           or what's included in specific plans.
           
        2. TECHNICAL: Questions about product functionality, errors, bugs, usage instructions, 
           features, technical specifications, compatibility, or troubleshooting.
           
        3. ORDER: Questions about order status, shipping, delivery, returns, product availability,
           tracking packages, or order modifications.
        
        Examples of each category:
        
        BILLING examples:
        - "How much does the premium plan cost?"
        - "What's included in my plan?"
        - "When will I be charged?"
        - "How do I update my payment method?"
        - "What's included in the $99 plan?"
        
        TECHNICAL examples:
        - "How do I reset my password?"
        - "Why is the app crashing?"
        - "How do I enable dark mode?"
        - "Is the system compatible with iOS?"
        - "How do I backup my data?"
        
        ORDER examples:
        - "Where is my order #12345?"
        - "How long does shipping take?"
        - "Can I change my shipping address?"
        - "What's your return policy?"
        - "How do I track my package?"
        
        Respond with JSON that conforms to this Pydantic model:
        
        ```python
        class QueryClassification(BaseModel):
            query_type: str  # Must be one of: "billing", "technical", "order"
            confidence: float  # A value between 0.0 and 1.0
            reasoning: str  # Your reasoning for this classification
            key_terms: List[str]  # Key terms that influenced your decision
            suggested_agent: Optional[str]  # Suggested agent (same as query_type)
        ```
        
        IMPORTANT:
        1. Return only the raw JSON with no markdown formatting, no code blocks, no backticks.
        2. Ensure the query_type field is exactly one of: "billing", "technical", "order"
        3. The confidence value must be between 0.0 and 1.0
        4. Be thorough in your reasoning and explain your classification
        5. For key_terms, extract specific words or phrases that indicate the query type
        """
        
        # Call the LLM to get classification
        try:
            response = self.classifier.invoke(prompt)
            
            # Extract and clean the JSON from the response
            json_str = response.content
            
            # Remove any potential markdown formatting
            if "```" in json_str:
                import re
                json_match = re.search(r'```(?:json|python)?\s*(.*?)\s*```', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
            
            # Parse the result into the Pydantic model
            result = QueryClassification.model_validate_json(json_str)
            logger.info(f"Query classified as {result.query_type} with confidence {result.confidence}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            
            # Implement fallback classification
            fallback_type = self._fallback_classification(query)
            logger.warning(f"Using fallback classification: {fallback_type}")
            
            return QueryClassification(
                query_type=fallback_type,
                confidence=0.6,
                reasoning="Fallback classification based on keyword matching due to parsing error",
                key_terms=self._extract_key_terms(query),
                suggested_agent=fallback_type
            )
    
    def _fallback_classification(self, query: str) -> str:
        """
        Provide a fallback classification based on simple keyword matching.
        
        Args:
            query: The user's query
            
        Returns:
            Classification string
        """
        query_lower = query.lower()
        
        # Billing keywords
        billing_keywords = ["bill", "payment", "refund", "charge", "cost", "price", "plan", 
                           "$", "dollar", "subscription", "pay", "pricing", "fee", "discount"]
        
        # Technical keywords
        technical_keywords = ["how do i", "not working", "error", "bug", "feature", "help", 
                             "password", "login", "account", "reset", "update", "install", 
                             "connect", "setup", "configure"]
        
        # Order keywords
        order_keywords = ["order", "ship", "delivery", "track", "package", "return", 
                         "cancel", "status", "arrive", "shipping", "delivered", "purchase"]
        
        # Count matches for each category
        billing_count = sum(1 for kw in billing_keywords if kw in query_lower)
        technical_count = sum(1 for kw in technical_keywords if kw in query_lower)
        order_count = sum(1 for kw in order_keywords if kw in query_lower)
        
        # Determine the classification based on the highest count
        if billing_count > technical_count and billing_count > order_count:
            return "billing"
        elif technical_count > billing_count and technical_count > order_count:
            return "technical"
        elif order_count > billing_count and order_count > technical_count:
            return "order"
        
        # Default to technical if no clear winner
        return "technical"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract potential key terms from a query for fallback classification.
        
        Args:
            query: The user's query
            
        Returns:
            List of key terms
        """
        # Simple extraction of potential key terms
        query_lower = query.lower()
        terms = []
        
        # Check for price mentions
        import re
        price_match = re.search(r'\$\d+', query)
        if price_match:
            terms.append(price_match.group(0))
        
        # Check for specific keywords
        keywords = ["plan", "order", "account", "password", "shipping", 
                   "delivery", "payment", "refund", "charge", "technical"]
        
        for kw in keywords:
            if kw in query_lower:
                terms.append(kw)
        
        return terms
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route the query to the appropriate agent and return its response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Dict containing the agent's response and metadata
        """
        # Classify the query
        classification = self.classify_query(query)
        
        # Get the appropriate agent
        agent_type = classification.query_type
        if agent_type not in self.agents:
            logger.warning(f"Unknown agent type: {agent_type}, defaulting to technical")
            agent_type = "technical"
        
        agent = self.agents[agent_type]
        
        # Let the agent process the query
        response = await agent.process_query(query)
        
        # Return the result with metadata
        return {
            "query": query,
            "classification": classification.model_dump(),
            "agent_type": agent_type,
            "response": response
        }