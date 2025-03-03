"""
Logging Configuration - Centralized logging setup
"""

import os
import logging
import logging.config
import yaml
from typing import Optional

def setup_logging(config_path: Optional[str] = None):
    """
    Configure logging based on a YAML configuration file.
    
    Args:
        config_path: Path to logging config YAML file. If None, uses default config.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "..", 
            "config", 
            "logging_config.yaml"
        )
    
    # Create logs directory if needed
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logs_dir, "multi_agent.log"))
            ]
        )
        logging.warning(f"Logging config not found at {config_path}, using basic configuration")
    
    # Register custom logging handlers if needed
    # ...

class QueryLogger:
    """
    Specialized logger for query tracking and monitoring.
    """
    
    def __init__(self, log_file: str = "queries.log"):
        """
        Initialize a query logger.
        
        Args:
            log_file: Name of the log file for query logging
        """
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        self.logger = logging.getLogger("query_logger")
        
        # Create a file handler for query logs
        handler = logging.FileHandler(os.path.join(logs_dir, log_file))
        handler.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.logger.addHandler(handler)
    
    def log_query(self, query_id: str, user_id: str, query: str, 
                  agent_type: str, response: str, processing_time: float):
        """
        Log a processed query.
        
        Args:
            query_id: Unique identifier for the query
            user_id: User identifier
            query: Original query text
            agent_type: Type of agent that processed the query
            response: Generated response
            processing_time: Time taken to process the query (in seconds)
        """
        self.logger.info(
            f"QUERY: id={query_id}, user={user_id}, agent={agent_type}, "
            f"time={processing_time:.2f}s, query='{query}', "
            f"response='{response[:100]}...'" if len(response) > 100 else f"response='{response}'"
        )
    
    def log_error(self, query_id: str, user_id: str, query: str, error: str):
        """
        Log a query processing error.
        
        Args:
            query_id: Unique identifier for the query
            user_id: User identifier
            query: Original query text
            error: Error message
        """
        self.logger.error(
            f"ERROR: id={query_id}, user={user_id}, "
            f"query='{query}', error='{error}'"
        )