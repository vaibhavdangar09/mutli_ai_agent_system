"""
FAISS Configuration - Settings for FAISS indexes
"""

import os
import logging
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)

class FAISSConfig:
    """
    Configuration for FAISS indexes.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize FAISS configuration.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "config",
                "app_config.yaml"
            )
        
        self.config = self._load_config(config_path)
        self.faiss_config = self.config.get("faiss", {})
        logger.info("FAISS configuration loaded")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    @property
    def index_type(self) -> str:
        """Get the FAISS index type."""
        return self.faiss_config.get("index_type", "flat")
    
    @property
    def metric(self) -> str:
        """Get the distance metric."""
        return self.faiss_config.get("metric", "l2")
    
    @property
    def nprobe(self) -> int:
        """Get the nprobe parameter for IVF indices."""
        return self.faiss_config.get("nprobe", 10)
    
    @property
    def ef_search(self) -> int:
        """Get the ef_search parameter for HNSW indices."""
        return self.faiss_config.get("ef_search", 128)
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        data_config = self.config.get("data_processing", {})
        return data_config.get("embedding_dimension", 384)
    
    def get_index_params(self, domain: str = None) -> Dict[str, Any]:
        """
        Get index parameters for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with index parameters
        """
        # Base parameters
        params = {
            "index_type": self.index_type,
            "metric": self.metric,
            "dimension": self.dimension,
            "nprobe": self.nprobe,
            "ef_search": self.ef_search
        }
        
        # Domain-specific overrides
        if domain and "agents" in self.config:
            domain_config = self.config.get("agents", {}).get(domain, {})
            if "faiss" in domain_config:
                domain_faiss = domain_config["faiss"]
                params.update(domain_faiss)
        
        return params

def get_faiss_config() -> FAISSConfig:
    """
    Get the FAISS configuration singleton.
    
    Returns:
        FAISSConfig instance
    """
    if not hasattr(get_faiss_config, "_instance"):
        get_faiss_config._instance = FAISSConfig()
    return get_faiss_config._instance