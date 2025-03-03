#!/usr/bin/env python3
"""
Data Ingestion Script - Processes raw data files and creates FAISS indices
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data_processor import DataProcessor, DataSource
from src.utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Ingestion for Multi-Agent RAG System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        choices=["billing", "technical", "order", "all"],
        default="all",
        help="Domain to process"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reprocessing of all data"
    )
    return parser.parse_args()

def load_data_config(config_path: str) -> Dict[str, Any]:
    """Load data configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}

def process_domain(processor: DataProcessor, domain: str, sources: List[DataSource], force: bool) -> bool:
    """Process data for a specific domain."""
    logging.info(f"Processing {domain} domain data")
    
    # Check if index already exists
    index_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "src",
        "database",
        "faiss_db", 
        f"{domain}_index.faiss"
    )
    
    if os.path.exists(index_path) and not force:
        logging.info(f"FAISS index for {domain} already exists. Use --force to reprocess.")
        return True
    
    result = processor.process_domain_data(domain, sources)
    if result:
        logging.info(f"Successfully processed {domain} domain data")
    else:
        logging.error(f"Failed to process {domain} domain data")
    
    return result

def main():
    """Main entry point for data ingestion."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_data_config(config_path)
    
    if not config:
        logging.error(f"Failed to load configuration from {config_path}")
        return 1
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process domains
    if args.domain == "all":
        domains = config.get("domains", {})
        results = {}
        
        for domain, domain_config in domains.items():
            sources = []
            for source_config in domain_config.get("sources", []):
                sources.append(DataSource(**source_config))
            
            result = process_domain(processor, domain, sources, args.force)
            results[domain] = result
        
        # Check if all domains were processed successfully
        if all(results.values()):
            logging.info("All domains processed successfully")
            return 0
        else:
            failed_domains = [domain for domain, success in results.items() if not success]
            logging.error(f"Failed to process domains: {', '.join(failed_domains)}")
            return 1
    else:
        # Process a specific domain
        domain = args.domain
        domain_config = config.get("domains", {}).get(domain, {})
        
        if not domain_config:
            logging.error(f"Domain {domain} not found in configuration")
            return 1
        
        sources = []
        for source_config in domain_config.get("sources", []):
            sources.append(DataSource(**source_config))
        
        result = process_domain(processor, domain, sources, args.force)
        return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())