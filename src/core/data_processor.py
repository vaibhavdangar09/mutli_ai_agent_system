"""
Data Processor - Manages data ingestion and vectorization for the knowledge base
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field

from src.utils.nlp_utils import get_embeddings, chunk_text
from src.database.faiss_db.faiss_utils import create_and_save_domain_index

logger = logging.getLogger(__name__)

class DataSource(BaseModel):
    """Schema for a data source to be processed"""
    name: str = Field(..., description="Source name")
    path: str = Field(..., description="Path to the source file")
    domain: str = Field(..., description="Domain (billing, technical, order)")
    format: str = Field("csv", description="File format (csv, json, txt)")
    text_column: str = Field(None, description="Column containing the text data")
    id_column: str = Field(None, description="Column containing unique identifiers")
    chunk_size: int = Field(512, description="Size of text chunks for long documents")
    chunk_overlap: int = Field(50, description="Overlap between chunks")

class DataProcessor:
    """
    Processes raw data into vectorized format for the FAISS index.
    """
    
    def __init__(self, base_data_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            base_data_dir: Base directory for data files
        """
        if base_data_dir is None:
            base_data_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data"
            )
        
        self.base_data_dir = base_data_dir
        self.raw_data_dir = os.path.join(base_data_dir, "raw_data")
        self.embeddings_dir = os.path.join(base_data_dir, "embeddings")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        logger.info(f"DataProcessor initialized with base dir: {base_data_dir}")
    
    def load_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """
        Load data from a source file.
        
        Args:
            source: DataSource configuration
            
        Returns:
            List of document dictionaries
        """
        # file_path = os.path.join(self.raw_data_dir, source.path)
        # Fix the path to avoid duplicate raw_data directory
        if source.path.startswith("raw_data/"):
            file_path = os.path.join(self.base_data_dir, source.path)
        else:
            file_path = os.path.join(self.raw_data_dir, source.path)
        logger.info(f"Loading data from {file_path} for domain {source.domain}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            if source.format == "csv":
                df = pd.read_csv(file_path)
            elif source.format == "json":
                df = pd.read_json(file_path)
            else:
                logger.error(f"Unsupported format: {source.format}")
                return []
            
            # Check if required columns exist
            if source.text_column and source.text_column not in df.columns:
                logger.error(f"Text column '{source.text_column}' not found in {file_path}")
                return []
            
            documents = []
            
            # Process each row
            for idx, row in df.iterrows():
                doc_id = row[source.id_column] if source.id_column and source.id_column in row else idx
                
                # Get text from the specified column
                if source.text_column:
                    text = str(row[source.text_column])
                else:
                    # If no specific column is specified, use the entire row
                    text = " ".join([f"{k}: {v}" for k, v in row.items()])
                
                # For long texts, create chunks
                if len(text) > source.chunk_size:
                    chunks = chunk_text(text, source.chunk_size, source.chunk_overlap)
                    
                    # Create a document for each chunk
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{doc_id}_{i}",
                            "content": chunk,
                            "source": source.name,
                            "domain": source.domain,
                            "metadata": {
                                "original_id": doc_id,
                                "chunk": i,
                                "total_chunks": len(chunks),
                                **{k: v for k, v in row.items() if k != source.text_column}
                            }
                        })
                else:
                    # Create a single document
                    documents.append({
                        "id": doc_id,
                        "content": text,
                        "source": source.name,
                        "domain": source.domain,
                        "metadata": {
                            **{k: v for k, v in row.items() if k != source.text_column}
                        }
                    })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def process_domain_data(self, domain: str, sources: List[DataSource]) -> bool:
        """
        Process all data sources for a domain and create a FAISS index.
        
        Args:
            domain: Domain name (billing, technical, order)
            sources: List of DataSource configurations
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing data for domain: {domain}")
        
        all_documents = []
        
        # Load all documents for the domain
        for source in sources:
            if source.domain != domain:
                continue
                
            documents = self.load_data(source)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.warning(f"No documents found for domain: {domain}")
            return False
        
        # Extract content for embedding
        texts = [doc["content"] for doc in all_documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        try:
            embeddings = get_embeddings(texts)
            
            # Save embeddings
            embeddings_path = os.path.join(self.embeddings_dir, f"{domain}_embeddings.npy")
            np.save(embeddings_path, embeddings)
            logger.info(f"Saved embeddings to {embeddings_path}")
            
            # Create and save FAISS index
            base_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "database",
                "faiss_db"
            )
            
            success, msg = create_and_save_domain_index(
                domain=domain,
                documents=all_documents,
                embeddings=embeddings,
                base_dir=base_dir
            )
            
            logger.info(msg)
            return success
            
        except Exception as e:
            logger.error(f"Error processing domain data: {e}")
            return False
    
    def process_all_domains(self, config_path: str) -> Dict[str, bool]:
        """
        Process data for all domains based on a configuration file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            Dictionary of domain names to success status
        """
        import yaml
        
        logger.info(f"Processing all domains with config: {config_path}")
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            domains = config.get("domains", {})
            results = {}
            
            for domain, domain_config in domains.items():
                sources = []
                
                for source_config in domain_config.get("sources", []):
                    sources.append(DataSource(**source_config))
                
                success = self.process_domain_data(domain, sources)
                results[domain] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing all domains: {e}")
            return {}