"""
FAISS Utilities - Functions for managing FAISS indices
"""

import os
import logging
import numpy as np
import faiss
from typing import Dict, List, Tuple, Any, Optional
import pickle

logger = logging.getLogger(__name__)

def create_faiss_index(
    embeddings: np.ndarray, 
    dimension: int = 384,
    index_type: str = "flat"
) -> faiss.Index:
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        dimension: Dimensionality of embeddings
        index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        
    Returns:
        FAISS index
    """
    logger.info(f"Creating {index_type} FAISS index with dimension {dimension}")
    
    if embeddings.shape[1] != dimension:
        raise ValueError(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {dimension}")
    
    if index_type == "flat":
        # Simple but exact search
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "ivf":
        # Inverted file index, faster but approximate
        nlist = min(int(len(embeddings) / 10), 100)  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        # Need to train with some data
        index.train(embeddings)
    elif index_type == "hnsw":
        # Hierarchical Navigable Small World, very fast and memory efficient
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add embeddings to the index
    index.add(embeddings)
    logger.info(f"Added {len(embeddings)} vectors to index")
    
    return index

def save_faiss_index(index: faiss.Index, path: str) -> bool:
    """
    Save a FAISS index to disk.
    
    Args:
        index: FAISS index to save
        path: Path to save the index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)
        logger.info(f"FAISS index saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return False

def load_faiss_index(path: str) -> Optional[faiss.Index]:
    """
    Load a FAISS index from disk.
    
    Args:
        path: Path to the FAISS index
        
    Returns:
        FAISS index or None if loading fails
    """
    try:
        index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path}")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None

def query_faiss_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    top_k: int = 5
) -> Tuple[List[int], List[float]]:
    """
    Query a FAISS index with a vector.
    
    Args:
        index: FAISS index to query
        query_vector: Vector to search for
        top_k: Number of results to return
        
    Returns:
        Tuple of (doc_ids, distances)
    """
    # Ensure query_vector is correctly shaped
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    
    # Perform the search
    distances, indices = index.search(query_vector, top_k)
    
    return indices[0].tolist(), distances[0].tolist()

def create_and_save_domain_index(
    domain: str,
    documents: List[Dict[str, Any]],
    embeddings: np.ndarray,
    base_dir: str = None
) -> Tuple[bool, str]:
    """
    Create and save a FAISS index and document lookup for a specific domain.
    
    Args:
        domain: Domain name (e.g., 'billing', 'technical', 'order')
        documents: List of document dictionaries
        embeddings: Numpy array of document embeddings
        base_dir: Base directory for saving files
        
    Returns:
        Tuple of (success, error_message)
    """
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "database",
            "faiss_db"
        )
    
    try:
        # Create document lookup
        doc_lookup = {i: doc for i, doc in enumerate(documents)}
        
        # Save document lookup
        docs_path = os.path.join(base_dir, f"{domain}_documents.npy")
        np.save(docs_path, doc_lookup)
        logger.info(f"Saved {len(documents)} {domain} documents to {docs_path}")
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = create_faiss_index(embeddings, dimension)
        
        index_path = os.path.join(base_dir, f"{domain}_index.faiss")
        success = save_faiss_index(index, index_path)
        
        if not success:
            return False, f"Failed to save {domain} FAISS index"
        
        return True, f"Successfully created and saved {domain} index and documents"
    
    except Exception as e:
        error_msg = f"Error creating {domain} index: {str(e)}"
        logger.error(error_msg)
        return False, error_msg