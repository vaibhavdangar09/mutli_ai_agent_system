"""
NLP Utilities - Text embedding and NLP functions
"""

import os
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import re
import unicodedata

logger = logging.getLogger(__name__)

# Global model instance (lazy loading)
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model.
    
    Returns:
        SentenceTransformer instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded with dimension {_embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    return _embedding_model

def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Get embedding for a text string.
    
    Args:
        text: Input text
        
    Returns:
        Numpy array of embedding or None if failed
    """
    try:
        # Clean and normalize the text
        text = clean_text(text)
        
        # Get the model
        model = get_embedding_model()
        
        # Generate embedding
        embedding = model.encode(text, show_progress_bar=False)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        Numpy array of embeddings
    """
    # Clean and normalize the texts
    texts = [clean_text(text) for text in texts]
    
    # Get the model
    model = get_embedding_model()
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split a long text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk size in characters
        overlap: Number of characters to overlap
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk)
            
            # Create overlap by including the last part of the previous chunk
            if overlap > 0 and current_chunk:
                overlap_text = " ".join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity (0-1)
    """
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    return float(similarity)