# reranker.py
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class LegalCrossEncoder:
    """
    Cross-encoder reranker for legal documents that processes query-document pairs
    to provide more precise relevance scores.
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=None):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Pre-trained cross-encoder model to use
            max_length: Maximum sequence length for encoding
            device: Computing device (cpu or cuda)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing cross-encoder reranker with model {model_name} on {self.device}")
        
        # Initialize cross-encoder model
        self.model = CrossEncoder(
            model_name=model_name,
            max_length=max_length,
            device=self.device
        )
        
    def rerank(self, query: str, documents: List[Dict], top_k: int = None, batch_size: int = 32) -> List[Dict]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: User query string
            documents: List of documents from first-stage retriever
            top_k: Number of top results to return
            batch_size: Batch size for inference
            
        Returns:
            List of reranked documents with updated scores
        """
        if not documents:
            return []
        
        # Prepare document texts and create query-document pairs
        text_pairs = []
        for doc in documents:
            # Extract document text based on the existing document structure
            doc_text = doc['text'] if 'text' in doc else str(doc)
            text_pairs.append([query, doc_text])
            
        print(f"Reranking {len(text_pairs)} documents...")
        
        # Get scores from cross-encoder
        scores = self.model.predict(
            text_pairs, 
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # Create new document list with updated scores
        reranked_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Create a new document dict with the same fields as the original
            reranked_doc = doc.copy()
            reranked_doc['original_score'] = doc['score']  # Keep original score for reference
            reranked_doc['score'] = float(score)  # Update with cross-encoder score
            reranked_docs.append(reranked_doc)
            
        # Sort by the new score in descending order
        reranked_docs = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)
        
        # Return top_k results if specified
        if top_k:
            reranked_docs = reranked_docs[:top_k]
            
        return reranked_docs
