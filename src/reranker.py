# reranker.py
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class LegalCrossEncoder:
    """
    Cross-encoder reranker for legal text.
    Uses a pretrained transformer model to rerank documents.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, 
                 device=None, batch_size=8):
        """
        Initialize the reranker with a pretrained model.
        
        Args:
            model_name: Name of the pretrained cross-encoder model
            max_length: Maximum input length for the model
            device: Device to run the model on (None for auto-detection)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading reranker model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
    def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """
        Score query-passage pairs using the cross-encoder model.
        
        Args:
            query: Query string
            passages: List of passage strings to score
            
        Returns:
            List of relevance scores for each query-passage pair
        """
        # Prepare input pairs
        features = []
        for passage in passages:
            features.append((query, passage))
        
        # Process in batches
        all_scores = []
        for i in range(0, len(features), self.batch_size):
            batch_features = features[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_features,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            ).to(self.device)
            
            # Score
            with torch.no_grad():
                scores = self.model(**inputs).logits
                
                # For models with multiple outputs, take the positive class score
                if scores.shape[1] > 1:
                    scores = scores[:, 1]  # Take positive class score
                else:
                    scores = scores.squeeze(-1)  # For single score output models
                
            all_scores.extend(scores.cpu().tolist())
            
        return all_scores
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents using the cross-encoder.
        
        Args:
            query: Query string
            documents: List of document dictionaries (must contain 'id', 'text', and 'score' fields)
            top_k: Number of documents to return after reranking (None for all)
            
        Returns:
            List of reranked document dictionaries with updated scores
        """
        if not documents:
            return []
        
        # Extract texts for scoring
        texts = [doc["text"] for doc in documents]
        
        # Score documents
        scores = self.score_pairs(query, texts)
        
        # Create reranked documents with new scores
        reranked_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            reranked_doc = doc.copy()
            reranked_doc["original_score"] = doc["score"]  # Keep original score
            reranked_doc["score"] = float(score)  # Update with reranker score
            reranked_docs.append(reranked_doc)
            
        # Sort by score in descending order
        reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
        
        # Take top-k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
            
        return reranked_docs