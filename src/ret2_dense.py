# modified_dense_retriever.py
import json
import os

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Copy from dense_retriever.py, but omit the code that runs at import time
class DenseRetriever:
    """Dense retriever class"""
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", index_name="legal_dense_index"):
        self.model_name = model_name
        self.index_name = index_name
        self.index_dir = os.path.join(os.getcwd(), index_name)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.index = None
        self.documents = None
        self.doc_ids = None
        self.embedding_dim = None
        
    def _initialize_model(self):
        """Load the model and tokenizer if not already loaded"""
        if self.tokenizer is None:
            print(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            
    def get_embedding(self, text, max_length=512):
        """Generate embedding for a single text"""
        # Ensure model and tokenizer are loaded
        self._initialize_model()
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length,
                              padding="max_length", truncation=True)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as text representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]  # Return as 1D array
        
    def index_corpus(self, documents, doc_ids, batch_size=8):
        """Generate embeddings for all documents and build a search index."""
        print(f"Building dense index with {len(documents)} documents...")
        
        # Store document texts and IDs
        self.documents = documents
        self.doc_ids = doc_ids
        
        # Ensure model and tokenizer are loaded
        self._initialize_model()
        
        # Generate embeddings for all documents
        print("Generating document embeddings...")
        all_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
            batch_docs = documents[i:i+batch_size]
            batch_inputs = self.tokenizer(batch_docs, padding=True, truncation=True,
                                       return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                
            # Use CLS token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        # Concatenate all batch embeddings
        document_embeddings = np.vstack(all_embeddings)
        self.embedding_dim = document_embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(document_embeddings)
        
        # Build FAISS index for fast similarity search
        print(f"Building FAISS index with {document_embeddings.shape[1]} dimensions...")
        self.index = faiss.IndexFlatIP(document_embeddings.shape[1])  # Inner product for cosine similarity
        self.index.add(document_embeddings)
        
        # Save the index and metadata
        self.save_index()
        print("Dense index built successfully")
        return self
        
    def save_index(self):
        """Save the index and associated data to disk"""
        print(f"Saving index to {self.index_dir}...")
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
            
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents) if self.documents else 0
        }
        
        with open(os.path.join(self.index_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        # Save documents and doc_ids
        with open(os.path.join(self.index_dir, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
        with open(os.path.join(self.index_dir, "doc_ids.json"), 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)
            
    def load_index(self):
        """Load pre-built index and associated data"""
        index_path = os.path.join(self.index_dir, "faiss.index")
        if not os.path.exists(index_path):
            raise ValueError(f"Index not found at {index_path}. Build index first with index_corpus()")
            
        print(f"Loading index from {self.index_dir}...")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(os.path.join(self.index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        self.model_name = metadata["model_name"]
        self.embedding_dim = metadata["embedding_dim"]
        
        # Load documents and doc_ids
        with open(os.path.join(self.index_dir, "documents.json"), 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        with open(os.path.join(self.index_dir, "doc_ids.json"), 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
            
        print(f"Loaded dense index with {len(self.documents)} documents")
        return self
        
    def retrieve(self, query, k=100):
        """Retrieve top-k documents for a query."""
        if self.index is None:
            self.load_index()
            
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Safety check
                results.append({
                    "id": self.doc_ids[idx],
                    "score": float(scores[0][i]),
                    "text": self.documents[idx]
                })
                
        return results
        
    def batch_retrieve(self, queries, k=100):
        """Retrieve top-k documents for multiple queries."""
        if self.index is None:
            self.load_index()
            
        all_results = {}
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            all_results[str(i)] = self.retrieve(query, k=k)
            
        return all_results
        
    def save_results(self, results, output_file):
        """Save retrieval results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved retrieval results to {output_file}")
