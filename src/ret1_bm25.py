# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: llm692_venv
#     language: python
#     name: python3
# ---

# %%
# %%
import json
import os
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from data_utils import load_casehold

class SimpleBM25Retriever:
    """
    A simple BM25 retriever implementation using rank_bm25 package.
    No Java dependencies required.
    """
    
    def __init__(self, index_name="legal_bm25_index"):
        self.index_name = index_name
        self.index_dir = os.path.join(os.getcwd(), index_name)
        os.makedirs(self.index_dir, exist_ok=True)
        self.bm25 = None
        self.tokenized_corpus = None
        self.documents = None
        self.doc_ids = None
    
    def tokenize(self, text):
        """Simple whitespace tokenization"""
        return text.lower().split()
    
    def index_corpus(self, documents, doc_ids):
        """Build BM25 index from documents"""
        print(f"Building BM25 index with {len(documents)} documents...")
        
        self.documents = documents
        self.doc_ids = doc_ids
        
        # Tokenize corpus
        print("Tokenizing documents...")
        self.tokenized_corpus = [self.tokenize(doc) for doc in tqdm(documents, total=len(documents))]
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Save the index
        self.save_index()
        
        print("BM25 index built successfully")
        return self
    
    def save_index(self):
        """Save the index to disk"""

        # bm25.pkl: The BM25 scoring object with term frequencies and IDF values
        with open(os.path.join(self.index_dir, "bm25.pkl"), 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # tokenized_corpus.pkl: The tokenized versions of all documents
        with open(os.path.join(self.index_dir, "tokenized_corpus.pkl"), 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        
        # documents.json: The original document texts
        with open(os.path.join(self.index_dir, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
        
        # doc_ids.json: The document IDs
        with open(os.path.join(self.index_dir, "doc_ids.json"), 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)
    
    def load_index(self):
        """Load pre-built BM25 index"""
        index_path = os.path.join(self.index_dir, "bm25.pkl")
        if not os.path.exists(index_path):
            raise ValueError(f"Index not found at {index_path}. Build index first with index_corpus()")
        
        with open(index_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        with open(os.path.join(self.index_dir, "tokenized_corpus.pkl"), 'rb') as f:
            self.tokenized_corpus = pickle.load(f)
        
        with open(os.path.join(self.index_dir, "documents.json"), 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(os.path.join(self.index_dir, "doc_ids.json"), 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
        
        print(f"Loaded BM25 index with {len(self.documents)} documents")
        return self
    
    def retrieve(self, query, k=100):
        """Retrieve top-k documents for a query"""
        if self.bm25 is None:
            self.load_index()
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k document indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        results = []
        for i in top_indices:
            if scores[i] > 0:  # Only include documents with non-zero scores
                results.append({
                    "id": self.doc_ids[i],
                    "score": float(scores[i]),
                    "text": self.documents[i]
                })
        
        return results
    
    def batch_retrieve(self, queries, k=100):
        """Retrieve top-k documents for multiple queries"""
        if self.bm25 is None:
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
