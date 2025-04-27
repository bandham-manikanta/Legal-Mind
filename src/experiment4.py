# experiment4.py
import json
import os
import time
from typing import Any, Dict, List, Tuple


class DenseRerankerPipeline:
    """
    Pipeline for Dense retrieval with cross-encoder reranking (Experiment 4)
    """

    def __init__(self, dense_retriever=None, reranker=None, retriever_k=100, reranker_k=10):
        """
        Initialize the Dense Retriever + Reranker pipeline.

        Args:
            dense_retriever: DenseRetriever instance
            reranker: LegalCrossEncoder instance
            retriever_k: Number of documents to retrieve with dense retriever
            reranker_k: Final number of documents after reranking
        """
        # Initialize Dense retriever if not provided
        if dense_retriever is None:
            raise ValueError("Dense retriever must be provided")
        self.retriever = dense_retriever

        # Initialize reranker if not provided
        if reranker is None:
            raise ValueError("Reranker must be provided")
        self.reranker = reranker

        self.retriever_k = retriever_k
        self.reranker_k = reranker_k

    def search(self, query: str) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Perform the complete retrieval and reranking process.

        Args:
            query: User query string

        Returns:
            Tuple of (reranked_results, timing_info)
        """
        timing = {}

        # Step 1: Dense Retrieval
        retrieve_start = time.time()
        initial_results = self.retriever.retrieve(query, k=self.retriever_k)
        retrieve_end = time.time()
        timing['retrieve_time'] = retrieve_end - retrieve_start

        print(
            f"Retrieved {len(initial_results)} documents with dense retriever")

        # Step 2: Cross-Encoder Reranking
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(
            query=query,
            documents=initial_results,
            top_k=self.reranker_k
        )
        rerank_end = time.time()
        timing['rerank_time'] = rerank_end - rerank_start
        timing['total_time'] = timing['retrieve_time'] + timing['rerank_time']

        print(f"Reranked to {len(reranked_results)} documents")

        return reranked_results, timing

    def batch_search(self, queries: List[str]) -> Dict[str, Any]:
        """
        Process multiple queries and return results for each.

        Args:
            queries: List of query strings

        Returns:
            Dictionary mapping query indices to results and timing
        """
        all_results = {}

        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query}")
            results, timing = self.search(query)
            all_results[str(i)] = {
                "query": query,
                "results": results,
                "timing": timing
            }

        return all_results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save search results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved search results to {output_file}")
