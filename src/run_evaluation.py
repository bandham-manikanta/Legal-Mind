"""
Comprehensive evaluation script for legal document retrieval systems.

This script evaluates and compares:
1. BM25 retriever
2. Dense Retriever
3. BM25 + Cross-encoder
4. Dense Retriever + Cross-encoder

It measures:
- Precision@k (k=5, 10, 20)
- Recall@k (k=5, 10, 20)
- F1 Score
- MAP (Mean Average Precision)
- Retrieval Time
"""

import os
import json
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Import components
from ret1_bm25 import SimpleBM25Retriever
from ret2_dense import DenseRetriever
from reranker import LegalCrossEncoder
from exp3_bm25_rerank import BM25RerankerPipeline
from exp4_dense_rerank import DenseRerankerPipeline
from data_utils import load_casehold, find_project_root
from evaluation_script import RetrievalEvaluator, prepare_evaluation_data
from visualization_script import main as visualize_results

def ensure_dirs(dirs: List[str]):
    """Ensure directories exist"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def initialize_retrieval_systems(args):
    """Initialize all retrieval systems"""
    print("\n1. Initializing retrieval systems...\n" + "="*50)
    
    # Initialize BM25 retriever
    print("\nInitializing BM25 retriever...")
    bm25 = SimpleBM25Retriever(index_name=args.bm25_index_dir)
    
    # Check if BM25 index exists
    if not os.path.exists(os.path.join(args.bm25_index_dir, "bm25.pkl")):
        print("Building BM25 index...")
        docs, doc_ids = load_casehold(args.data_path)
        bm25.index_corpus(docs, doc_ids)
    else:
        print("Loading existing BM25 index...")
        bm25.load_index()
    
    # Initialize Dense retriever
    print("\nInitializing Dense retriever...")
    dense = DenseRetriever(model_name=args.dense_model, index_name=args.dense_index_dir)
    
    # Check if Dense index exists
    if not os.path.exists(os.path.join(args.dense_index_dir, "faiss.index")):
        print("Building Dense index...")
        docs, doc_ids = load_casehold(args.data_path)
        dense.index_corpus(docs, doc_ids, batch_size=args.batch_size)
    else:
        print("Loading existing Dense index...")
        dense.load_index()
    
    # Initialize reranker
    print("\nInitializing Cross-encoder reranker...")
    reranker = LegalCrossEncoder(
        model_name=args.reranker_model,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # Initialize combined pipelines
    print("\nInitializing combined pipelines...")
    bm25_rerank = BM25RerankerPipeline(
        bm25_retriever=bm25,
        reranker=reranker,
        retriever_k=args.retriever_k,
        reranker_k=args.reranker_k
    )
    
    dense_rerank = DenseRerankerPipeline(
        dense_retriever=dense,
        reranker=reranker,
        retriever_k=args.retriever_k,
        reranker_k=args.reranker_k
    )
    
    # Return all systems
    return {
        "bm25": bm25,
        "dense": dense,
        "reranker": reranker,
        "bm25_rerank": bm25_rerank,
        "dense_rerank": dense_rerank
    }

# In run_evaluation.py

import time # Make sure time is imported at the top of run_evaluation.py

def create_retrieval_functions(systems):
    """Create standardized retrieval functions for evaluation"""
    retrieval_funcs = {}
    eval_k_for_retrieval = 20 # Or get from args.reranker_k or max(args.evaluation_k_values)

    # BM25 retrieval
    def bm25_retrieve_eval_wrapper(query_text: str):
        start_time = time.time()
        # Assuming systems["bm25"] is an instance of SimpleBM25Retriever
        # and its retrieve method returns a list of doc dicts
        results = systems["bm25"].retrieve(query_text, k=eval_k_for_retrieval) 
        end_time = time.time()
        timing_info = {'retrieve_time': end_time - start_time, 'total_time': end_time - start_time}
        return results, timing_info
    retrieval_funcs["BM25"] = bm25_retrieve_eval_wrapper # Use the new wrapper name

    # Dense retrieval
    def dense_retrieve_eval_wrapper(query_text: str):
        start_time = time.time()
        # Assuming systems["dense"] is an instance of DenseRetriever
        results = systems["dense"].retrieve(query_text, k=eval_k_for_retrieval)
        end_time = time.time()
        timing_info = {'retrieve_time': end_time - start_time, 'total_time': end_time - start_time}
        return results, timing_info
    retrieval_funcs["Dense Retriever"] = dense_retrieve_eval_wrapper # Use the new wrapper name

    # BM25 + Reranker
    def bm25_rerank_eval_wrapper(query_text: str):
        # Assuming systems["bm25_rerank"].search returns (results, timing_dict)
        # No change needed here if the .search() method already complies
        results, timing_info = systems["bm25_rerank"].search(query_text)
        return results, timing_info 
    retrieval_funcs["BM25 + Reranker"] = bm25_rerank_eval_wrapper # Use the new wrapper name

    # Dense + Reranker
    def dense_rerank_eval_wrapper(query_text: str):
        # Assuming systems["dense_rerank"].search returns (results, timing_dict)
        # No change needed here if the .search() method already complies
        results, timing_info = systems["dense_rerank"].search(query_text)
        return results, timing_info
    retrieval_funcs["Dense + Reranker"] = dense_rerank_eval_wrapper # Use the new wrapper name
    
    return retrieval_funcs

def run_evaluation(args):
    """Run the evaluation process"""
    # Create necessary directories
    ensure_dirs([args.bm25_index_dir, args.dense_index_dir, args.results_dir])
    
    # Initialize retrieval systems
    systems = initialize_retrieval_systems(args)
    
    print("\n2. Preparing evaluation data...\n" + "="*50)
    # Prepare evaluation data
    test_queries, relevance_judgments = prepare_evaluation_data(
        args.data_path, 
        num_queries=args.num_queries
    )
    
    # Create retrieval functions
    retrieval_funcs = create_retrieval_functions(systems)
    
    print("\n3. Running evaluation...\n" + "="*50)
    # Initialize evaluator
    evaluator = RetrievalEvaluator(k_values=args.evaluation_k_values)
    
    # Evaluate each system
    for system_name, retrieval_func in retrieval_funcs.items():
        evaluator.evaluate_system(system_name, retrieval_func, test_queries, relevance_judgments)
    
    # Compare systems
    # evaluator.compare_systems()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.results_dir, f"retrieval_eval_results_{timestamp}.json")
    evaluator.save_results(output_file)
    
    print("\n4. Generating visualizations...\n" + "="*50)
    # Generate visualizations
    visualize_results(output_file)
    
    return evaluator.get_all_aggregated_metrics()