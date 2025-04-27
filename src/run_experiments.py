# run_experiments.py
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from evaluator import RetrievalEvaluator
from experiment3 import BM25RerankerPipeline
from generate_dummy_data import generate_legal_corpus
from reranker import LegalCrossEncoder
from retriever_BM25 import SimpleBM25Retriever

# Import the modified DenseRetriever class instead of the original
if "--exp4" in sys.argv:
    try:
        # First try to import the modified version
        try:
            from modified_dense_retriever import DenseRetriever
            print("Using modified_dense_retriever module")
            dense_available = True
        except ImportError:
            print("Warning: modified_dense_retriever.py not found. Creating it on-the-fly.")
            
            # If it doesn't exist, create it by copying and modifying the original
            import importlib.util
            import re

            # Read the original file
            with open('dense_retriever.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove any code that runs at module level (after the class definition)
            content = re.sub(r'if\s+__name__\s*==\s*"__main__".*', '', content, flags=re.DOTALL)
            
            # Write to a temporary file
            with open('_temp_dense_retriever.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Import from the temporary file
            spec = importlib.util.spec_from_file_location("_temp_dense_retriever", "_temp_dense_retriever.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DenseRetriever = module.DenseRetriever
            dense_available = True
            
        from experiment4 import DenseRerankerPipeline
    except Exception as e:
        print(f"Warning: Could not import Dense retriever modules: {e}")
        print("Experiment 4 will not be available.")
        dense_available = False
else:
    dense_available = False

def load_queries(queries_file: str) -> List[str]:
    """Load queries from file"""
    with open(queries_file, 'r') as f:
        return json.load(f)

def ensure_corpus_exists(corpus_file: str = "legal_dummy_corpus.json") -> Dict:
    """Ensure the corpus exists, generate if needed"""
    if not os.path.exists(corpus_file):
        print(f"Corpus file not found at {corpus_file}. Generating dummy legal corpus...")
        corpus = generate_legal_corpus(120, corpus_file)
    else:
        print(f"Loading corpus from {corpus_file}...")
        with open(corpus_file, 'r') as f:
            corpus = json.load(f)
    
    print(f"Corpus has {len(corpus['documents'])} documents")
    return corpus

def ensure_bm25_index_exists(bm25_retriever, corpus_file: str = "legal_dummy_corpus.json"):
    """Ensure the BM25 index exists, build if needed"""
    index_path = os.path.join(bm25_retriever.index_dir, "bm25.pkl")
    
    if not os.path.exists(index_path):
        print(f"BM25 index not found at {index_path}. Building index...")
        # Load or generate corpus
        corpus = ensure_corpus_exists(corpus_file)
        
        # Build the index
        bm25_retriever.index_corpus(corpus["documents"], corpus["doc_ids"])
        print(f"BM25 index built successfully at {index_path}")
    else:
        print(f"Loading existing BM25 index from {index_path}")
        bm25_retriever.load_index()

def ensure_dense_index_exists(dense_retriever, corpus_file: str = "legal_dummy_corpus.json"):
    """Ensure the Dense index exists, build if needed"""
    index_path = os.path.join(dense_retriever.index_dir, "faiss.index")
    
    if not os.path.exists(index_path):
        print(f"Dense index not found at {index_path}. Building index...")
        # Load or generate corpus
        corpus = ensure_corpus_exists(corpus_file)
        
        # Build the index
        start_time = time.time()
        print("This may take several minutes. Please be patient...")
        dense_retriever.index_corpus(corpus["documents"], corpus["doc_ids"])
        elapsed = time.time() - start_time
        print(f"Dense index built successfully at {index_path} in {elapsed:.2f} seconds")
    else:
        print(f"Loading existing Dense index from {index_path}")
        dense_retriever.load_index()

def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run LegalMind retrieval experiments")
    parser.add_argument("--bm25_index", default="legal_bm25_retr1", help="BM25 index name")
    parser.add_argument("--dense_index", default="legal_dense_retr2", help="Dense index name")
    parser.add_argument("--corpus", default="legal_dummy_corpus.json", help="Corpus file")
    parser.add_argument("--queries", default="legal_sample_queries.json", help="Queries file")
    parser.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Reranker model name")
    parser.add_argument("--retriever_k", type=int, default=100, help="Number of documents to retrieve")
    parser.add_argument("--reranker_k", type=int, default=10, help="Number of documents after reranking")
    parser.add_argument("--output_dir", default="experiment_results", help="Output directory")
    parser.add_argument("--exp3", action="store_true", help="Run Experiment 3 (BM25 + Reranker)")
    parser.add_argument("--exp4", action="store_true", help="Run Experiment 4 (Dense + Reranker)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Make sure we have a corpus file
    corpus = ensure_corpus_exists(args.corpus)
    
    # Load queries (or generate if not exists)
    if not os.path.exists(args.queries):
        print(f"Query file not found. Using default queries...")
        queries = [
            "What are the essential elements of a valid contract?",
            "How is negligence defined in tort law?",
            "What constitutes probable cause for a search warrant?",
            "What rights are protected under the First Amendment?",
            "How does adverse possession work in property law?"
        ]
        with open(args.queries, 'w') as f:
            json.dump(queries, f, indent=2)
        print(f"Created default queries file at {args.queries}")
    else:
        print(f"Loading queries from {args.queries}")
    
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    # Initialize shared components
    reranker = LegalCrossEncoder(
        model_name=args.reranker_model,
        max_length=512
    )
    
    results = {}
    
    # Run Experiment 3: BM25 + Reranker
    if args.exp3:
        print("\n=== Running Experiment 3: BM25 + Cross-Encoder Reranker ===\n")
        # Initialize BM25 retriever
        bm25_retriever = SimpleBM25Retriever(index_name=args.bm25_index)
        
        # Ensure BM25 index exists or build it
        ensure_bm25_index_exists(bm25_retriever, args.corpus)
        
        # Create the pipeline
        exp3_pipeline = BM25RerankerPipeline(
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            retriever_k=args.retriever_k,
            reranker_k=args.reranker_k
        )
        
        # Run the search
        exp3_results = exp3_pipeline.batch_search(queries)
        exp3_pipeline.save_results(
            exp3_results, 
            os.path.join(args.output_dir, "exp3_results.json")
        )
        
        results["BM25_Reranker"] = exp3_results
        
        # Print sample results
        print("\n=== Sample Results from BM25 + Reranker ===")
        sample_query = queries[0]
        sample_results = exp3_results["0"]["results"][:3]  # First 3 results of first query
        print(f"Query: {sample_query}")
        for i, doc in enumerate(sample_results):
            print(f"  Result {i+1}: Score={doc['score']:.4f}, Original={doc.get('original_score', 0):.4f}")
            print(f"  ID: {doc['id']}")
            print(f"  Text: {doc['text'][:150]}...\n")
    
    # Run Experiment 4: Dense + Reranker
    if args.exp4 and dense_available:
        try:
            print("\n=== Running Experiment 4: Dense Retriever + Cross-Encoder Reranker ===\n")
            
            # Initialize dense retriever without running problematic code
            dense_retriever = DenseRetriever(
                model_name="nlpaueb/legal-bert-base-uncased",
                index_name=args.dense_index
            )
            
            # Ensure Dense index exists or build it
            ensure_dense_index_exists(dense_retriever, args.corpus)
            
            # Create the pipeline
            exp4_pipeline = DenseRerankerPipeline(
                dense_retriever=dense_retriever,
                reranker=reranker,
                retriever_k=args.retriever_k,
                reranker_k=args.reranker_k
            )
            
            # Run the search
            exp4_results = exp4_pipeline.batch_search(queries)
            exp4_pipeline.save_results(
                exp4_results, 
                os.path.join(args.output_dir, "exp4_results.json")
            )
            
            results["Dense_Reranker"] = exp4_results
            
            # Print sample results
            print("\n=== Sample Results from Dense + Reranker ===")
            sample_query = queries[0]
            sample_results = exp4_results["0"]["results"][:3]  # First 3 results of first query
            print(f"Query: {sample_query}")
            for i, doc in enumerate(sample_results):
                print(f"  Result {i+1}: Score={doc['score']:.4f}, Original={doc.get('original_score', 0):.4f}")
                print(f"  ID: {doc['id']}")
                print(f"  Text: {doc['text'][:150]}...\n")
            
        except Exception as e:
            print(f"Error running Experiment 4: {e}")
            print("Skipping Experiment 4.")
            import traceback
            traceback.print_exc()
    elif args.exp4:
        print("\nDense retriever modules not available. Skipping Experiment 4.")
        print("Check your dense_retriever.py file for issues.")
    
    # Evaluate and compare methods
    if len(results) > 0:
        print("\n=== Evaluating and Comparing Methods ===\n")
        evaluator = RetrievalEvaluator(output_dir=args.output_dir)
        comparison = evaluator.compare_methods(results)
        evaluator.save_comparison(comparison)
        
        # Try to plot if possible
        try:
            evaluator.plot_timing_comparison(comparison)
        except Exception as e:
            print(f"Could not create plots: {e}")
        
        print("\nExperiments completed successfully!")
        print(f"Results saved to {args.output_dir}/")
    else:
        print("No experiments were run. Use --exp3 and/or --exp4 flags to run experiments.")

if __name__ == "__main__":
    main()
