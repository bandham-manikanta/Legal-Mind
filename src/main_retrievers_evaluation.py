"""
Legal Document Retrieval System Evaluation

This script provides a simple entry point to run the comprehensive evaluation
of four retrieval approaches:
1. BM25
2. Dense Retrieval
3. BM25 + Cross-encoder Reranker
4. Dense Retrieval + Cross-encoder Reranker

Usage:
  python main.py --config config.json  # Run with config file
  python main.py --skip-evaluation      # Only visualize latest results
"""

import sys
import os
from config_loader import get_config
from run_evaluation import run_evaluation
from visualization_script import main as visualize_results

def main():
    """Main function"""
    # Get configuration
    config = get_config()
    
    print("\n" + "="*50)
    print("LEGAL DOCUMENT RETRIEVAL EVALUATION")
    print("="*50)
    
    print(f"Configuration settings:")
    print(f"- Data path: {config.data_path}")
    print(f"- Number of queries: {config.num_queries}")
    print(f"- Dense model: {config.dense_model}")
    print(f"- Reranker model: {config.reranker_model}")
    print(f"- First-stage retrieval k: {config.retriever_k}")
    print(f"- Reranker k: {config.reranker_k}")
    
    # Visualize only if skip-evaluation is set
    if hasattr(config, 'skip_evaluation') and config.skip_evaluation:
        print("\nSkipping evaluation, visualizing latest results...")
        visualize_results(None)  # None means use the latest results file
        return
    
    # Run full evaluation
    results = run_evaluation(config)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()
