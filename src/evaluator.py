# evaluator.py
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


class RetrievalEvaluator:
    """
    Evaluator for comparing different retrieval methods.
    """
    
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_timing(self, results_dict: Dict[str, Dict[str, Any]], method_name: str) -> Dict[str, float]:
        """
        Evaluate timing statistics for a retrieval method.
        
        Args:
            results_dict: Results dictionary from batch_search
            method_name: Name of the retrieval method
            
        Returns:
            Dictionary of timing statistics
        """
        retrieve_times = []
        rerank_times = []
        total_times = []
        
        for query_id, data in results_dict.items():
            timing = data.get('timing', {})
            if timing:
                retrieve_times.append(timing.get('retrieve_time', 0))
                rerank_times.append(timing.get('rerank_time', 0))
                total_times.append(timing.get('total_time', 0))
        
        stats = {
            'method': method_name,
            'avg_retrieve_time': np.mean(retrieve_times) if retrieve_times else 0,
            'avg_rerank_time': np.mean(rerank_times) if rerank_times else 0,
            'avg_total_time': np.mean(total_times) if total_times else 0,
            'min_total_time': np.min(total_times) if total_times else 0,
            'max_total_time': np.max(total_times) if total_times else 0
        }
        
        return stats
    
    def compare_methods(self, methods_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple retrieval methods.
        
        Args:
            methods_results: Dictionary mapping method names to their batch results
            
        Returns:
            Comparison results
        """
        comparison = {
            'timing_stats': {},
            'query_counts': {},
        }
        
        for method_name, results in methods_results.items():
            # Timing statistics
            comparison['timing_stats'][method_name] = self.evaluate_timing(results, method_name)
            
            # Document counts
            query_counts = {}
            for query_id, data in results.items():
                query = data.get('query', f'Query {query_id}')
                doc_count = len(data.get('results', []))
                query_counts[query] = doc_count
            
            comparison['query_counts'][method_name] = query_counts
        
        return comparison
    
    def save_comparison(self, comparison: Dict[str, Any], filename: str = 'method_comparison.json'):
        """Save comparison results to file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved comparison results to {filepath}")
        
    def plot_timing_comparison(self, comparison: Dict[str, Any], filename: str = 'timing_comparison.png'):
        """
        Plot timing comparison between methods.
        """
        print("Plotting functionality requires matplotlib, which is optional.")
        print("To enable plots, install matplotlib: pip install matplotlib")
        print("Skipping plot generation.")
