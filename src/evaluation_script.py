import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from tqdm import tqdm
import math # Needed for log in NDCG
import random

# Import your components (ensure these paths are correct for your project structure)
# from ret1_bm25 import SimpleBM25Retriever
# from ret2_dense import DenseRetriever
# from exp3_bm25_rerank import BM25RerankerPipeline
# from exp4_dense_rerank import DenseRerankerPipeline
# from reranker import LegalCrossEncoder
# import data_utils # You MUST have this file

# K_VALUES will be loaded from config, but define a default if needed
DEFAULT_K_VALUES = [5, 10, 20]

class RetrievalEvaluator:
    def __init__(self, k_values: List[int]):
        """
        Initializes the evaluator.
        Args:
            k_values (List[int]): A list of k values for which to compute metrics (e.g., [5, 10, 20]).
        """
        self.k_values = sorted(list(set(k_values))) # Ensure k_values are sorted and unique
        self.results_agg = {} # To store aggregated results per system

    def _dcg_at_k(self, ranked_relevance: List[int], k: int) -> float:
        """
        Calculates Discounted Cumulative Gain at k.
        Assumes binary relevance (0 or 1).
        Args:
            ranked_relevance (List[int]): List of relevance scores (0 or 1) in ranked order.
            k (int): The cut-off for calculation.
        Returns:
            float: DCG@k score.
        """
        dcg = 0.0
        for i in range(min(k, len(ranked_relevance))):
            relevance = ranked_relevance[i]
            dcg += relevance / math.log2(i + 2) # i+2 because rank is i+1, log base 2 of (rank+1)
        return dcg

    def _ndcg_at_k(self, retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain at k.
        Assumes binary relevance.
        Args:
            retrieved_doc_ids (List[str]): List of retrieved document IDs in order.
            relevant_doc_ids (Set[str]): Set of true relevant document IDs.
            k (int): The cut-off for calculation.
        Returns:
            float: NDCG@k score.
        """
        # Get relevance scores for retrieved docs (1 if relevant, 0 if not)
        ranked_relevance = [1 if doc_id in relevant_doc_ids else 0 for doc_id in retrieved_doc_ids]
        
        dcg_at_k = self._dcg_at_k(ranked_relevance, k)
        
        # Ideal DCG: sort all relevant items first (up to k)
        num_relevant = len(relevant_doc_ids)
        ideal_ranked_relevance = [1] * min(num_relevant, k)
        if len(ideal_ranked_relevance) < k :
            ideal_ranked_relevance.extend([0] * (k - len(ideal_ranked_relevance)))

        idcg_at_k = self._dcg_at_k(ideal_ranked_relevance, k)
        
        return dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0

    def _reciprocal_rank(self, retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
        """
        Calculates the Reciprocal Rank for a single query.
        Args:
            retrieved_doc_ids (List[str]): List of retrieved document IDs in order.
            relevant_doc_ids (Set[str]): Set of true relevant document IDs.
        Returns:
            float: Reciprocal rank (1/rank of first relevant item, or 0 if none found).
        """
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                return 1.0 / (i + 1) # Rank is i+1
        return 0.0

    def _calculate_metrics_for_query(self, retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], timing_info: Dict) -> Dict:
        """
        Calculates all metrics for a single query.
        """
        query_metrics = {
            'precision_at_k': {}, 
            'recall_at_k': {}, 
            'f1_score_at_k': {},
            'ndcg_at_k': {}, # Added for NDCG
            'average_precision': 0.0,
            'reciprocal_rank': 0.0, # Added for MRR
            'time_taken': timing_info.get('total_time', timing_info.get('retrieve_time', 0.0))
        }
        
        if not relevant_doc_ids: # No relevant documents for this query
            for k_val in self.k_values:
                # Note: The original pasted code used 'precision', 'recall', 'f1_score' directly as keys.
                # For consistency with how P@k, R@k etc. are usually structured and with the Canvas version,
                # I'm using 'precision_at_k', 'recall_at_k', 'f1_score_at_k'.
                query_metrics['precision_at_k'][str(k_val)] = 0.0
                query_metrics['recall_at_k'][str(k_val)] = 0.0 
                query_metrics['f1_score_at_k'][str(k_val)] = 0.0
                query_metrics['ndcg_at_k'][str(k_val)] = 0.0
            query_metrics['average_precision'] = 0.0 
            query_metrics['reciprocal_rank'] = 0.0
            return query_metrics

        # Calculate P@k, R@k, F1@k, NDCG@k
        for k_val in self.k_values:
            retrieved_at_k = retrieved_doc_ids[:k_val]
            if not retrieved_at_k: # No documents retrieved up to k
                query_metrics['precision_at_k'][str(k_val)] = 0.0
                query_metrics['recall_at_k'][str(k_val)] = 0.0
                query_metrics['f1_score_at_k'][str(k_val)] = 0.0
                query_metrics['ndcg_at_k'][str(k_val)] = 0.0
                continue

            num_relevant_retrieved_at_k = len(set(retrieved_at_k).intersection(relevant_doc_ids))
            
            p_at_k = num_relevant_retrieved_at_k / len(retrieved_at_k) if len(retrieved_at_k) > 0 else 0.0
            query_metrics['precision_at_k'][str(k_val)] = p_at_k
            
            r_at_k = num_relevant_retrieved_at_k / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0.0
            query_metrics['recall_at_k'][str(k_val)] = r_at_k
            
            if p_at_k + r_at_k > 0:
                query_metrics['f1_score_at_k'][str(k_val)] = 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)
            else:
                query_metrics['f1_score_at_k'][str(k_val)] = 0.0
            
            query_metrics['ndcg_at_k'][str(k_val)] = self._ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k_val)

        # Average Precision (AP) for the query
        hits = 0
        sum_precision_at_hits = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids):
            rank = i + 1
            if doc_id in relevant_doc_ids:
                hits += 1
                precision_at_this_hit = hits / rank
                sum_precision_at_hits += precision_at_this_hit
        
        query_metrics['average_precision'] = sum_precision_at_hits / len(relevant_doc_ids) if hits > 0 and len(relevant_doc_ids) > 0 else 0.0
        
        query_metrics['reciprocal_rank'] = self._reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        
        return query_metrics

    def evaluate_system(self, system_name: str, retrieval_function, 
                        queries_to_evaluate: Dict[str, str], 
                        relevance_judgments: Dict[str, Set[str]]):
        print(f"\n--- Evaluating system: {system_name} ---")
        
        if not queries_to_evaluate:
            print(f"No queries provided for evaluation of {system_name}. Skipping.")
            empty_metrics = {
                'system_name': system_name, 'num_queries': 0,
                'precision_at_k': {str(k): np.nan for k in self.k_values},
                'recall_at_k': {str(k): np.nan for k in self.k_values},
                'f1_score_at_k': {str(k): np.nan for k in self.k_values},
                'ndcg_at_k': {str(k): np.nan for k in self.k_values},
                'map': np.nan, 'mrr': np.nan,
                'average_retrieval_time': np.nan
            }
            self.results_agg[system_name] = empty_metrics
            return empty_metrics

        all_query_metrics_list = []
        
        for query_id, query_text in tqdm(queries_to_evaluate.items(), desc=f"Queries for {system_name}", unit="query"):
            relevant_docs_for_query = relevance_judgments.get(query_id, set())
            
            retrieved_docs_with_scores, timing_info = retrieval_function(query_text)
            retrieved_doc_ids = [doc['id'] for doc in retrieved_docs_with_scores]

            query_eval_metrics = self._calculate_metrics_for_query(retrieved_doc_ids, relevant_docs_for_query, timing_info)
            query_eval_metrics['query_id'] = query_id 
            all_query_metrics_list.append(query_eval_metrics)

        # Aggregate metrics
        aggregated_metrics = {'system_name': system_name, 'num_queries': len(queries_to_evaluate)}
        # Corrected keys for aggregation to match what _calculate_metrics_for_query produces
        aggregated_metrics['precision_at_k'] = {str(k): np.mean([m['precision_at_k'][str(k)] for m in all_query_metrics_list]) for k in self.k_values}
        aggregated_metrics['recall_at_k'] = {str(k): np.mean([m['recall_at_k'][str(k)] for m in all_query_metrics_list]) for k in self.k_values}
        aggregated_metrics['f1_score_at_k'] = {str(k): np.mean([m['f1_score_at_k'][str(k)] for m in all_query_metrics_list]) for k in self.k_values}
        aggregated_metrics['ndcg_at_k'] = {str(k): np.mean([m['ndcg_at_k'][str(k)] for m in all_query_metrics_list]) for k in self.k_values}
        aggregated_metrics['map'] = np.mean([m['average_precision'] for m in all_query_metrics_list])
        aggregated_metrics['mrr'] = np.mean([m['reciprocal_rank'] for m in all_query_metrics_list])
        aggregated_metrics['average_retrieval_time'] = np.mean([m['time_taken'] for m in all_query_metrics_list])
        
        self.results_agg[system_name] = aggregated_metrics 

        print(f"Results for {system_name}:")
        print(f"  MAP: {aggregated_metrics['map']:.4f}")
        print(f"  MRR: {aggregated_metrics['mrr']:.4f}")
        for k_val_str in aggregated_metrics['precision_at_k']:
            k_int = int(k_val_str)
            print(f"  Metrics@K={k_int}:")
            print(f"    P@{k_int}: {aggregated_metrics['precision_at_k'][k_val_str]:.4f}")
            print(f"    R@{k_int}: {aggregated_metrics['recall_at_k'][k_val_str]:.4f}")
            print(f"    F1@{k_int}: {aggregated_metrics['f1_score_at_k'][k_val_str]:.4f}")
            print(f"    NDCG@{k_int}: {aggregated_metrics['ndcg_at_k'][k_val_str]:.4f}")
        print(f"  Avg. Time: {aggregated_metrics['average_retrieval_time']:.4f}s")
        
        return aggregated_metrics

    def get_all_aggregated_metrics(self):
        return self.results_agg
    
    def save_results(self, output_file_path: str):
        """Saves the aggregated evaluation results to a JSON file."""
        if not self.results_agg:
            print("No results to save.")
            return
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                json.dump(self.results_agg, f, indent=4)
            print(f"Aggregated evaluation results saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving results to {output_file_path}: {e}")


# The following functions (run_evaluation and prepare_evaluation_data) are part of the user's pasted code.
# The prepare_evaluation_data function here is the one selected by the user in the Canvas.
# The run_evaluation function is from the user's pasted context, not necessarily the most up-to-date from other files.
# For this response, I am keeping them as they were in the user's pasted block,
# as the primary request was to add NDCG/MRR logic to the RetrievalEvaluator.

# It's assumed that data_utils.find_project_root() and data_utils.load_casehold()
# are available and correctly implemented in a data_utils.py file.
# Also, SimpleBM25Retriever, DenseRetriever, LegalCrossEncoder,
# BM25RerankerPipeline, DenseRerankerPipeline are assumed to be importable.

def run_evaluation(config): # This function is from the user's pasted code
    """
    Main function to run the full evaluation pipeline.
    """
    # This function would need access to data_utils, retrievers, rerankers, pipelines
    # For example:
    # ROOT_DIR = data_utils.find_project_root() 
    # DATA_FILE_PATH = os.path.join(ROOT_DIR, config.data_path) if not os.path.isabs(config.data_path) else config.data_path
    # ... and so on for other paths and initializations ...

    print("Illustrative run_evaluation: Actual implementation depends on your project files.")
    print("Loading data (using the prepare_evaluation_data from this script)...")
    
    # Use the prepare_evaluation_data function defined in this script (selected by user)
    # This path would come from 'config.data_path'
    # Ensure DATA_FILE_PATH is correctly defined based on your config object structure
    # For this example, let's assume config is a SimpleNamespace or dict with data_path
    
    data_file_path_for_eval = getattr(config, 'data_path', 'path/to/your/eval_data.jsonl')
    num_eval_queries = getattr(config, 'num_queries', 10) # Default to 10 if not in config
    eval_random_seed = getattr(config, 'evaluation_random_seed', 42)

    test_queries_dict, relevance_judgments_dict = prepare_evaluation_data(
        data_file_path_for_eval,
        num_queries=num_eval_queries,
        random_seed=eval_random_seed
    )

    if not test_queries_dict:
        print("No evaluation queries loaded by prepare_evaluation_data. Cannot proceed with evaluation.")
        return {}

    # --- Initialize Retrievers and Build/Load Indexes (Conceptual) ---
    # This section would require your actual retriever and pipeline classes
    # print("\nInitializing BM25 retriever...")
    # bm25_retriever = SimpleBM25Retriever(index_name=config.bm25_index_dir)
    # ... logic to build/load index ...

    # print("\nInitializing Dense retriever...")
    # dense_retriever = DenseRetriever(model_name=config.dense_model, index_name=config.dense_index_dir)
    # ... logic to build/load index ...
    
    # print("\nInitializing Reranker...")
    # reranker = LegalCrossEncoder(model_name=config.reranker_model, batch_size=config.batch_size)

    # print("\nInitializing Pipelines...")
    # bm25_rerank_pipeline = BM25RerankerPipeline(...)
    # dense_rerank_pipeline = DenseRerankerPipeline(...)

    # --- Define Wrapper Functions for Evaluation (Conceptual) ---
    eval_k_for_retrieval = max(getattr(config, 'evaluation_k_values', DEFAULT_K_VALUES))

    def mock_retrieval_wrapper(query_text: str): # Placeholder
        print(f"Mock retrieving for: {query_text[:30]}...")
        # Simulate some results
        mock_results = [{'id': f'doc{i}', 'score': 1.0/(i+1)} for i in range(eval_k_for_retrieval)]
        random.shuffle(mock_results)
        timing = {'retrieve_time': 0.01, 'total_time': 0.01}
        return mock_results, timing

    systems_to_evaluate = {
        "MockBM25": mock_retrieval_wrapper,
        # Add your actual system wrappers here, e.g.:
        # "BM25": bm25_retrieve_eval_wrapper,
        # "Dense Retriever": dense_retrieve_eval_wrapper,
        # "BM25 + Reranker": bm25_rerank_eval_wrapper,
        # "Dense Retriever + Reranker": dense_rerank_eval_wrapper
    }
    
    evaluator = RetrievalEvaluator(k_values=getattr(config, 'evaluation_k_values', DEFAULT_K_VALUES))
    
    for name, func in systems_to_evaluate.items():
        evaluator.evaluate_system(name, func, test_queries_dict, relevance_judgments_dict)
    
    all_system_metrics = evaluator.get_all_aggregated_metrics()

    # Save results (Conceptual path)
    results_dir = getattr(config, 'results_dir', 'evaluation_results_script_level')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    results_file_name = f"retrieval_eval_results_{timestamp}.json"
    results_file_path = os.path.join(results_dir, results_file_name)
    
    evaluator.save_results(results_file_path)
    print(f"\nEvaluation results saved to: {results_file_path}")
    
    return all_system_metrics


# This is the prepare_evaluation_data function that was selected in the Canvas.
# It is kept here as per the user's context.
# Note: `import random` and `import json` should be at the top of the file.
# (They are already there in this combined script)

def prepare_evaluation_data(data_file_path: str, num_queries: int = None, random_seed: int = 42) -> tuple[dict[str, str], dict[str, set[str]]]:
    """
    Loads evaluation data (queries and relevance judgments) from the specified file.

    Args:
        data_file_path: Path to the data file (e.g., a JSONL file).
        num_queries: Optional number of queries to select. If None, all are used.
        random_seed: Seed for random sampling if num_queries is specified.

    Returns:
        A tuple containing:
            - test_queries (Dict[str, str]): query_id -> query_text
            - relevance_judgments (Dict[str, Set[str]]): query_id -> set of relevant_doc_ids
    """
    print(f"Loading evaluation data from: {data_file_path}") # Changed from "Placeholder: Loading..."
    all_queries = {}
    all_relevance_judgments = {}

    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Adjusted heuristic for early stopping if num_queries is met
                if num_queries is not None and len(all_queries) >= num_queries:
                     # Only break if we have actually collected enough unique queries
                    break 
                try:
                    record = json.loads(line.strip())
                    query_id = str(record.get("query_id")) 
                    query_text = record.get("query_text")
                    relevant_docs_list = record.get("relevant_doc_ids", [])

                    if query_id and query_text and query_id != "None": 
                        if query_id not in all_queries: # Ensure unique queries if sampling later
                             all_queries[query_id] = query_text
                             all_relevance_judgments[query_id] = set(str(doc_id) for doc_id in relevant_docs_list)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in evaluation data: {line.strip()}")
                # Removed the generic Exception catch here to allow more specific ones above.
    except FileNotFoundError:
        print(f"CRITICAL Error: Evaluation data file not found at {data_file_path}")
        return {}, {}
    except Exception as e: # Catch other potential errors during file operations
        print(f"CRITICAL Error loading evaluation data: {e}")
        return {}, {}


    if not all_queries:
        print("Error: No queries were loaded. Please check the data file and parsing logic.")
        return {}, {}

    # Sampling logic
    if num_queries is not None and num_queries < len(all_queries):
        print(f"Randomly selecting {num_queries} queries out of {len(all_queries)} using seed {random_seed}.")
        if random_seed is not None: # Ensure random seed is used if provided
            random.seed(random_seed)
        selected_query_ids = random.sample(list(all_queries.keys()), num_queries)
        
        test_queries = {qid: all_queries[qid] for qid in selected_query_ids}
        relevance_judgments = {qid: all_relevance_judgments.get(qid, set()) for qid in selected_query_ids}
    else:
        test_queries = all_queries
        relevance_judgments = all_relevance_judgments
        if num_queries is not None and len(all_queries) < num_queries :
            print(f"Warning: Requested {num_queries} queries, but only {len(all_queries)} were available. Using all available queries.")


    print(f"Loaded {len(test_queries)} queries for evaluation.")
    if not test_queries:
        print("CRITICAL Warning: No queries loaded. Evaluation will be empty.")
    elif not any(relevance_judgments.values()): # Check if any query has any relevant documents
        print("Warning: No relevant documents found for any loaded queries. Recall, MAP, MRR, NDCG will be affected (likely 0).")
        
    return test_queries, relevance_judgments

if __name__ == '__main__':
    # This main block is for testing the RetrievalEvaluator and prepare_evaluation_data directly.
    # It won't run the full pipeline from run_evaluation unless you set up a mock config.
    print("Running a minimal example of RetrievalEvaluator with prepare_evaluation_data...")

    # Create a dummy evaluation data file for testing prepare_evaluation_data
    dummy_eval_file_content = [
        {"query_id": "q1", "query_text": "example query one", "relevant_doc_ids": ["doc1", "doc3", "doc5"]},
        {"query_id": "q2", "query_text": "example query two", "relevant_doc_ids": ["docA", "docB"]},
        {"query_id": "q3", "query_text": "query with no relevant docs", "relevant_doc_ids": []},
        {"query_id": "q4", "query_text": "another query", "relevant_doc_ids": ["docX", "docY", "docZ", "doc1"]},
    ]
    dummy_eval_file_path = "temp_dummy_eval_data.jsonl"
    with open(dummy_eval_file_path, 'w') as f:
        for item in dummy_eval_file_content:
            f.write(json.dumps(item) + '\n')

    # Test prepare_evaluation_data
    print("\n--- Testing prepare_evaluation_data ---")
    test_queries, test_relevance = prepare_evaluation_data(dummy_eval_file_path, num_queries=3, random_seed=42)
    print(f"Loaded test_queries: {len(test_queries)}")
    # print(f"Test queries: {test_queries}")
    # print(f"Test relevance: {test_relevance}")


    if test_queries: # Proceed with evaluator test only if queries were loaded
        # Mock data for demonstration
        mock_k_values = [2, 5] # Using different k for this test

        # Mock retrieval function
        def mock_retrieval_system_func(query_text_arg):
            _ = query_text_arg 
            retrieved_for_q1 = [{'id': 'doc1', 'score': 1.0}, {'id': 'doc2', 'score':0.9}, {'id': 'doc3', 'score':0.8}, {'id': 'doc4', 'score':0.7}, {'id': 'doc5', 'score':0.6}]
            retrieved_for_q2 = [{'id': 'docX', 'score': 0.9}, {'id': 'docA', 'score': 0.8}, {'id': 'docY', 'score': 0.7}]
            retrieved_for_q3 = [{'id': 'non_relevant_doc1', 'score': 0.5}]
            retrieved_for_q4 = [{'id': 'doc1', 'score': 1.0}, {'id': 'docX', 'score':0.9}]


            # Simulate based on query_id if possible, or use a generic list
            # For simplicity, let's assume we can map query_text back to query_id for this mock
            if "one" in query_text_arg: current_retrieved = retrieved_for_q1
            elif "two" in query_text_arg: current_retrieved = retrieved_for_q2
            elif "no relevant" in query_text_arg: current_retrieved = retrieved_for_q3
            elif "another" in query_text_arg: current_retrieved = retrieved_for_q4
            else: current_retrieved = [{'id': f'rand_doc{j}', 'score': 0.1} for j in range(5)]
            
            random.shuffle(current_retrieved) # Add some randomness to ranks
            return current_retrieved, {'total_time': random.uniform(0.01, 0.05)}

        print("\n--- Testing RetrievalEvaluator ---")
        evaluator = RetrievalEvaluator(k_values=mock_k_values)
        evaluator.evaluate_system("MockSystemForTest", mock_retrieval_system_func, test_queries, test_relevance)
        
        all_results = evaluator.get_all_aggregated_metrics()
        print("\n--- All Aggregated Metrics (from test) ---")
        print(json.dumps(all_results, indent=4))
    else:
        print("\nSkipping RetrievalEvaluator test as no queries were loaded by prepare_evaluation_data.")

    # Clean up dummy file
    if os.path.exists(dummy_eval_file_path):
        os.remove(dummy_eval_file_path)
