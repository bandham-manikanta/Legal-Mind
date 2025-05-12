import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from tqdm import tqdm

# Import your components
from ret1_bm25 import SimpleBM25Retriever
from ret2_dense import DenseRetriever
from exp3_bm25_rerank import BM25RerankerPipeline
from exp4_dense_rerank import DenseRerankerPipeline
from reranker import LegalCrossEncoder # Assuming reranker.py.py is named reranker.py
import data_utils # You MUST have this file

# K_VALUES will be loaded from config, but define a default if needed
DEFAULT_K_VALUES = [5, 10, 20]

class RetrievalEvaluator:
    def __init__(self, k_values: List[int]):
        self.k_values = sorted(k_values)
        self.results_agg = {}

    def _calculate_metrics_for_query(self, retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], timing_info: Dict) -> Dict:
        query_metrics = {
            'precision': {}, 'recall': {}, 'f1_score': {}, 
            'average_precision': 0.0, 
            'time_taken': timing_info.get('total_time', timing_info.get('retrieve_time', 0.0))
        }
        
        if not relevant_doc_ids: # No relevant documents for this query, metrics are tricky or undefined.
            for k in self.k_values:
                query_metrics['precision'][str(k)] = 0.0
                query_metrics['recall'][str(k)] = 0.0 # Or 1.0 if no relevant docs means all (zero) are found
                query_metrics['f1_score'][str(k)] = 0.0
            # AP is typically 0 if no relevant documents exist, or undefined.
            query_metrics['average_precision'] = 0.0 
            return query_metrics

        # Calculate P@k, R@k, F1@k
        for k_val in self.k_values:
            retrieved_at_k = retrieved_doc_ids[:k_val]
            if not retrieved_at_k:
                query_metrics['precision'][str(k_val)] = 0.0
                query_metrics['recall'][str(k_val)] = 0.0
                query_metrics['f1_score'][str(k_val)] = 0.0
                continue

            num_relevant_retrieved_at_k = len(set(retrieved_at_k).intersection(relevant_doc_ids))
            
            # Precision@k
            p_at_k = num_relevant_retrieved_at_k / len(retrieved_at_k)
            query_metrics['precision'][str(k_val)] = p_at_k
            
            # Recall@k
            r_at_k = num_relevant_retrieved_at_k / len(relevant_doc_ids)
            query_metrics['recall'][str(k_val)] = r_at_k
            
            # F1 Score@k
            if p_at_k + r_at_k > 0:
                query_metrics['f1_score'][str(k_val)] = 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)
            else:
                query_metrics['f1_score'][str(k_val)] = 0.0

        # Average Precision (AP) for the query
        hits = 0
        sum_precision_at_hits = 0.0
        for i, doc_id in enumerate(retrieved_doc_ids): # Consider all retrieved docs for AP
            rank = i + 1
            if doc_id in relevant_doc_ids:
                hits += 1
                precision_at_this_hit = hits / rank
                sum_precision_at_hits += precision_at_this_hit
        
        query_metrics['average_precision'] = sum_precision_at_hits / len(relevant_doc_ids) if hits > 0 else 0.0
        
        return query_metrics

    def evaluate_system(self, system_name: str, retrieval_function, 
                        queries_to_evaluate: Dict[str, str], 
                        relevance_judgments: Dict[str, Set[str]]):
        print(f"\n--- Evaluating system: {system_name} ---")
        
        all_query_metrics = []
        
        for query_id, query_text in tqdm(queries_to_evaluate.items(), desc=f"Queries for {system_name}"):
            relevant_docs_for_query = relevance_judgments.get(query_id, set())
            
            # retrieval_function must return: (list_of_retrieved_docs, timing_dict)
            # list_of_retrieved_docs: [{'id': 'doc1', 'score': 0.9, ...}, ...]
            # timing_dict: {'retrieve_time': X, 'rerank_time': Y (optional), 'total_time': Z}
            retrieved_docs_with_scores, timing_info = retrieval_function(query_text)
            retrieved_doc_ids = [doc['id'] for doc in retrieved_docs_with_scores]

            query_eval_metrics = self._calculate_metrics_for_query(retrieved_doc_ids, relevant_docs_for_query, timing_info)
            query_eval_metrics['query_id'] = query_id # For detailed logging
            all_query_metrics.append(query_eval_metrics)

        # Aggregate metrics
        aggregated_metrics = {'system_name': system_name, 'num_queries': len(queries_to_evaluate)}
        aggregated_metrics['precision_at_k'] = {str(k): np.mean([m['precision'][str(k)] for m in all_query_metrics]) for k in self.k_values}
        aggregated_metrics['recall_at_k'] = {str(k): np.mean([m['recall'][str(k)] for m in all_query_metrics]) for k in self.k_values}
        aggregated_metrics['f1_score_at_k'] = {str(k): np.mean([m['f1_score'][str(k)] for m in all_query_metrics]) for k in self.k_values}
        aggregated_metrics['map'] = np.mean([m['average_precision'] for m in all_query_metrics])
        aggregated_metrics['average_retrieval_time'] = np.mean([m['time_taken'] for m in all_query_metrics])
        
        # Store detailed per-query results if needed, for now, just aggregated
        self.results_agg[system_name] = aggregated_metrics 
        # self.results_detailed[system_name] = all_query_metrics # Optionally store for deeper analysis

        print(f"Results for {system_name}:")
        print(f"  MAP: {aggregated_metrics['map']:.4f}")
        for k in self.k_values:
            print(f"  P@{k}: {aggregated_metrics['precision_at_k'][str(k)]:.4f}, R@{k}: {aggregated_metrics['recall_at_k'][str(k)]:.4f}, F1@{k}: {aggregated_metrics['f1_score_at_k'][str(k)]:.4f}")
        print(f"  Avg. Time: {aggregated_metrics['average_retrieval_time']:.4f}s")
        
        return aggregated_metrics

    def get_all_aggregated_metrics(self):
        return self.results_agg
    
    # Inside RetrievalEvaluator class in evaluation_script.py
    def save_results(self, output_file_path: str):
        """Saves the aggregated evaluation results to a JSON file."""
        if not self.results_agg:
            print("No results to save.")
            return
        try:
            with open(output_file_path, 'w') as f:
                json.dump(self.results_agg, f, indent=4)
            print(f"Aggregated evaluation results saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving results to {output_file_path}: {e}")


def run_evaluation(config):
    """
    Main function to run the full evaluation pipeline.
    """
    ROOT_DIR = data_utils.find_project_root() # Ensure this works
    DATA_FILE_PATH = os.path.join(ROOT_DIR, config.data_path) if not os.path.isabs(config.data_path) else config.data_path
    BM25_INDEX_DIR = os.path.join(ROOT_DIR, config.bm25_index_dir)
    DENSE_INDEX_DIR = os.path.join(ROOT_DIR, config.dense_index_dir)
    EVAL_RESULTS_DIR = os.path.join(ROOT_DIR, config.results_dir)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    # Ensure load_casehold returns these four distinct elements:
    # 1. corpus_documents: List[Dict{'id':str, 'text':str}] for indexing
    # 2. corpus_doc_ids: List[str] - corresponding IDs
    # 3. test_queries: Dict[str, str] - query_id to query_text for evaluation
    # 4. relevance_judgments: Dict[str, Set[str]] - query_id to set of relevant doc_ids
    corpus_docs_list_of_dicts, corpus_doc_ids_list, test_queries_dict, relevance_judgments_dict = \
        data_utils.load_casehold(
            DATA_FILE_PATH, 
            num_queries=config.num_queries, 
            random_seed=config.evaluation_random_seed # from config: "random_seed"
        )
    
    corpus_texts_list = [doc['text'] for doc in corpus_docs_list_of_dicts]

    # --- Initialize Retrievers and Build/Load Indexes ---
    print("\nInitializing BM25 retriever...")
    bm25_retriever = SimpleBM25Retriever(index_name=BM25_INDEX_DIR)
    # Check if index exists (e.g., by checking for a specific file)
    # This check might need to be more robust based on how SimpleBM25Retriever saves.
    if not os.path.exists(os.path.join(bm25_retriever.index_dir, "bm25_model.pkl")): 
        print("Building BM25 index...")
        bm25_retriever.index_corpus(documents=corpus_texts_list, doc_ids=corpus_doc_ids_list)
    else:
        print("Loading existing BM25 index...")
        bm25_retriever.load_index()
        # CRITICAL: Ensure documents and doc_ids are available after loading index
        # If load_index() doesn't restore them, set them manually:
        bm25_retriever.documents = corpus_texts_list
        bm25_retriever.doc_ids = corpus_doc_ids_list

    print("\nInitializing Dense retriever...")
    dense_retriever = DenseRetriever(model_name=config.dense_model, index_name=DENSE_INDEX_DIR)
    # Similar check for dense index
    if not os.path.exists(os.path.join(dense_retriever.index_dir, "faiss.index")):
        print("Building Dense index...")
        dense_retriever.index_corpus(documents=corpus_texts_list, doc_ids=corpus_doc_ids_list, batch_size=config.batch_size)
    else:
        print("Loading existing Dense index...")
        dense_retriever.load_index()
        # CRITICAL: Ensure documents and doc_ids are available
        dense_retriever.documents = corpus_texts_list # Must match the order in the index
        dense_retriever.doc_ids = corpus_doc_ids_list

    print("\nInitializing Reranker...")
    reranker = LegalCrossEncoder(model_name=config.reranker_model, batch_size=config.batch_size)

    print("\nInitializing Pipelines...")
    # Pipelines use retriever_k for first pass, reranker_k for final output
    bm25_rerank_pipeline = BM25RerankerPipeline(
        bm25_retriever=bm25_retriever, reranker=reranker,
        retriever_k=config.retriever_k, reranker_k=config.reranker_k
    )
    dense_rerank_pipeline = DenseRerankerPipeline(
        dense_retriever=dense_retriever, reranker=reranker,
        retriever_k=config.retriever_k, reranker_k=config.reranker_k
    )

    # --- Define Wrapper Functions for Evaluation ---
    # These wrappers ensure a consistent (results, timing_dict) return type.
    # The 'k' for retrieval here should be enough for the largest K_VALUE in evaluation_k_values.
    # config.reranker_k or max(config.evaluation_k_values) would be a good choice.
    # If a retriever always returns more, that's fine; metrics will be sliced.
    eval_k_for_retrieval = max(config.evaluation_k_values) # e.g., 20
    # If pipelines' reranker_k is already this value, great.

    def bm25_retrieve_eval_wrapper(query_text: str):
        start_time = time.time()
        # BM25 retriever's own 'k' param in retrieve() method
        results = bm25_retriever.retrieve(query_text, k=eval_k_for_retrieval) 
        end_time = time.time()
        timing = {'retrieve_time': end_time - start_time, 'total_time': end_time - start_time}
        return results, timing

    def dense_retrieve_eval_wrapper(query_text: str):
        start_time = time.time()
        results = dense_retriever.retrieve(query_text, k=eval_k_for_retrieval)
        end_time = time.time()
        timing = {'retrieve_time': end_time - start_time, 'total_time': end_time - start_time}
        return results, timing

    def bm25_rerank_eval_wrapper(query_text: str):
        # The pipeline's search method already returns (results, timing_dict)
        # Ensure its reranker_k is appropriate (e.g. >= eval_k_for_retrieval)
        return bm25_rerank_pipeline.search(query_text)

    def dense_rerank_eval_wrapper(query_text: str):
        return dense_rerank_pipeline.search(query_text)

    # --- Perform Evaluation ---
    evaluator = RetrievalEvaluator(k_values=config.evaluation_k_values)
    
    systems_to_evaluate = {
        "BM25": bm25_retrieve_eval_wrapper,
        "Dense Retriever": dense_retrieve_eval_wrapper,
        "BM25 + Reranker": bm25_rerank_eval_wrapper,
        "Dense Retriever + Reranker": dense_rerank_eval_wrapper
    }

    for name, func in systems_to_evaluate.items():
        evaluator.evaluate_system(name, func, test_queries_dict, relevance_judgments_dict)
    
    all_system_metrics = evaluator.get_all_aggregated_metrics()

    # Save results
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    results_file_name = f"retrieval_eval_results_{timestamp}.json"
    results_file_path = os.path.join(EVAL_RESULTS_DIR, results_file_name)
    
    with open(results_file_path, 'w') as f:
        json.dump(all_system_metrics, f, indent=4)
    print(f"\nEvaluation results saved to: {results_file_path}")
    
    return results_file_path

# Add this function to evaluation_script.py
# Make sure to import any necessary libraries like json or random at the top of evaluation_script.py

import json # Make sure json is imported
import random # If you use num_queries and random sampling

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
    print(f"Loading evaluation data from: {data_file_path}")
    all_queries = {}
    all_relevance_judgments = {}

    # --- BEGIN Placeholder Logic ---
    # You MUST replace this with logic specific to your data file format.
    # Example assumes a JSONL file where each line is a JSON object
    # with fields like 'query_id', 'query_text', and 'relevant_doc_ids' (a list).

    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    query_id = str(record.get("query_id")) # Ensure query_id is a string
                    query_text = record.get("query_text")
                    # Assuming 'relevant_doc_ids' is a list of document IDs
                    relevant_docs_list = record.get("relevant_doc_ids", [])

                    if query_id and query_text and query_id != "None": # Basic validation
                        all_queries[query_id] = query_text
                        all_relevance_judgments[query_id] = set(str(doc_id) for doc_id in relevant_docs_list) # Ensure doc_ids are strings
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line.strip()}")
                except Exception as e:
                    print(f"Warning: Error processing record: {record} - {e}")


    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {data_file_path}")
        return {}, {}
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return {}, {}

    if not all_queries:
        print("Error: No queries were loaded. Please check the data file and parsing logic.")
        return {}, {}

    # --- END Placeholder Logic ---

    if num_queries is not None and num_queries < len(all_queries):
        print(f"Randomly selecting {num_queries} queries out of {len(all_queries)} using seed {random_seed}.")
        random.seed(random_seed)
        selected_query_ids = random.sample(list(all_queries.keys()), num_queries)
        
        test_queries = {qid: all_queries[qid] for qid in selected_query_ids}
        relevance_judgments = {qid: all_relevance_judgments.get(qid, set()) for qid in selected_query_ids}
    else:
        test_queries = all_queries
        relevance_judgments = all_relevance_judgments
        if num_queries is not None:
            print(f"Requested {num_queries} queries, but only {len(all_queries)} were available. Using all available queries.")


    print(f"Loaded {len(test_queries)} queries for evaluation.")
    # Sanity check: print a few loaded queries and their relevant docs
    # count_relevant = 0
    # for q_id in list(test_queries.keys())[:3]:
    #     print(f"  Query ID: {q_id}, Text: {test_queries[q_id][:50]}..., Relevant IDs: {list(relevance_judgments.get(q_id, set()))[:3]}")
    #     if relevance_judgments.get(q_id):
    #         count_relevant +=1
    # if not count_relevant:
    #     print("Warning: No relevant documents found for the sampled queries. Check relevance data.")


    return test_queries, relevance_judgments