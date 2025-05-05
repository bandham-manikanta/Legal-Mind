# src/demo.py
import os
import sys
# Ensure the 'src' directory is in the Python path if running from project root
# Or adjust imports if running directly from within 'src'
from data_utils import load_casehold, find_project_root
# Use the suggested new, clearer names if you renamed the files
from ret1_bm25 import SimpleBM25Retriever        # Was retriever_BM25.py
from ret2_dense import DenseRetriever            # Was modified_dense_retriever.py
from reranker import LegalCrossEncoder           # Unchanged
from exp3_bm25_rerank import BM25RerankerPipeline # Was experiment3.py
from exp4_dense_rerank import DenseRerankerPipeline # Was experiment4.py


# -------- 0. Configuration ----------
# Adjust paths if your structure differs or if running from a different directory
# Assumes 'data' directory is at the same level as 'src'
root_dir = find_project_root()
DATA_FILE_PATH = "data/casehold_test_processed.jsonl"
DATA_FILE_PATH = os.path.join(root_dir, DATA_FILE_PATH)
BM25_INDEX_DIR = "casehold_bm25_index" # Store index in project root
DENSE_INDEX_DIR = "casehold_dense_index" # Store index in project root

# -------- 1. Load corpus ----------
print(f"Loading CaseHOLD dataset from {DATA_FILE_PATH}...")
docs, doc_ids = load_casehold(path=DATA_FILE_PATH)
if not docs:
     print("Failed to load documents. Exiting.")
     sys.exit(1)
print(f"Loaded {len(docs)} documents.")

# -------- 2. Build / load indices --
print("\nInitializing BM25 Retriever...")
bm25 = SimpleBM25Retriever(index_name=BM25_INDEX_DIR)
# Check if the actual index *file* exists inside the directory
bm25_pkl_path = os.path.join(bm25.index_dir, "bm25.pkl")
if not os.path.exists(bm25_pkl_path):
    print(f"BM25 index file not found at {bm25_pkl_path}. Building index...")
    bm25.index_corpus(docs, doc_ids)
else:
    print(f"Loading existing BM25 index from {bm25.index_dir}...")
    bm25.load_index()

print("\nInitializing Dense Retriever...")
dense = DenseRetriever(model_name="nlpaueb/legal-bert-base-uncased",
                       index_name=DENSE_INDEX_DIR)
# Check if the actual index *file* exists inside the directory
faiss_index_path = os.path.join(dense.index_dir, "faiss.index")
if not os.path.exists(faiss_index_path):
    print(f"Dense index file not found at {faiss_index_path}. Building index (this may take a while)...")
    dense.index_corpus(docs, doc_ids, batch_size=16) # Added batch_size
else:
    print(f"Loading existing Dense index from {dense.index_dir}...")
    dense.load_index()

# -------- 3. Create reranker -------
print("\nInitializing Reranker...")
reranker = LegalCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

# Create pipelines using the initialized components
print("Creating Experiment Pipelines...")
pipe_exp3 = BM25RerankerPipeline(bm25_retriever=bm25,
                                 reranker=reranker,
                                 retriever_k=50,  # Retrieve top 50 from BM25
                                 reranker_k=5)    # Rerank to final top 5

pipe_exp4 = DenseRerankerPipeline(dense_retriever=dense,
                                  reranker=reranker,
                                  retriever_k=50, # Retrieve top 50 from Dense
                                  reranker_k=5)   # Rerank to final top 5

# -------- 4. Two sample queries ----
sample_queries = [
    "Did the trial court err in granting summary judgment?",
    "What is the standard for reviewing a jury instruction on negligence?"
]

# -------- 5. Run & print ----------
print("\nStarting Query Execution...")
TOP_K_DISPLAY = 3 # How many results to print for each method

for q_idx, q in enumerate(sample_queries):
    print("\n" + "="*80)
    print(f"QUERY {q_idx+1}: {q}")
    print("="*80)

    try:
        print(f"\n[RET-1] BM25 Results (Top {TOP_K_DISPLAY}):")
        results_bm25 = bm25.retrieve(q, k=TOP_K_DISPLAY)
        if not results_bm25: print("  No results found.")
        for r in results_bm25:
            print(f"  Score: {r['score']:.4f} | ID: {r['id']} | Text: {r['text'][:150].replace(os.linesep, ' ')}...") # Replaced newline for cleaner print

    except Exception as e:
        print(f"  Error running RET-1: {e}")

    try:
        print(f"\n[RET-2] Dense Retriever Results (Top {TOP_K_DISPLAY}):")
        results_dense = dense.retrieve(q, k=TOP_K_DISPLAY)
        if not results_dense: print("  No results found.")
        for r in results_dense:
            print(f"  Score: {r['score']:.4f} | ID: {r['id']} | Text: {r['text'][:150].replace(os.linesep, ' ')}...")

    except Exception as e:
         print(f"  Error running RET-2: {e}")

    try:
        print(f"\n[EXP-3] BM25 + Reranker Results (Top {TOP_K_DISPLAY}):")
        results_exp3, timing_exp3 = pipe_exp3.search(q) # Reranker_k already set to 5, display top 3 of those 5
        if not results_exp3: print("  No results found.")
        for r in results_exp3[:TOP_K_DISPLAY]:
            orig_score = r.get('original_score', 'N/A') # Handle if key missing
            orig_score_str = f"{orig_score:.4f}" if isinstance(orig_score, float) else str(orig_score)
            print(f"  Rerank Score: {r['score']:.4f} (BM25: {orig_score_str}) | ID: {r['id']} | Text: {r['text'][:150].replace(os.linesep, ' ')}...")
        print(f"  Timing: {timing_exp3}")

    except Exception as e:
         print(f"  Error running EXP-3: {e}")


    try:
        print(f"\n[EXP-4] Dense + Reranker Results (Top {TOP_K_DISPLAY}):")
        results_exp4, timing_exp4 = pipe_exp4.search(q) # Reranker_k already set to 5, display top 3 of those 5
        if not results_exp4: print("  No results found.")
        for r in results_exp4[:TOP_K_DISPLAY]:
            orig_score = r.get('original_score', 'N/A')
            orig_score_str = f"{orig_score:.4f}" if isinstance(orig_score, float) else str(orig_score)
            print(f"  Rerank Score: {r['score']:.4f} (Dense: {orig_score_str}) | ID: {r['id']} | Text: {r['text'][:150].replace(os.linesep, ' ')}...")
        print(f"  Timing: {timing_exp4}")

    except Exception as e:
         print(f"  Error running EXP-4: {e}")


print("\n" + "="*80)
print("Demo Finished.")
print("="*80)
