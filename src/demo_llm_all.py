"""Callling LLMs to answer user queries using all available methods."""

import os
import sys
import time


from config import config

from data_utils import find_project_root, load_casehold
from exp3_bm25_rerank import BM25RerankerPipeline
from exp4_dense_rerank import DenseRerankerPipeline
from legal_llama3_70b_answerer import LegalLLAMA3_70B_Answerer  # Llama-3

from legal_nemotron_answerer import LegalNemotronAnswerer  # Nemotron
from legal_falcon3_answerer import LegalFalcon3_7B_Answerer  # Zephyr
from reranker import LegalCrossEncoder
from ret1_bm25 import SimpleBM25Retriever
from ret2_dense import DenseRetriever
from legal_deepseek_answerer import LegalDeepseekAnswerer

# ---------------------------------------------------------------------------
#  2.  Configuration
# ---------------------------------------------------------------------------
root_dir          = find_project_root()
DATA_FILE_PATH    = os.path.join(root_dir, "data/casehold_test_processed.jsonl")

BM25_INDEX_DIR    = os.path.join(root_dir, "casehold_bm25_index")
DENSE_INDEX_DIR   = os.path.join(root_dir, "casehold_dense_index")
TOP_K             = 5          # How many docs each LLM sees

NVIDIA_API_KEY    = config['NVIDIA_API_KEY']

# ---------------------------------------------------------------------------
#  3.  Load corpus & build / load indices
# ---------------------------------------------------------------------------
print("Loading CaseHOLD corpus …")
docs, doc_ids = load_casehold(DATA_FILE_PATH)
print(f"  {len(docs):,} passages loaded\n")

# ---- BM-25
bm25 = SimpleBM25Retriever(index_name=BM25_INDEX_DIR)
if not os.path.exists(os.path.join(BM25_INDEX_DIR, "bm25.pkl")):
    print("Building BM-25 index …")
    bm25.index_corpus(docs, doc_ids)
else:
    bm25.load_index()

# ---- Dense retriever
dense = DenseRetriever(model_name="nlpaueb/legal-bert-base-uncased",
                       index_name=DENSE_INDEX_DIR)
if not os.path.exists(os.path.join(DENSE_INDEX_DIR, "faiss.index")):
    print("Building Dense/FAISS index … (can take a while)")
    dense.index_corpus(docs, doc_ids, batch_size=16)
else:
    dense.load_index()

# ---- Cross-encoder reranker
reranker = LegalCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
)

pipe_bm25_rerank  = BM25RerankerPipeline(
    bm25_retriever=bm25,  reranker=reranker,
    retriever_k=50,  reranker_k=TOP_K
)
pipe_dense_rerank = DenseRerankerPipeline(
    dense_retriever=dense, reranker=reranker,
    retriever_k=50, reranker_k=TOP_K
)

# ---------------------------------------------------------------------------
#  4.  Instantiate the four LLM clients
# ---------------------------------------------------------------------------
nemotron = LegalNemotronAnswerer(api_key=NVIDIA_API_KEY)
llama3   = LegalLLAMA3_70B_Answerer(api_key=NVIDIA_API_KEY)
falcon  = LegalFalcon3_7B_Answerer(api_key=NVIDIA_API_KEY)
deepseek   = LegalDeepseekAnswerer(api_key=NVIDIA_API_KEY)



def build_prompt(query, context_docs):
    """Prompt bulding for LLMs."""
    context = "\n\n".join([f"Context {i+1}:\n{doc.strip()}" for i, doc in enumerate(context_docs)])
    prompt = (
        f"{context}\n\n"
        f"Question: {query.strip()}\n\n"
        f"Answer with a complete response followed by a short reasoning.\n"
        f"Format:\nAnswer: <your answer>\nReasoning: <your reasoning>"
    )
    return prompt

def run_all(query: str):
    """..."""
    # ---- 6-A build four contexts -----------------------------------------
    print("Retrieving contexts...")
    start_ctx = time.time()
    ctx_bm25          = [r["text"] for r in bm25.retrieve(query, k=TOP_K)]
    ctx_dense         = [r["text"] for r in dense.retrieve(query, k=TOP_K)]
    results_bm25_rr, _  = pipe_bm25_rerank.search(query)
    ctx_bm25_rerank   = [r["text"] for r in results_bm25_rr[:TOP_K]]
    results_dense_rr, _ = pipe_dense_rerank.search(query)
    ctx_dense_rerank  = [r["text"] for r in results_dense_rr[:TOP_K]]
    print(f"Context retrieval finished in {time.time() - start_ctx:.1f}s\n")

    # ---- 6-B call every LLM on every context (explicitly, no loops) -------
    print("\n" + "="*120)
    print(f"QUERY: {query}")
    print("="*120 + "\n")

    # --- Group 1: BM-25 Context ---
    print("--- Using BM-25 Context ---")
    current_context = ctx_bm25

    prompt = build_prompt(query, current_context)
    
    # Nemotron
    title = "BM25  →  Nemotron-70B"
    print(title)
    try:
        start_llm = time.time()
        result = nemotron.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Llama-3
    title = "BM25  →  Llama-3-70B"
    print(title)
    try:
        start_llm = time.time()
        result = llama3.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # falcon
    title = "BM25  →  falcon-7B"
    print(title)
    try:
        start_llm = time.time()
        result = falcon.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Deepseek
    title = "BM25  →  Deepseek-7B"
    print(title)
    try:
        start_llm = time.time()
        result = deepseek.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # --- Group 2: Dense Context ---
    print("--- Using Dense Context ---")
    current_context = ctx_dense

    # Nemotron
    title = "Dense →  Nemotron-70B"
    print(title)
    try:
        start_llm = time.time()
        result = nemotron.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Llama-3
    title = "Dense →  Llama-3-70B"
    print(title)
    try:
        start_llm = time.time()
        result = llama3.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # falcon
    title = "Dense →  falcon-7B"
    print(title)
    try:
        start_llm = time.time()
        result = falcon.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Deepseek
    title = "Dense →  Deepseek-7B"
    print(title)
    try:
        start_llm = time.time()
        result = deepseek.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")


    # --- Group 3: BM-25 + Rerank Context ---
    print("--- Using BM-25 + Rerank Context ---")
    current_context = ctx_bm25_rerank

    # Nemotron
    title = "BM25+RR → Nemotron-70B"
    print(title)
    try:
        start_llm = time.time()
        result = nemotron.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Llama-3
    title = "BM25+RR → Llama-3-70B"
    print(title)
    try:
        start_llm = time.time()
        result = llama3.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # falcon
    title = "BM25+RR → falcon-7B"
    print(title)
    try:
        start_llm = time.time()
        result = falcon.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Deepseek
    title = "BM25+RR → Deepseek-7B"
    print(title)
    try:
        start_llm = time.time()
        result = deepseek.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")


    # --- Group 4: Dense + Rerank Context ---
    print("--- Using Dense + Rerank Context ---")
    current_context = ctx_dense_rerank

    # Nemotron
    title = "Dense+RR → Nemotron-70B"
    print(title)
    try:
        start_llm = time.time()
        result = nemotron.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Llama-3
    title = "Dense+RR → Llama-3-70B"
    print(title)
    try:
        start_llm = time.time()
        result = llama3.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # falcon
    title = "Dense+RR → falcon-7B"
    print(title)
    try:
        start_llm = time.time()
        result = falcon.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

    # Deepseek
    title = "Dense+RR → Deepseek-7B"
    print(title)
    try:
        start_llm = time.time()
        result = deepseek.generate(prompt)
        print(f"  Answer:    {result.get('answer', 'ERROR: Key missing')}")
        print(f"  Reasoning: {result.get('reasoning', 'ERROR: Key missing')}")
        print(f"  (Time: {time.time() - start_llm:.1f}s)\n")
    except Exception as e:
        print(f"  ERROR: {e}\n")

# ---------------------------------------------------------------------------
#  7.  Script entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) or \
                 "What is the standard for reviewing a jury instruction?"

    t0 = time.time()
    run_all(user_query)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
