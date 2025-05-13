# LegalMind

## Overview
LegalMind is a specialized legal question-answering system that employs structured retrieval and reranking techniques to provide transparent and accurate responses to legal queries.

## Architecture
The system uses a two-stage retrieval approach:
1. **First-stage retrieval**: Quickly identifies potentially relevant documents using either:
   - BM25 (keyword-based retrieval)
   - Dense Retrieval (embedding-based semantic search)
2. **Cross-encoder reranking**: Re-scores documents for higher precision using context-aware neural models

## How to Run Experiments

### Running Rerankers
```bash
# Run both experiments
python run_experiments.py --exp3 --exp4

# Run individual experiments
python run_experiments.py --exp3  # BM25 + Reranker
python run_experiments.py --exp4  # Dense Retriever + Reranker
```

## Output Formats

### 1. BM25 First-stage Retriever
```json
[
  {
    "id": "doc_123",             // Document identifier
    "score": 12.456,             // BM25 relevance score (higher is better)
    "text": "The Fourth Amendment protects citizens against unreasonable searches..."
  },
  // Up to 100 documents by default
]
```

### 2. Dense First-stage Retriever
```json
[
  {
    "id": "doc_789",             // Document identifier
    "score": 0.912,              // Cosine similarity score (range -1 to 1)
    "text": "The Supreme Court has ruled that searches conducted without a warrant..."
  },
  // Up to 100 documents by default
]
```

### 3. Cross-Encoder Reranker (Final Output)
```json
[
  {
    "id": "doc_456",             // Document identifier
    "score": 0.967,              // Neural model relevance score (typically 0-1)
    "text": "Warrantless searches may be permitted under certain conditions..."
  },
  // Top 10 documents by default
]
```

## How to get Retriever Metrics: 
- `cd src`
- `python main_retrievers_evaluation.py`



## Performance Summary - Retrievers

| System             |   MAP    |   MRR    | NDCG@5  | NDCG@10 | NDCG@20 | Precision@5 | Precision@10 | Precision@20 | Recall@5 | Recall@10 | Recall@20 |   F1@5   |  F1@10  |  F1@20  | Retrieval Time |
|--------------------|----------|----------|---------|---------|---------|--------------|---------------|---------------|----------|-----------|-----------|----------|---------|---------|----------------|
| BM25               | 0.390435 | 0.824548 | 0.486589 | 0.490163 | 0.493176 | 0.186        | 0.095         | 0.0485        | 0.430833 | 0.438333  | 0.448333  | 0.243175 | 0.149096 | 0.085134 | 0.022867       |
| Dense Retriever    | 0.033664 | 0.075725 | 0.043566 | 0.048882 | 0.053894 | 0.018        | 0.012         | 0.0080        | 0.051500 | 0.066000  | 0.081333  | 0.024794 | 0.019130 | 0.014091 | 1.872362       |
| BM25 + Reranker    | 0.402081 | 0.836111 | 0.501395 | 0.505669 | 0.505669 | 0.190        | 0.098         | 0.0490        | 0.445000 | 0.455333  | 0.455333  | 0.249087 | 0.153914 | 0.085977 | 18.257292      |
| Dense + Reranker   | 0.162739 | 0.314583 | 0.193879 | 0.195813 | 0.196759 | 0.068        | 0.035         | 0.0180        | 0.169333 | 0.174333  | 0.176333  | 0.089333 | 0.055089 | 0.031522 | 20.038220      |
