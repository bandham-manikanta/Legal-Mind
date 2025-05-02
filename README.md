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

## Performance Metrics
Each query processing also returns timing information:
```json
{
  "retrieve_time": 0.125,        // Time for first-stage retrieval (seconds)
  "rerank_time": 1.345,          // Time for reranking (seconds)
  "total_time": 1.470            // Total processing time (seconds)
}
```