# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
from openai import OpenAI
from typing import List, Dict
import itertools
import time
from tqdm import tqdm
import evaluate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re

# Initialize BLEU and BERTScore metrics
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

# Simple token counter (approximation)
def estimate_tokens(string: str) -> int:
    """
    Roughly estimates the number of tokens in a string.
    This is an approximation - actual tokenizers will count differently.
    """
    # Count words
    words = len(re.findall(r'\b\w+\b', string))
    # Count punctuation and special characters
    punct = len(re.findall(r'[^\w\s]', string))
    # Space for safety margin (LLaMA tokenizer might count differently)
    safety_factor = 1.3
    return int((words + punct) * safety_factor)

class LegalLLaMA3_70B_Answerer:
    def __init__(self, api_key, model="meta/llama3-70b-instruct"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model = model
        self.max_context_length = 6500  # Leave plenty of room for safety with our estimator

    def build_prompt(self, query, context_docs):
        # Start with query and instruction which we'll always keep
        question_part = f"Question: {query.strip()}\n\n"
        instruction_part = f"Answer with a complete response followed by a short reasoning.\nFormat:\nAnswer: <your answer>\nReasoning: <your reasoning>"
        
        # Calculate tokens used by these parts
        base_tokens = estimate_tokens(question_part + instruction_part)
        
        # Determine how many tokens we can use for context
        available_context_tokens = self.max_context_length - base_tokens
        
        # Build context with token tracking
        context_parts = []
        tokens_used = 0
        
        for i, doc in enumerate(context_docs):
            doc_text = f"Context {i+1}:\n{doc.strip()}"
            doc_tokens = estimate_tokens(doc_text)
            
            # If adding this document would exceed token limit, skip it
            if tokens_used + doc_tokens > available_context_tokens:
                print(f"⚠️ Skipping context document {i+1} due to token limit")
                continue
                
            context_parts.append(doc_text)
            tokens_used += doc_tokens
        
        # Join contexts that fit
        context = "\n\n".join(context_parts)
        
        # Assemble final prompt
        prompt = f"{context}\n\n{question_part}{instruction_part}"
        
        # Show estimated token count
        final_tokens = estimate_tokens(prompt)
        print(f"ℹ️ Prompt estimated at ~{final_tokens} tokens (limit: {self.max_context_length})")
        
        return prompt

    def generate(self, query, context_docs, temperature=0.3, top_p=1.0, best_of_n=1, debug=False, retries=3, wait_time=10):
        prompt = self.build_prompt(query, context_docs)
        messages = [{"role": "user", "content": prompt}]
        candidates = []

        for _ in range(best_of_n):
            for attempt in range(retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=1024,
                        stream=False
                    )
                    response_text = completion.choices[0].message.content.strip()

                    if debug:
                        print("🧠 Raw Response:\n", response_text)

                    candidates.append(response_text)
                    break  # Success
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed: {e}")
                    if attempt < retries - 1:
                        print(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ All retries failed for LLaMA3-70B: {e}")
                        # Instead of raising an exception, return an error message
                        candidates.append(f"Error occurred: {e}")

        return candidates

def simple_reranker(candidates: List[str]) -> str:
    if not candidates:
        return "No valid responses generated"
    return max(candidates, key=lambda x: len(x))

def evaluate_outputs(predictions: List[str], references: List[str]) -> Dict:
    # Check for error messages in predictions
    if any("Error occurred" in pred for pred in predictions):
        print("⚠️ Error message detected in predictions, returning zero scores")
        return {"BLEU": 0.0, "BERTScore": 0.0}
    
    try:
        bleu = bleu_metric.compute(predictions=predictions,
                               references=[[ref] for ref in references])['bleu']
        bert = bertscore_metric.compute(predictions=predictions, references=references, lang='en')['f1']
        bert_avg = sum(bert) / len(bert)
        return {"BLEU": bleu, "BERTScore": bert_avg}
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return {"BLEU": 0.0, "BERTScore": 0.0}

def hyperparameter_grid_search(answerer: LegalLLaMA3_70B_Answerer,
                                query: str,
                                context_docs: List[str],
                                reference: str,
                                temperatures: List[float],
                                top_ps: List[float],
                                best_of_n: int = 3):
    all_results = []
    grid = list(itertools.product(temperatures, top_ps))

    for temp, top_p in tqdm(grid, desc="Grid Search"):
        try:
            candidates = answerer.generate(
                query, context_docs,
                temperature=temp,
                top_p=top_p,
                best_of_n=best_of_n
            )
            best_answer = simple_reranker(candidates)
            predictions = [best_answer]
            references = [reference]
            scores = evaluate_outputs(predictions, references)
            result = {
                "temperature": temp,
                "top_p": top_p,
                "BLEU": scores["BLEU"],
                "BERTScore": scores["BERTScore"],
                "Best Answer": best_answer
            }
            print(f"Config Tested: {result}")
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error in grid search for temp={temp}, top_p={top_p}: {e}")
            result = {
                "temperature": temp,
                "top_p": top_p,
                "BLEU": 0.0,
                "BERTScore": 0.0,
                "Best Answer": f"Error occurred: {e}"
            }
            all_results.append(result)

    return all_results

def log_results_and_plot(results: List[Dict], csv_path: str, heatmap_path: str):
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        pivot_bleu = df.pivot(index="temperature", columns="top_p", values="BLEU")
        sns.heatmap(pivot_bleu, annot=True, fmt=".5f", cmap="YlGnBu")
        plt.title("BLEU Score Heatmap")

        plt.subplot(1, 2, 2)
        pivot_bert = df.pivot(index="temperature", columns="top_p", values="BERTScore")
        sns.heatmap(pivot_bert, annot=True, fmt=".5f", cmap="YlOrRd")
        plt.title("BERTScore Heatmap")

        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
    
    print(f"\nResults saved to {csv_path} and heatmap to {heatmap_path}")

# Improved context processing with length limits
def process_context_docs(combined_text, max_chars_per_doc=3000):
    # Split the combined text into separate documents
    # Based on the sample, documents appear to be separated by '&'
    docs = combined_text.split('&')
    
    # Clean up and limit each document
    cleaned_docs = []
    for doc in docs:
        if doc.strip():
            # Truncate long documents
            if len(doc) > max_chars_per_doc:
                print(f"⚠️ Truncating document from {len(doc)} to {max_chars_per_doc} characters")
                doc = doc[:max_chars_per_doc] + "..."
            cleaned_docs.append(doc.strip())
    
    # If no documents after splitting, use the original text (truncated)
    if not cleaned_docs and combined_text.strip():
        if len(combined_text) > max_chars_per_doc:
            combined_text = combined_text[:max_chars_per_doc] + "..."
        cleaned_docs = [combined_text.strip()]
    
    return cleaned_docs

# Function to load JSONL data
def load_data(jsonl_path):
    try:
        # Try using pandas to read the jsonl file
        df = pd.read_json(jsonl_path, lines=True)
        return df
    except:
        # If that fails, load manually
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    print(f"Error parsing line: {line[:50]}...")
        return pd.DataFrame(data)

# =========================
# Example Usage with Dataset Integration
# =========================
if __name__ == "__main__":
    # Set your API key
    API_KEY = "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"
    
    # Initialize the answerer
    answerer = LegalLLaMA3_70B_Answerer(api_key=API_KEY)
    
    # Load the dataset
    print("Loading data from data.jsonl...")
    df = load_data('/home/bandham/Documents/StonyBrook_CourseWork/Spring 2025/LLM-AMS692.02/Legal-Mind/src/evaluation_results/retrieved_documents_per_experiment/Dense_and_Reranker_retrieved_docs_with_retrieved_docs_answers.jsonl')
    print(f"Loaded {len(df)} rows from data.jsonl")
    
    # Hyperparameter configurations
    temperatures = [0.1, 0.3, 0.5, 0.7]
    top_ps = [0.5, 0.7, 0.9]
    
    # Process a limited number of questions (for testing)
    # Set to None to process all questions
    num_questions = 10  # Adjust as needed
    
    if num_questions is not None:
        df = df.head(num_questions)
    
    # Process each question
    all_results = []
    
    for idx, row in df.iterrows():
        query_id = row['query_id']
        print(f"\nProcessing question {idx+1}/{len(df)}: {query_id}")
        
        # Extract query and reference answer
        query = row['query_text']
        reference = row['answer']
        
        # Process context documents with length limits
        context_docs = process_context_docs(row['COMBINED_RETRIEVED_DOCS'], max_chars_per_doc=3000)
        print(f"Query: {query}")
        print(f"Number of context documents: {len(context_docs)}")
        
        # Run grid search for this question
        results = hyperparameter_grid_search(
            answerer=answerer,
            query=query,
            context_docs=context_docs,
            reference=reference,
            temperatures=temperatures,
            top_ps=top_ps,
            best_of_n=3
        )
        
        # Add question ID to each result
        for result in results:
            result['query_id'] = query_id
        
        # Save results for this question
        query_csv = f"{query_id}_llama3_70b_hyperparameter.csv"
        query_heatmap = f"{query_id}_llama3_70b_hyperparameter_heatmap.png"
        log_results_and_plot(results, query_csv, query_heatmap)
        
        # Add to overall results
        all_results.extend(results)
    
    # Save combined results from all questions
    if all_results:
        # Combined results
        all_csv = "all_questions_llama3_70b_hyperparameter.csv"
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(all_csv, index=False)
        print(f"\nCombined results for all questions saved to {all_csv}")
        
        # Average performance by parameter settings
        print("\nAverage performance across all questions:")
        avg_by_params = all_results_df.groupby(['temperature', 'top_p'])[['BLEU', 'BERTScore']].mean().reset_index()
        print(avg_by_params.sort_values(by='BERTScore', ascending=False))
        
        # Create heatmap for average results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        pivot_bleu = avg_by_params.pivot(index="temperature", columns="top_p", values="BLEU")
        sns.heatmap(pivot_bleu, annot=True, fmt=".5f", cmap="YlGnBu")
        plt.title("Average BLEU Score Across All Questions")
        
        plt.subplot(1, 2, 2)
        pivot_bert = avg_by_params.pivot(index="temperature", columns="top_p", values="BERTScore")
        sns.heatmap(pivot_bert, annot=True, fmt=".5f", cmap="YlOrRd")
        plt.title("Average BERTScore Across All Questions")
        
        plt.tight_layout()
        plt.savefig("average_llama3_70b_hyperparameter_heatmap.png", dpi=300)
        plt.close()
        print("Average performance heatmap saved to average_llama3_70b_hyperparameter_heatmap.png")
