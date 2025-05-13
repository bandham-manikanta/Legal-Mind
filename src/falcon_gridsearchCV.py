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
    words = len(re.findall(r'\b\w+\b', string))
    punct = len(re.findall(r'[^\w\s]', string))
    safety_factor = 1.3
    return int((words + punct) * safety_factor)

class LegalFalcon3_7B_Answerer:
    def __init__(self, api_key, model="tiiuae/falcon3-7b-instruct"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model = model
        self.max_context_length = 6500  # Conservative limit

    def build_prompt(self, query, context_docs):
        """
        Build a prompt for the Falcon model with context management.
        """
        # Start with query and instruction
        question_part = f"Question: {query.strip()}"
        instruction_part = f"Answer with a complete response followed by a short reasoning."
        
        # Calculate tokens used by these parts
        base_tokens = estimate_tokens(question_part + instruction_part)
        
        # Determine available tokens for context
        available_context_tokens = self.max_context_length - base_tokens
        
        # Build context with token tracking
        context_parts = []
        tokens_used = 0
        
        for i, doc in enumerate(context_docs):
            doc_text = f"Context {i+1}:\n{doc.strip()}"
            doc_tokens = estimate_tokens(doc_text)
            
            # Skip if adding this would exceed the limit
            if tokens_used + doc_tokens > available_context_tokens:
                print(f"⚠️ Skipping context document {i+1} due to token limit")
                continue
                
            context_parts.append(doc_text)
            tokens_used += doc_tokens
        
        # Join contexts that fit
        context = "\n\n".join(context_parts)
        
        # Assemble final prompt
        prompt = f"{context}\n\n{question_part}\n\n{instruction_part}"
        
        # Log estimated token count
        final_tokens = estimate_tokens(prompt)
        print(f"ℹ️ Prompt estimated at ~{final_tokens} tokens (limit: {self.max_context_length})")
        
        return prompt

    def generate(self, prompt, temperature=0.3, top_p=1.0, best_of_n=1, debug=False, retries=3, wait_time=10):
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
                    break
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed: {e}")
                    if attempt < retries - 1:
                        print(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ All retries failed for Falcon3-7B: {e}")
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

def hyperparameter_grid_search(answerer: LegalFalcon3_7B_Answerer,
                                prompts: List[str],
                                references: List[str],
                                temperatures: List[float],
                                top_ps: List[float],
                                best_of_n: int = 3):
    all_results = []
    grid = list(itertools.product(temperatures, top_ps))

    for temp, top_p in tqdm(grid, desc="Grid Search"):
        try:
            batch_predictions = []
            for prompt in prompts:
                candidates = answerer.generate(
                    prompt,
                    temperature=temp,
                    top_p=top_p,
                    best_of_n=best_of_n
                )
                best_answer = simple_reranker(candidates)
                batch_predictions.append(best_answer)
            
            scores = evaluate_outputs(batch_predictions, references)
            result = {
                "temperature": temp,
                "top_p": top_p,
                "BLEU": scores["BLEU"],
                "BERTScore": scores["BERTScore"]
            }
            print(f"Config Tested: {result}")
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error in grid search for temp={temp}, top_p={top_p}: {e}")
            result = {
                "temperature": temp,
                "top_p": top_p,
                "BLEU": 0.0,
                "BERTScore": 0.0
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
    answerer = LegalFalcon3_7B_Answerer(api_key=API_KEY)
    
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
    
    # Process each batch of questions
    # Note: For Falcon implementation, we'll do one hyperparameter grid search per question
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
        
        # Build the prompt for this question
        prompt = answerer.build_prompt(query, context_docs)
        
        # Run grid search for this question
        results = hyperparameter_grid_search(
            answerer=answerer,
            prompts=[prompt],  # List with single prompt for this question
            references=[reference],  # List with single reference for this question
            temperatures=temperatures,
            top_ps=top_ps,
            best_of_n=3
        )
        
        # Add question ID to each result
        for result in results:
            result['query_id'] = query_id
        
        # Save results for this question
        query_csv = f"{query_id}_falcon3_7b_hyperparameter.csv"
        query_heatmap = f"{query_id}_falcon3_7b_hyperparameter_heatmap.png"
        log_results_and_plot(results, query_csv, query_heatmap)
        
        # If this is the first question, also save a copy of the results for future reference
        if idx == 0:
            first_results = results
    
    # Create a summary of results by averaging across questions
    if len(df) > 0:
        # Create a DataFrame for summary
        summary_df = pd.DataFrame(columns=['temperature', 'top_p', 'avg_BLEU', 'avg_BERTScore'])
        
        # For each temperature and top_p combination
        for temp in temperatures:
            for top_p in top_ps:
                # For each question, get the result for this parameter combination
                bleu_scores = []
                bert_scores = []
                
                for idx, row in df.iterrows():
                    query_id = row['query_id']
                    results_file = f"{query_id}_falcon3_7b_hyperparameter.csv"
                    
                    try:
                        query_results = pd.read_csv(results_file)
                        param_result = query_results[(query_results['temperature'] == temp) & 
                                                     (query_results['top_p'] == top_p)]
                        
                        if not param_result.empty:
                            bleu_scores.append(param_result['BLEU'].values[0])
                            bert_scores.append(param_result['BERTScore'].values[0])
                    except Exception as e:
                        print(f"Could not load results for {query_id}: {e}")
                
                # Calculate averages if we have scores
                if bleu_scores and bert_scores:
                    avg_bleu = sum(bleu_scores) / len(bleu_scores)
                    avg_bert = sum(bert_scores) / len(bert_scores)
                    
                    # Add to summary
                    summary_df = pd.concat([summary_df, pd.DataFrame({
                        'temperature': [temp],
                        'top_p': [top_p],
                        'avg_BLEU': [avg_bleu],
                        'avg_BERTScore': [avg_bert]
                    })], ignore_index=True)
        
        # Save summary
        if not summary_df.empty:
            summary_df.to_csv("falcon3_7b_parameter_summary.csv", index=False)
            
            # Create summary heatmaps
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            pivot_bleu = summary_df.pivot(index="temperature", columns="top_p", values="avg_BLEU")
            sns.heatmap(pivot_bleu, annot=True, fmt=".5f", cmap="YlGnBu")
            plt.title("Average BLEU Score Across All Questions")
            
            plt.subplot(1, 2, 2)
            pivot_bert = summary_df.pivot(index="temperature", columns="top_p", values="avg_BERTScore")
            sns.heatmap(pivot_bert, annot=True, fmt=".5f", cmap="YlOrRd")
            plt.title("Average BERTScore Across All Questions")
            
            plt.tight_layout()
            plt.savefig("falcon3_7b_summary_heatmap.png", dpi=300)
            plt.close()
            
            print("\nSummary of results saved to falcon3_7b_parameter_summary.csv")
            print("Summary heatmap saved to falcon3_7b_summary_heatmap.png")
            
            # Print the best parameters
            best_bleu = summary_df.loc[summary_df['avg_BLEU'].idxmax()]
            best_bert = summary_df.loc[summary_df['avg_BERTScore'].idxmax()]
            
            print(f"\nBest parameters by BLEU: temp={best_bleu['temperature']}, top_p={best_bleu['top_p']} (score={best_bleu['avg_BLEU']:.4f})")
            print(f"Best parameters by BERTScore: temp={best_bert['temperature']}, top_p={best_bert['top_p']} (score={best_bert['avg_BERTScore']:.4f})")
