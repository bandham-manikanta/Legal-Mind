import json
import random
import os
import time

# --- Dependencies for the LLM client ---
from openai import OpenAI # Used by LegalLLAMA3_70B_Answerer
from config import config

# --- Configuration ---
# IMPORTANT: REPLACE "YOUR_NVIDIA_API_KEY_HERE" with your actual NVIDIA API key
NVIDIA_API_KEY = config['NVIDIA_API_KEY']

CORPUS_FILE_PATH = "/home/bandham/Documents/StonyBrook_CourseWork/Spring 2025/LLM-AMS692.02/Legal-Mind/data/casehold_test_processed.jsonl"
OUTPUT_EVAL_FILE = "/home/bandham/Documents/StonyBrook_CourseWork/Spring 2025/LLM-AMS692.02/Legal-Mind/data/llm_generated_evaluation_queries_llama3_inscript.jsonl"

NUMBER_OF_QUERIES_TO_GENERATE = 100 # Target number of queries
DOCS_PER_QUERY_GENERATION_MIN = 1  # Min number of docs to feed to LLM for one query
DOCS_PER_QUERY_GENERATION_MAX = 5  # Max number of docs to feed to LLM

# --- LLM Client Class Definition (Integrated) ---
class LegalLLAMA3QueryGenerator:
    def __init__(self, api_key, model="meta/llama3-70b-instruct"):
        """
        Initializes the Llama-3 70B client using the NVIDIA API endpoint.

        Args:
            api_key (str): Your NVIDIA API key.
            model (str): The model identifier for Llama-3 70B on NVIDIA's platform.
        """
        if not api_key or api_key == "YOUR_NVIDIA_API_KEY_HERE":
            raise ValueError("NVIDIA_API_KEY is not set. Please replace 'YOUR_NVIDIA_API_KEY_HERE' in the script.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model = model
        print(f"LegalLLAMA3_70B_Answerer initialized with model: {self.model}")

    def generate(self, prompt, debug=False, retries=3, wait_time=10):
        """
        Generates a response from Llama-3 70B based on the provided prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            debug (bool): If True, prints the raw response from the LLM.
            retries (int): Number of times to retry the API call in case of failure.
            wait_time (int): Seconds to wait between retries.

        Returns:
            dict: A dictionary containing "answer" and "reasoning".
                  For query generation, the "answer" will be the generated query.
        
        Raises:
            RuntimeError: If all retries fail.
        """
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3, # Lower temperature for more focused query generation
                    top_p=1.0,
                    max_tokens=150,  # Queries are usually not extremely long
                    stream=False
                )
                response_text = completion.choices[0].message.content.strip()

                if debug:
                    print(f"--- LLM Raw Response (Attempt {attempt+1}) ---\n{response_text}\n--------------------------------------")

                # Parse output based on the prompt's instruction for "Answer:"
                answer = ""
                # The prompt asks the LLM to prefix the query with "Answer:"
                if "Answer:" in response_text:
                    answer = response_text.split("Answer:", 1)[-1].strip()
                else:
                    # If "Answer:" is not found, take the whole response as the potential query.
                    # This might happen if the LLM doesn't follow the formatting instruction perfectly.
                    print(f"Warning: 'Answer:' prefix not found in LLM response. Using full response: '{response_text}'")
                    answer = response_text 

                # For query generation, we don't expect "Reasoning:" in the same way as Q&A.
                # The "answer" field will hold our generated query.
                return {
                    "answer": answer,
                    "reasoning": "(Reasoning not applicable for query generation task)"
                }

            except Exception as e:
                print(f"⚠️ LLM API Call Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    print(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # After all retries, return a failure indicator or raise an error
                    # For this script, let's return an empty answer to allow the loop to continue/log
                    print(f"❌ All {retries} retries failed for Llama-3 70B for this prompt.")
                    return {"answer": "", "reasoning": f"Failed after {retries} retries: {e}"}
        return {"answer": "", "reasoning": "Should not reach here if retries are handled."} # Should ideally not be reached

# --- Instantiate your LLM Client ---
try:
    llm_client = LegalLLAMA3QueryGenerator(api_key=NVIDIA_API_KEY)
except ValueError as e: # Catch the custom error from the constructor
    print(f"ERROR: {e}")
    print("Script will exit. Please set your NVIDIA_API_KEY in the script.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to initialize LegalLLAMA3_70B_Answerer: {e}")
    print("Script will exit. Please check your LLM class, API key, and dependencies.")
    exit()


def load_corpus_documents(file_path):
    """
    Loads documents from a JSONL file.
    Expects each line to be a JSON object with 'id' and 'contents' keys.
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    if 'id' in doc and 'contents' in doc:
                        documents.append(doc)
                    else:
                        print(f"Warning: Document at line {i+1} is missing 'id' or 'contents'. Skipping.")
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON at line {i+1}. Skipping: {line.strip()}")
        print(f"Successfully loaded {len(documents)} documents from '{file_path}'.")
        return documents
    except FileNotFoundError:
        print(f"Error: Corpus file not found at '{file_path}'.")
        return []
    except Exception as e:
        print(f"An error occurred while loading corpus: {e}")
        return []

def create_llm_prompt_for_query_generation(document_contents, document_ids):
    """
    Creates a structured prompt to ask the LLM to generate a query.
    """
    if not document_contents:
        return ""

    intro = "Based on the following legal document excerpt(s):"
    if len(document_ids) > 1:
        intro = f"Based on the following {len(document_ids)} related legal document excerpts (IDs: {', '.join(document_ids)}):"
    elif document_ids:
        intro = f"Based on the following legal document excerpt (ID: {document_ids[0]}):"

    formatted_contents = ""
    for i, content in enumerate(document_contents):
        formatted_contents += f"\n--- Document Excerpt {i+1} ---\n{content}\n--- End of Document Excerpt {i+1} ---\n"

    prompt = (
        f"{intro}\n"
        f"{formatted_contents}\n"
        "Please formulate a specific and clear legal question that a user might ask, "
        "for which the provided text contains a direct and substantial answer. "
        "The question should be suitable for evaluating an information retrieval system. "
        "Focus on a key legal concept, argument, or holding within the text.\n"
        "Your response should be formatted with 'Answer:' followed by the question. For example:\n"
        "Answer: What is the standard for ...?\n"
        "Provide ONLY the 'Answer:' part containing the question."
    )
    return prompt

def generate_evaluation_queries():
    """
    Main function to generate evaluation queries using an LLM.
    """
    if NVIDIA_API_KEY == "YOUR_NVIDIA_API_KEY_HERE":
        print("CRITICAL ERROR: NVIDIA_API_KEY is not set in the script.")
        print("Please replace 'YOUR_NVIDIA_API_KEY_HERE' with your actual key and rerun.")
        return

    corpus = load_corpus_documents(CORPUS_FILE_PATH)
    if not corpus:
        print("Cannot proceed without a loaded corpus.")
        return

    generated_queries_data = []
    used_doc_indices = set()

    print(f"\nAttempting to generate {NUMBER_OF_QUERIES_TO_GENERATE} queries using Llama-3...")

    for i in range(NUMBER_OF_QUERIES_TO_GENERATE):
        if len(generated_queries_data) >= NUMBER_OF_QUERIES_TO_GENERATE:
            break # Target reached

        num_docs_for_this_query = random.randint(DOCS_PER_QUERY_GENERATION_MIN, DOCS_PER_QUERY_GENERATION_MAX)
        
        if len(corpus) < num_docs_for_this_query:
            print(f"Warning: Not enough documents in corpus ({len(corpus)}) to select {num_docs_for_this_query}. Reducing.")
            num_docs_for_this_query = len(corpus)
            if num_docs_for_this_query == 0:
                print("Error: Corpus is empty. Stopping.")
                break
        
        # Simple sampling strategy
        if len(used_doc_indices) >= len(corpus): # Allow reuse if all docs have been starting points
            used_doc_indices.clear()

        start_idx = -1
        for _ in range(len(corpus)): # Try to find an unused starting index
            potential_start_idx = random.randrange(len(corpus) - num_docs_for_this_query + 1 if len(corpus) >= num_docs_for_this_query else len(corpus))
            if potential_start_idx not in used_doc_indices:
                start_idx = potential_start_idx
                break
        if start_idx == -1: # Fallback if all have been used or list is too small
            start_idx = random.randrange(len(corpus) - num_docs_for_this_query + 1 if len(corpus) >= num_docs_for_this_query else len(corpus))
        
        used_doc_indices.add(start_idx)
        
        current_selection_indices = [(start_idx + offset) % len(corpus) for offset in range(num_docs_for_this_query)]
        # Ensure unique indices if num_docs_for_this_query > 1, though contiguous selection usually handles this unless corpus is tiny
        current_selection_indices = sorted(list(set(current_selection_indices))) 
        
        selected_docs_for_prompt = [corpus[j] for j in current_selection_indices]
        doc_contents = [doc['contents'] for doc in selected_docs_for_prompt]
        doc_ids = [doc['id'] for doc in selected_docs_for_prompt]

        llm_prompt = create_llm_prompt_for_query_generation(doc_contents, doc_ids)
        if not llm_prompt:
            print("Warning: Could not generate prompt (no doc_contents). Skipping this iteration.")
            continue

        try:
            print(f"\nGenerating query {len(generated_queries_data) + 1}/{NUMBER_OF_QUERIES_TO_GENERATE} using {len(doc_ids)} document(s): {doc_ids}")
            
            response_dict = llm_client.generate(llm_prompt, debug=False) # Set debug=True for more LLM output
            generated_query_text = response_dict.get("answer", "").strip()
            
            generated_query_text = generated_query_text.replace('\n', ' ').strip()
            if generated_query_text and not generated_query_text.endswith('?'):
                 generated_query_text += '?'

            if generated_query_text and len(generated_query_text) > 10: # Basic check
                query_entry = {
                    "query_id": f"llama3_gen_q{len(generated_queries_data) + 1:03d}",
                    "query_text": generated_query_text,
                    "relevant_doc_ids": doc_ids
                }
                generated_queries_data.append(query_entry)
                print(f"-> Successfully generated query: \"{generated_query_text}\"")
            else:
                print(f"Warning: LLM generated an empty or too short query for docs {doc_ids}. Raw LLM 'answer' field: '{response_dict.get('answer')}'")

        except Exception as e:
            print(f"Error during LLM call or processing for docs {doc_ids}: {e}")

    try:
        with open(OUTPUT_EVAL_FILE, 'w', encoding='utf-8') as f_out:
            for entry in generated_queries_data:
                f_out.write(json.dumps(entry) + '\n')
        print(f"\nSuccessfully saved {len(generated_queries_data)} generated evaluation queries to '{OUTPUT_EVAL_FILE}'.")
        print("IMPORTANT: Please MANUALLY REVIEW and REFINE this file before using it for serious evaluation.")
    except Exception as e:
        print(f"Error saving generated queries to file: {e}")

if __name__ == '__main__':
    generate_evaluation_queries()
