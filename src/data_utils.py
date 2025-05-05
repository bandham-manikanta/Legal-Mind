# src/data_utils.py
import json
import os


def find_project_root(project_dir_name: str = 'Legal-Mind') -> str:
    """Find the project root directory by walking up until we find the target directory name"""
    current_dir = os.getcwd()
    root_dir = current_dir
    
    while not os.path.basename(root_dir) == project_dir_name:
        parent_dir = os.path.dirname(root_dir)
        if parent_dir == root_dir:  # Reached filesystem root
            raise ValueError(f"Project root '{project_dir_name}' not found. Check your directory structure.")
        root_dir = parent_dir
        
    return root_dir

def load_casehold(path="data/casehold_test_processed.jsonl",
                  text_key="contents", # Adjust if your JSONL uses different keys
                  id_key="id"):      # Adjust if your JSONL uses different keys
    """
    Reads a JSON Lines file and returns parallel lists of documents and doc_ids.
    Assumes each line is a JSON object with keys specified by text_key and id_key.
    """
    documents, doc_ids = [], []
    print(f"Attempting to load data from: {path}")
    root_dir = find_project_root()
    path = os.path.join(root_dir, path)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    if text_key not in row:
                        print(f"Warning: Line {i+1} missing key '{text_key}'. Skipping.")
                        continue
                    if id_key not in row:
                        print(f"Warning: Line {i+1} missing key '{id_key}'. Skipping.")
                        continue
                    documents.append(row[text_key])
                    doc_ids.append(row[id_key])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {i+1}. Skipping.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        print("Please ensure '../data/casehold_etst_processed.jsonl' exists relative to where you run the script.")
        # Depending on your setup, you might need a more robust path handling,
        # e.g., constructing the path relative to the script's location.
        # import os
        # script_dir = os.path.dirname(__file__) # Get directory of data_utils.py
        # parent_dir = os.path.dirname(script_dir) # Get project root (assuming src/ structure)
        # full_path = os.path.join(parent_dir, path)
        # print(f"Attempting absolute path: {full_path}")
        # # Re-try with full_path if needed...
        return [], [] # Return empty lists on failure

    print(f"Successfully loaded {len(documents)} documents.")
    return documents, doc_ids

