import json
import argparse
from types import SimpleNamespace
import os

def get_config():
    parser = argparse.ArgumentParser(description="Run legal document retrieval evaluation.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation and only visualize latest results.')
    parser.add_argument('--num-queries', type=int, help='Override number of queries to evaluate from config.')
    # Add any other CLI overrides you might need

    args = parser.parse_args()

    # Ensure the config path is correct, potentially relative to the script or project root
    config_path = args.config
    if not os.path.isabs(config_path):
         # Assuming main.py is at project root, or adjust as needed
        config_path = os.path.join(os.path.dirname(__file__), '..', args.config) # If config_loader is in a subdir
        config_path = os.path.abspath(config_path) # Make it absolute
        if not os.path.exists(config_path) and args.config == 'config.json': # Fallback to current dir if not found relative to main
             config_path = args.config


    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Flatten the dictionary for easier access if using SimpleNamespace
    flat_config = {}
    for key in config_dict:
        flat_config.update(config_dict[key])

    cfg = SimpleNamespace(**flat_config)

    # Override with command-line arguments if provided
    if args.num_queries is not None:
        cfg.num_queries = args.num_queries

    cfg.skip_evaluation = args.skip_evaluation

    # You might want to make paths in config absolute here
    # For example, if ROOT_DIR is defined by find_project_root() from data_utils
    # from data_utils import find_project_root (handle potential circular import or define find_project_root here/elsewhere accessible)
    # ROOT_DIR = find_project_root()
    # cfg.data_path = os.path.join(ROOT_DIR, cfg.data_path) if not os.path.isabs(cfg.data_path) else cfg.data_path
    # cfg.bm25_index_dir = os.path.join(ROOT_DIR, cfg.bm25_index_dir) # etc.

    return cfg