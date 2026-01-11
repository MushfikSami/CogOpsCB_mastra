# /config.py
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

def load_yaml_config(file_path="config.yml"):
    """Loads the YAML configuration file from a given path."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # This will provide a clear, immediate error if the config is missing.
        raise FileNotFoundError(
            f"FATAL ERROR: The configuration file was not found at '{os.path.abspath(file_path)}'. "
            "Please ensure 'config.yml' exists in the project root."
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config.yml: {e}")

# --- Load the entire configuration once ---
_config = load_yaml_config()

# --- Expose specific sections for easy import by other modules ---
CRAWLER_CONFIG = _config.get('crawler', {})
SEARCH_CONFIG = _config.get('search', {})
API_CONFIG = _config.get('api', {})

# --- Expose environment variables ---
SEARXNG_API_URL = os.getenv("SEARXNG_API_URL")
SEARXNG_API_KEY = os.getenv("SEARXNG_API_KEY")

# --- Perform validation checks on startup ---
if not SEARXNG_API_URL or not SEARXNG_API_KEY:
    raise ValueError("SEARXNG_API_URL and SEARXNG_API_KEY must be set in your .env file.")

if 'filter_domain' not in API_CONFIG:
     raise ValueError("The 'filter_domain' key is missing in the 'api' section of your config.yml.")