"""
cogops/config/loader.py

Configuration loading: load_config(), EndpointConfig.
Replaces the in-module copies from cogops/models/llm.py and
cogops/tools/graphiti_tools.py.
"""

import os
import yaml
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yml") -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def _load_endpoint_config(config: dict, section_name: str) -> 'EndpointConfig':
    """Load an EndpointConfig from a config dict section."""
    sec = config.get(section_name, {})
    return EndpointConfig(
        api_key_env=sec.get('api_key_env', ''),
        model_name_env=sec.get('model_name_env', ''),
        base_url_env=sec.get('base_url_env', ''),
        max_context_tokens=sec.get('max_context_tokens', 32000),
        thinking=sec.get('thinking', False),
    )


class EndpointConfig:
    """Configuration for a single vLLM endpoint."""
    def __init__(self, api_key_env: str, model_name_env: str, base_url_env: str,
                 max_context_tokens: int, thinking: bool = False):
        self.api_key = os.getenv(api_key_env, "")
        self.model = os.getenv(model_name_env, "")
        self.base_url = os.getenv(base_url_env, "")
        self.max_context_tokens = max_context_tokens
        self.thinking = thinking
