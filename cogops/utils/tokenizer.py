"""
cogops/utils/tokenizer.py

Tokenizer wrapper: load tokenizer + count tokens.
Pulled from cogops/utils/token_manager.py.
"""

import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Tokenizer:
    """Simple tokenizer wrapper for counting tokens."""

    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("Tokenizer requires an explicit model_name.")
        logger.info(f"Loading tokenizer from '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded.")

    def count(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
