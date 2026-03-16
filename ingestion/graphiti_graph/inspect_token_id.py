"""
Script to find the correct token IDs for 'True' and 'False' in the current Qwen3.5 tokenizer.
"""
import sys
import os

sys.path.insert(0, '/home/vpa/CogOpsCB')

from transformers import AutoTokenizer


def find_token_ids():
    """Find token IDs for True/False and variations with leading spaces."""
    # Try different Qwen3 model names - user mentioned they use "qwen3.5-35b"
    # Common variations to try:
    model_names = [
        # User's model based on "cyankiwi--Qwen3.5-35B-A3B-AWQ-4bit"
        "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
        "Qwen/Qwen3-30B-A3B-Instruct",
        "Qwen/Qwen3-30B-Instruct",
        "Qwen/Qwen3-8B-Instruct",
        # Fallback to 2.5
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    tokenizer = None
    for model_name in model_names:
        try:
            print(f"Trying: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"Success: {model_name}")
            break
        except Exception as e:
            print(f"Failed: {model_name} - {type(e).__name__}: {e}")
            continue

    if tokenizer is None:
        raise RuntimeError("No Qwen model could be loaded from the available options.")

    print("Token IDs for True/False and variations:")
    print("=" * 60)

    test_strings = ["True", " True", "False", " False", "true", " true", "false", " false"]

    for text in test_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f'"{text}": token_ids = {token_ids}')

    print("\n" + "=" * 60)
    print("Recommended configuration for qwen3_reranker.py:")
    print("-" * 60)

    # Find the token IDs for each variation
    true_ids = tokenizer.encode("True", add_special_tokens=False)
    true_with_space = tokenizer.encode(" True", add_special_tokens=False)
    false_ids = tokenizer.encode("False", add_special_tokens=False)
    false_with_space = tokenizer.encode(" False", add_special_tokens=False)

    print(f"QWEN_TRUE_IDS = {true_ids + true_with_space}")
    print(f"QWEN_FALSE_IDS = {false_ids + false_with_space}")

    # Print as they should appear in the file (with quotes for compatibility)
    all_ids = true_ids + true_with_space + false_ids + false_with_space
    print(f"\nQWEN_LOGIT_BIAS = {{tid: 100 for tid in {true_ids + true_with_space + false_ids + false_with_space}}}")


if __name__ == "__main__":
    find_token_ids()
