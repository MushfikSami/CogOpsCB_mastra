import os
import sys
import time
import random
import string
import numpy as np
from transformers import AutoTokenizer
import tritonclient.http as httpclient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
TOKENIZER_PATH = 'onnx-community/embeddinggemma-300m-ONNX'
TRITON_URL = 'localhost:6000'
MODEL_NAME = 'gemma_embedding'
QUERY_PREFIX = "task: search result | query: "


# --- Load Generation Functions ---

def generate_random_text(max_length):
    """Generates random text with a length up to the specified max_length."""
    length = random.randint(max_length // 4, max_length) # Generate varied lengths
    return ''.join(random.choices(string.ascii_lowercase + ' ', k=length))

def send_inference_request(triton_client, tokenizer, batch_size, context_length):
    """Sends a single inference request to Triton."""
    # 1. Prepare the text data with random context lengths
    texts = [QUERY_PREFIX + generate_random_text(context_length) for _ in range(batch_size)]

    # 2. Tokenize the text
    tokens = tokenizer(
        texts,
        return_tensors='np',
        padding='max_length',
        truncation=True,
        max_length=context_length
    )

    # 3. Prepare Triton inputs
    input_ids = tokens['input_ids'].astype(np.int64)
    attention_mask = tokens['attention_mask'].astype(np.int64)

    inputs = [
        httpclient.InferInput('input_ids', input_ids.shape, "INT64"),
        httpclient.InferInput('attention_mask', attention_mask.shape, "INT64")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    # 4. Send the request
    try:
        triton_client.infer(model_name=MODEL_NAME, inputs=inputs)
        return "SUCCESS"
    except Exception as e:
        return f"FAILURE: {e}"

def run_load_scenario(config):
    """Runs a full load testing scenario based on the given configuration."""
    num_workers = config["workers"]
    requests_per_worker = config["requests_per_worker"]
    batch_size = config["batch_size"]
    context_length = config["context_length"]
    total_requests = num_workers * requests_per_worker

    print("\n" + "="*80)
    print(f"Starting Load Test Scenario: {config.get('name', 'Unnamed')}")
    print(f"   - Concurrent Workers: {num_workers}")
    print(f"   - Batch Size per Request: {batch_size}")
    print(f"   - Max Context Length: {context_length} tokens")
    print(f"   - Total Inference Requests: {total_requests}")
    print("="*80)

    # We create a new client and tokenizer for each scenario to ensure thread safety
    triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    success_count = 0
    start_time = time.time()

    # Use a ThreadPoolExecutor to send requests concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_inference_request, triton_client, tokenizer, batch_size, context_length)
            for _ in range(total_requests)
        ]

        # Use tqdm to create a progress bar for the completed requests
        for future in tqdm(as_completed(futures), total=total_requests, desc="Processing Requests"):
            result = future.result()
            if result == "SUCCESS":
                success_count += 1

    end_time = time.time()
    total_time = end_time - start_time

    # --- Report Results ---
    print("\n--- Scenario Results ---")
    print(f"  - Total Time Taken: {total_time:.2f} seconds")
    print(f"  - Successful Requests: {success_count} / {total_requests}")
    if total_time > 0:
        rps = total_requests / total_time
        print(f"  - Throughput (RPS): {rps:.2f} requests/second")
    print("="*80)


def main():
    """Defines and runs the load testing scenarios."""
    print("Initializing Triton Load Test...")

    # --- DEFINE YOUR TEST SCENARIOS HERE ---
    # You can add, remove, or modify these scenarios to test different loads.
    scenarios = [
        { "name": "Light Load", "workers": 1, "requests_per_worker": 25, "batch_size": 1, "context_length": 512 },
        { "name": "Medium Load - Small Batch", "workers": 1, "requests_per_worker": 25, "batch_size": 2, "context_length": 1024 },
        { "name": "Medium Load - Max Batch", "workers": 1, "requests_per_worker": 25, "batch_size": 8, "context_length": 512 },
        { "name": "High Load - Max Batch", "workers": 1, "requests_per_worker": 25, "batch_size": 8, "context_length": 1024 },
        { "name": "Max Load - Max Batch & Context", "workers": 1, "requests_per_worker": 25, "batch_size": 8, "context_length": 2048 },
    ]

    for scenario_config in scenarios:
        run_load_scenario(scenario_config)

if __name__ == "__main__":
    main()
