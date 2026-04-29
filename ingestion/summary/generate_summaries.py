"""
generate_summaries.py

Reads a CSV, summarizes a specified column per row via an LLM using concurrent requests,
appends a 'summary' column, and overwrites the CSV incrementally.

Usage:
    python generate_summaries.py [--config config.yaml]
"""

import argparse
import os
import shutil
import concurrent.futures

import openai
import pandas as pd
import yaml

# The heavy system instructions remain static to guarantee 100% prefix cache hits
SYSTEM_PROMPT = """You are an expert data indexer tasked with preparing text for semantic search and vector embeddings. Your goal is to create a highly structured, comprehensive "Index-Style Summary" of the provided text in ENGLISH. 

Do not write a traditional paragraph summary or translate the text word-for-word. Instead, map out *what information is contained in the text* by extracting the high-level structure, core concepts, and sequential steps.

### INSTRUCTIONS:
1. **Incorporate Meta-Information**: Use the provided Category, Sub-Category, Service, and Topic to establish the primary semantic context.
2. **Focus on High-Level Indexing**: Describe what the text covers comprehensively but exclude granular details (e.g., exact URLs, deep explanations, specific numeric examples). 
3. **Structure Processes and Lists**: If the text describes a process, guidelines, or required documents, index them concisely using a step-by-step or bulleted format. 
4. **Optimize for Embedding**: Use clear, descriptive nouns and action verbs. Ensure semantically rich keywords are preserved.

### REQUIRED OUTPUT FORMAT:
**Semantic Context**: [Category] > [Sub-Category] > [Service]
**Topic Overview**: [1-2 sentences stating exactly what information this document provides]
**Key Information Index**:
- ...
**Process / Steps (if applicable)**:
- Step 1: [Concise summary of step 1]
- Step 2: [Concise summary of step 2]
- Step N:[Concise summary of step N]"""


def load_config(path: str) -> dict:
    """Load YAML config, resolving ${ENV_VAR} placeholders."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    def _resolve(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj
        
    return {k: _resolve(v) for k, v in config.items()}


def build_messages(row, column):
    """Build separated System and User messages to maximize token alignment for caching."""
    val = str(row.get(column, "")).strip()
    return[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### DATA TO INDEX:\n{val}\n\nIndex-Style Summary:"}
    ]


def process(config):
    """Core processing: read CSV -> batch concurrently -> LLM -> write CSV incrementally."""
    csv_path = config.get("csv_path", "data.csv")
    column = config.get("column", "text")
    batch_size = config.get("batch_size", 10) # Used for both iteration batching and max concurrent workers
    model = config.get("model", "qwen36")
    api_key = config.get("api_key", "EMPTY")
    base_url = config.get("base_url", "http://localhost:5000/v1")

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # Create backup before modifying
    shutil.copy2(csv_path, csv_path + ".bak")
    
    # Ensure summary column exists
    if "summary" not in df.columns:
        df["summary"] = ""

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    total = len(df)

    def fetch_summary(idx, row_data):
        """Worker function for threading."""
        messages = build_messages(row_data, column)
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=messages,
                temperature=0.3, # Low temperature for consistent formatting
            )
            return idx, resp.choices[0].message.content.strip()
        except Exception as e:
            return idx, f"ERROR: {e}"

    print(f"Starting processing with batch_size/concurrency: {batch_size}")
    
    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        
        # Fire off concurrent requests using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all rows in the current batch
            futures = [executor.submit(fetch_summary, idx, batch.loc[idx]) for idx in batch.index]
            
            # As soon as a request finishes, update the dataframe safely in the main thread
            for future in concurrent.futures.as_completed(futures):
                idx_ret, result = future.result()
                if result.startswith("ERROR:"):
                    print(f"  ERROR on row {idx_ret}: {result}")
                else:
                    df.at[idx_ret, "summary"] = result

        # Save to CSV after the entire batch is completed
        df.to_csv(csv_path, index=False)
        end = min(start + batch_size, total)
        print(f"[{start+1}-{end}/{total} done]")

    print(f"\nDone! Saved to {csv_path} (backup preserved at {csv_path}.bak)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM summaries from a CSV")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    
    try:
        cfg = load_config(parser.parse_args().config)
        process(cfg)
    except FileNotFoundError:
        print("Config file not found. Please ensure 'config.yaml' exists or provide a valid path.")