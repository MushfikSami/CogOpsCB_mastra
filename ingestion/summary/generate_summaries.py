"""
generate_summaries.py

Reads a CSV, summarizes a specified column per row via an LLM,
appends a 'summary' column, and overwrites the CSV incrementally.

Usage:
    python generate_summaries.py [--config config.yaml]
"""

import argparse
import os
import shutil

import openai
import pandas as pd
import yaml

PROMPT_TEMPLATE = """Summarize the following data concisely in 2-5 sentences.  
**NOTE**: 
* while summerizing keep in mind that this data will be embedded by an embedder later so that any semantically similiar query can match. 
* In the text there is topic servic category and sub-category present as meta infomation . Use these as context while summerizing. 
* Language is Bnagla. So summary should be bangla as well . Bangladeshi Bangla . 
{content}

Summary:"""


def load_config(path: str) -> dict:
    """Load YAML config, resolving ${ENV_VAR} placeholders."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    def _resolve(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj
    return {k: _resolve(v) for k, v in config.items()}


def build_prompt(row, column):
    """Build the LLM prompt from the target column value."""
    val = str(row.get(column, "")).strip()
    return PROMPT_TEMPLATE.format(content=val)


def process(config):
    """Core processing: read CSV -> batch -> LLM -> write CSV incrementally."""
    csv_path = config["csv_path"]
    column = config["column"]
    batch_size = config["batch_size"]
    model = config["model"]
    api_key = config["api_key"]
    base_url = config["base_url"]

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    shutil.copy2(csv_path, csv_path + ".bak")
    df["summary"] = ""

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    total = len(df)

    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        for idx in batch.index:
            prompt = build_prompt(df.loc[idx], column)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                df.at[idx, "summary"] = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  ERROR on row {idx}: {e}")

        # Save after each batch
        df.to_csv(csv_path, index=False)
        end = min(start + batch_size, total)
        print(f"  [{start+1}-{end}/{total} done]")

    print(f"Done. Saved to {csv_path} (backup at {csv_path}.bak)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM summaries from a CSV")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    config = load_config(parser.parse_args().config)
    process(config)
