import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import tritonclient.http as httpclient
import sys
import os

# --- Configuration ---
ONNX_MODEL_PATH =  os.path.expanduser('~/gemma_repo/onnx/model.onnx')
TOKENIZER_PATH = 'onnx-community/embeddinggemma-300m-ONNX'
TRITON_URL = 'localhost:6000'
MODEL_NAME = 'gemma_embedding'

PREFIXES = {
    'query': "task: search result | query: ",
    'passage': "title: none | text: ",
}

def cosine_similarity(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    sim = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return sim

def validate_embeddings(name, onnx_embedding, triton_embedding):
    print(f"\n----- Validating: {name} -----")
    similarities = cosine_similarity(onnx_embedding, triton_embedding)
    min_similarity = np.min(similarities)
    is_similar = min_similarity > 0.999
    status = "SUCCESS" if is_similar else "FAILURE"
    print(f"{status}: Minimum Cosine Similarity is {min_similarity:.6f}.")
    if is_similar:
        print("   - The ONNX model in Triton is functionally identical to the local ONNX model.")
    else:
        print(f"   - WARNING: Similarity is below the 0.999 threshold. A review is needed.")
    return is_similar

def get_onnx_embedding(session, tokens):
    inputs = { 'input_ids': tokens['input_ids'].astype(np.int64), 'attention_mask': tokens['attention_mask'].astype(np.int64) }
    return session.run(None, inputs)[1]  # sentence_embedding

def get_triton_embedding(client, tokens):
    inputs = [ httpclient.InferInput('input_ids', tokens['input_ids'].shape, "INT64"), httpclient.InferInput('attention_mask', tokens['attention_mask'].shape, "INT64") ]
    inputs[0].set_data_from_numpy(tokens['input_ids'])
    inputs[1].set_data_from_numpy(tokens['attention_mask'])
    response = client.infer(model_name=MODEL_NAME, inputs=inputs)
    return response.as_numpy('sentence_embedding')

def main():
    print("Initializing models and client for FP32 Production Validation...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider'])
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        assert triton_client.is_server_live(), "Triton server is not live."
    except Exception as e:
        print(f"ERROR: Could not initialize. {e}")
        sys.exit(1)

    all_passed = True
    try:
        passage_text = [PREFIXES['passage'] + "EmbeddingGemma is a powerful embedding model."]
        passage_tokens = tokenizer(passage_text, return_tensors='np', padding=True)
        onnx_passage_emb = get_onnx_embedding(ort_session, passage_tokens)
        triton_passage_emb = get_triton_embedding(triton_client, passage_tokens)
        if not validate_embeddings("Passage (Batch=1, FP32)", onnx_passage_emb, triton_passage_emb):
            all_passed = False

        query1_text = [PREFIXES['query'] + "what is the best embedding model?"]
        query1_tokens = tokenizer(query1_text, return_tensors='np', padding=True)
        onnx_q1_emb = get_onnx_embedding(ort_session, query1_tokens)
        triton_q1_emb = get_triton_embedding(triton_client, query1_tokens)
        if not validate_embeddings("Query (Batch=1, FP32)", onnx_q1_emb, triton_q1_emb):
            all_passed = False

        query8_text = [PREFIXES['query'] + "what is the best embedding model?"] * 8
        query8_tokens = tokenizer(query8_text, return_tensors='np', padding=True)
        onnx_q8_emb = get_onnx_embedding(ort_session, query8_tokens)
        triton_q8_emb = get_triton_embedding(triton_client, query8_tokens)
        if not validate_embeddings("Query (Batch=8, FP32)", onnx_q8_emb, triton_q8_emb):
            all_passed = False

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        all_passed = False

    print("\n\n--- FINAL PRODUCTION VALIDATION ---")
    if all_passed:
        print("ALL FP32 MODELS ARE VALIDATED FOR PRODUCTION USE!")
    else:
        print("One or more models failed the functional similarity test.")


if __name__ == "__main__":
    main()
