# Triton-EmbeddingGemma-ONNX

## Installation

This section guides you through setting up the Triton Inference Server with EmbeddingGemma-300m ONNX model (FP16 variant with external data in .onnx_data), using the ONNX Runtime backend for inference on NVIDIA GPUs. This uses the original FP16 ONNX model directly. ONNX Runtime leverages CUDA for acceleration. The maximum context length is 2048 tokens. The model handles both queries and passages/documents via text prefixes prepended to the input.

Regarding GPU memory: ONNX Runtime with CUDA will allocate GPU memory dynamically based on batch size and sequence length. To keep memory usage more consistent, use a fixed max_batch_size (e.g., 8), pad inputs to consistent lengths in your client code, and limit instance groups to 1. However, exact memory usage may vary slightly due to model internals and Triton overhead. Monitor with `nvidia-smi` during testing. If consistency is critical, consider profiling with fixed shapes via ONNX optimization (not covered here).

---

### Step 1: System Prerequisites

Ensure your host machine is equipped with the necessary software to interact with NVIDIA GPUs inside Docker containers.

#### 1.1 Install Docker

Install Docker to manage containers for running the Triton Inference Server.

```bash
sudo apt-get update
sudo apt-get install -y docker.io

# Add your user to the docker group to run Docker without sudo
sudo usermod -aG docker ${USER}

echo "----------------------------------------------------"
echo "-> You will need to log out and log back in for docker group changes to apply."
echo "-> Or, you can start a new shell with: newgrp docker"
echo "----------------------------------------------------"
```

**Note:** Log out and back in, or run `newgrp docker` to apply the group changes.

#### 1.2 Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to leverage NVIDIA GPUs.

```bash
# Add the NVIDIA repository and key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### Step 2: NVIDIA NGC Authentication

Authenticate with the NVIDIA NGC catalog to access the Triton Server container image.

#### 2.1 Get Your NGC API Key

1. Visit **[https://ngc.nvidia.com](https://ngc.nvidia.com)**.
2. Sign in or create a free account.
3. Click your username in the top-right corner and select **Setup**.
4. Click **Get API Key** and then **Generate API Key**.
5. Copy the long alphanumeric API key and store it securely. **Note:** You cannot retrieve it again after closing the page.

#### 2.2 Log in via Docker

Use the API key to authenticate with the NVIDIA Container Registry (`nvcr.io`).

```bash
docker login nvcr.io
```

- **Username:** Enter `$oauthtoken` (literal string).
- **Password:** Paste your NGC API Key (it will not be visible when typed/pasted).

Upon success, you will see a `Login Succeeded` message.

---

### Step 3: EmbeddingGemma-300m Model

* Create directories for the model data.

```bash
# Create directories on the host machine
mkdir ~/gemma_repo
```

* Download the EmbeddingGemma-300m model weights from Hugging Face.

```bash
# Install the Hugging Face command-line tool
pip install huggingface_hub

# Log in with your Hugging Face token (it will prompt you to paste it)
huggingface-cli login

# Download the model files into the data directory
huggingface-cli download onnx-community/embeddinggemma-300m-ONNX \
  --repo-type model \
  --include "onnx/model.onnx*" \
  --local-dir ~/gemma_repo
```

---

### Step 4: Prepare Model for Triton

No conversion needed; use the original FP16 ONNX model with its external data file (.onnx_data).

---

### Step 5: Inference With Triton

```bash
# Create the main model repository directory
mkdir -p ~/triton_repo/gemma_embedding/1

# Copy the ONNX model and its external data file
cp ~/gemma_repo/onnx/model.onnx ~/triton_repo/gemma_embedding/1/model.onnx
cp ~/gemma_repo/onnx/model.onnx_data ~/triton_repo/gemma_embedding/1/model.onnx_data
```

* Create the config file manually (copy the contents provided below into the file). This uses the ONNX Runtime backend with CUDA execution provider.

```bash
cat << EOF > ~/triton_repo/gemma_embedding/config.pbtxt
name: "gemma_embedding"
backend: "onnxruntime"
max_batch_size: 8

input [
  { name: "input_ids", data_type: TYPE_INT64, dims: [ -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1 ] }
]
output [
  { name: "last_hidden_state", data_type: TYPE_FP32, dims: [ -1, 768 ] },
  { name: "sentence_embedding", data_type: TYPE_FP32, dims: [ 768 ] }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}

parameters {
  key: "onnx_opset"
  value {
    string_value: "21"
  }
}
EOF
```

* After this, the triton_repo should have the following structure:

```bash
triton_repo/
└── gemma_embedding/
    ├── 1/
    │   ├── model.onnx
    │   └── model.onnx_data
    └── config.pbtxt
```

* Pull the Triton server

```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-py3
```

* Run the Triton server

```bash
docker run --gpus all --rm -d --name embgemmatriton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-py3 \
  tritonserver --model-repository=/models
```

Its actually better use the  ```docker-compose.yml``` to avoid network conflicts, autorestarts and health checks.

```yaml
# ~/triton-server/docker-compose.yml
# A robust, conflict-free setup for the Triton Inference Server.

services:
  triton:
    # --- Base Configuration (from your 'docker run' command) ---
    image: nvcr.io/nvidia/tritonserver:25.05-py3
    container_name: embgemmatriton
    # This command is passed to the container on startup
    command: tritonserver --model-repository=/models
    volumes:
      # Maps your local model repository into the container
      - ~/triton_repo:/models:ro # ':ro' makes the volume read-only for added safety
    ports:
      - "6000:8000" # HTTP
      - "6001:8001" # gRPC
      - "6002:8002" # Metrics
      # We bind to 127.0.0.1 (localhost) for security, preventing network exposure.
      # If you need to access it from other machines, change to "6000:8000", etc.

    # --- GPU Access ---
    # This is the Docker Compose equivalent of '--gpus all'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # --- NEW: Stability and Health Monitoring ---
    # Automatic Restart Policy: Restarts the container unless you manually stop it.
    restart: unless-stopped

    # Health Check: Docker will periodically check if Triton is responsive.
    healthcheck:
      # Queries the standard Triton health endpoint.
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 15s      # Check every 15 seconds
      timeout: 10s       # Wait up to 10 seconds for a response
      retries: 5         # Try 5 times before marking as unhealthy
      start_period: 60s  # Give the server 60 seconds to start up before the first check

    # --- NEW: Conflict-Free Networking ---
    # Connects this service to our custom network defined below.
    networks:
      - triton-net

# --- Custom Network Definition to Avoid Conflicts ---
networks:
  triton-net:
    driver: bridge
    ipam:
      driver: default
      config:
        # We define a custom subnet that is very unlikely to conflict with VPNs or other networks.
        - subnet: "10.57.0.0/24"
```

**Usage Note:** To compute embeddings for queries or passages, prepend the appropriate prefix to the input text before tokenizing:
- Query: "task: search result | query: " + text
- Passage: "title: none | text: " + text

Verify the server is running by checking http://localhost:8002/metrics or using curl:

```bash
curl -v localhost:8001/v2/health/ready
```

---

### Step 6: Service Verification

Create a conda environment and install dependencies:

```bash
conda create -n gemmatriton python=3.11
conda activate gemmatriton
pip install tritonclient[http] onnxruntime-gpu transformers torch numpy nvidia-ml-py
python verify_model.py
```

This should confirm the Triton deployment matches the local ONNX inference.
If issues persist, check Triton logs with `docker logs -f embgemmatriton`.
If GPU memory varies, test with consistent input sizes (e.g., always pad to 512 tokens).
