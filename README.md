# graph-rag-llama

# 🛠️ Environment Setup & Troubleshooting Guide

## Overview
Building a fully local Hybrid Retrieval-Augmented Generation (RAG) system requires integrating bleeding-edge AI libraries with local infrastructure (Docker, Git, Python). During the environment provisioning on a legacy macOS machine (Apple Silicon), several system-level and dependency constraints were encountered.

This document serves as a comprehensive log of the errors, their root causes, and the specific mitigation steps required to establish a stable build environment.

---

### 1. The Missing `pip` Command
**Error:**
`bash: pip: command not found`

**Context:** The terminal did not recognize `pip`, either because the virtual environment didn't activate properly or the system defaults to `pip3`.
**Solution:** By calling the python module directly, we bypassed the missing alias.
```bash
python -m pip install -r requirements.txt
```

### 2. The Missing python Command
**Error:**
`bash: python: command not found`

**Context:** Modern macOS and Linux distributions have completely deprecated the python command in favor of python3.
**Solution:** Verified Python 3 was installed and swapped commands to explicitly use python3.

```bash
python3 --version
python3 -m venv venv
```

### 3. Pip Dependency Resolution Loop (Backtracking)
**Error:**
INFO: pip is looking at multiple versions of llama-index-graph-stores-neo4j to determine which version is compatible...
INFO: This is taking longer than usual...

**Context:** LlamaIndex updates its modular packages frequently. When installing them all at once alongside chromadb (which has strict Pydantic requirements), pip got stuck in an infinite dependency resolution loop trying to find the perfect combination.
**Solution:** Aborted the installation and installed the packages sequentially, starting with the strictest databases first.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install chromadb neo4j python-dotenv
python3 -m pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface
python3 -m pip install llama-index-vector-stores-chroma llama-index-graph-stores-neo4j
```

### 4. The Python 3.14 onnxruntime Wheel Crash
**Error:**
ERROR: Cannot install llama-index-vector-stores-chroma... because these package versions have conflicting dependencies... The conflict is caused by... onnxruntime. (Running Python 3.14)

**Context:** Python 3.14 is too new. The AI ecosystem relies on heavy C++ bindings (like onnxruntime used by ChromaDB), which take months to release pre-compiled wheels for brand-new Python versions. pip crashed because there was no compatible ML backend for 3.14.
**Solution:** Nuked the 3.14 environment and downgraded to the stable, industry-standard Python 3.11.
```bash
deactivate
rm -rf venv
brew install python@3.11
python3.11 -m venv venv
```

### 5. Missing Docker & OS Incompatibility
**Errors:**
bash: docker-compose: command not found
Error: This software does not run on macOS versions older than Sonoma.

**Context:** Docker Desktop recently dropped support for older macOS versions (like Big Sur/Monterey). The standard Docker Desktop installer blocked the installation entirely.
**Solution:** Bypassed the heavy GUI app and installed Colima, a lightweight CLI alternative for the Docker daemon, via Homebrew.
```bash
brew install colima docker docker-compose
colima start
```

### 6. Git HTTP/2 RPC Network Failure
**Error:**
error: RPC failed; curl 18 HTTP/2 stream 5 was not closed cleanly before end of the underlying stream
fatal: early EOF

**Context:** While Homebrew was trying to download the heavy Docker CLI packages from GitHub, the default HTTP/2 protocol dropped the connection mid-download.
**Solution:** Forced Git to downgrade to the more stable HTTP/1.1 protocol and massively increased its memory buffer.
```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```
# Ran the brew install again, then reverted Git to HTTP/2

### 7. Ghost Docker Desktop Credentials
**Error:**
error getting credentials - err: exec: "docker-credential-desktop": executable file not found in $PATH, out:

**Context:** A previous, failed attempt to install Docker Desktop left behind a ~/.docker/config.json file telling the system to use the (non-existent) Desktop app to manage passwords when pulling images.
**Solution:** Renamed/removed the corrupted configuration file so Docker/Colima could generate a fresh one.
```bash
mv ~/.docker/config.json ~/.docker/config.json.bak
```

### 8. Docker CLI Syntax Typo
**Error:**
unknown shorthand flag: 'd' in -d

**Context:** Typing docker -compose up -d or mis-spacing the V2 docker compose command caused the CLI to interpret -d as a flag for the base docker command, which doesn't exist.
**Solution:** Reverted to the explicit hyphenated binary installed via Homebrew.
```bash
docker-compose up -d
```

### 9. The docker.sock Socket Disconnection
**Error:**
unable to get image 'neo4j:5.15.0': failed to connect to the docker API at unix:///var/run/docker.sock; ... no such file or directory

**Context:** The Docker CLI was looking for the background engine at Apple's default socket location (/var/run/docker.sock). However, because we used Colima to bypass the OS limits, the engine was running at Colima's specific back-door socket.
**Solution:** Manually exported the environment variable to point the CLI to Colima's socket.
```bash
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
docker ps  # Verified connection successful
docker-compose up -d
```

### 10. macOS Version Incompatibility for Ollama (App & Binary)
**Error:** `Error: This software does not run on macOS versions older than Sonoma.`
`mlx: A full installation of Xcode.app 15.0 is required...`

**Context:** The Ollama GUI app and the Homebrew formula both require macOS Sonoma+ or modern Xcode versions.
**Solution:** Containerized the LLM engine. By running Ollama inside a Docker container via Colima, we bypassed the macOS version restrictions entirely.
```bash
# Started Ollama via Docker
docker-compose -f ollama-compose.yml up -d
# Pulled model via exec
docker exec -it $(docker ps -q -f ancestor=ollama/ollama:latest) ollama run llama3
```

### 11. Local LLM Out of Memory (Ollama)
**Error:** `500 Internal Server Error: model requires more system memory (4.6 GiB) than is available (1.8 GiB)`
**Context:** Llama 3 crashed because the Colima Docker engine defaults to only 2GB of RAM, which is insufficient for a 7B parameter model.
**Solution:** Shut down the Docker engine and restarted Colima with increased resource allocation.
```bash
# Stop the current underpowered VM
colima stop

# Restart with 4 CPUs and 8GB of RAM
colima start --cpu 4 --memory 8 --disk 60

# Ensure the terminal is linked to the new Colima instance
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"

# Spin the containers back up instantly
docker-compose up -d
```

### 12. Hugging Face Network / SSL Firewall Blocks
**Error:** [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate and CAS Client Error (403 Forbidden)

**Context:** The corporate network's strict SSL firewall and Hugging Face's new XetHub storage system blocked Python from downloading embedding models directly to the Mac host.
**Architectural Pivot:** Instead of fighting the host firewall, we shifted the architecture to a 100% Dockerized AI Stack by offloading embeddings to the containerized Ollama instance.
**Solution:**

```bash
# 1. Install the Ollama embedding connector for LlamaIndex
pip install llama-index-embeddings-ollama

# 2. Pull the embedding model directly inside the Docker container (bypassing the Mac's SSL block)
docker exec -it $(docker ps -q -f ancestor=ollama/ollama:latest) ollama pull nomic-embed-text
Note: We then updated config.py to route all embedding tasks locally to OllamaEmbedding(model_name="nomic-embed-text") instead of HuggingFaceEmbedding.
```

### 13. Neo4j Connection Refused / Boot Crash
**Error:** [Errno 61] Connection refused on port 7687 and Invalid admin username, it must be neo4j in the Docker logs.

**Context:** Neo4j requires the absolute first initialization of a brand new database to use the default neo4j admin username. Attempting to set a custom username (honey_badger_66) on the first boot caused the container to crash, leaving corrupted volume data behind.
**Solution:**

```bash
# 1. Check the "black box" flight recorder logs to find the exact crash reason
docker logs rag-demo-neo4j-1

# 2. Revert NEO4J_AUTH to neo4j/password123 in docker-compose.yml, then delete the corrupted volume data
rm -rf neo4j_data

# 3. Restart the container to allow a clean initialization
docker-compose up -d

# 4. Verify both Ollama and Neo4j are now stable and running
docker ps
```

### 14. Async Event Loop Crash & Missing File Readers
**Error:** `FAILED: Event loop is closed` during the "Generating embeddings" phase, accompanied by `WARNING: llama-index-readers-file package not found`.
**Context:** LlamaIndex uses asynchronous Python (`asyncio`) to speed up processing. It attempted to send all text chunks to the local Ollama embedding container simultaneously. The local Dockerized Ollama was overwhelmed by the sudden concurrency, causing it to drop the connection and forcing Python's async event loop to crash.
**Solution:** 1. Installed the missing file reader package and the `nest-asyncio` patch.
2. Applied the async patch at the top of the configuration file.
3. Throttled the embedding requests by reducing the batch size to 1, acting as a "traffic cop" to feed Ollama one chunk at a time.

```bash
# 1. Install missing dependencies
pip install llama-index-readers-file nest-asyncio
# 2. Add to the very top of config.py to patch the event loop
import nest_asyncio
nest_asyncio.apply()

# 3. Add to the setup_environment() function in config.py to throttle requests
Settings.embed_batch_size = 1
```
