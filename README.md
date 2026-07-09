# Gemma4 Private Learning Agent

A private AI learning assistant prototype built with **Gemma4**, **vLLM**, **FastAPI**, **RAG**, persistent conversation memory, feedback collection, and LoRA dataset export.

This project is designed as an educational AI agent for beginner-friendly artificial intelligence learning. It supports private knowledge retrieval, web-based interaction, conversation history, response quality feedback, and high-quality sample collection for future LoRA fine-tuning.

---

## Overview

The system aims to build a closed-loop learning assistant:

```text
Private Knowledge Base
        ↓
RAG Retrieval
        ↓
Gemma4 Inference
        ↓
Web Learning Assistant
        ↓
Conversation Memory + Feedback
        ↓
LoRA Training Data Export

The core idea is not only to answer questions, but also to collect high-quality interaction data that can later be used to improve the assistant's behavior through LoRA fine-tuning.

Features
Local Gemma4 inference with vLLM
FastAPI backend
Lightweight web frontend
RAG-based private knowledge retrieval
Markdown knowledge base support
Persistent conversation history with SQLite
Conversation rename, delete, and export
Message-level response quality feedback
Training sample selection for LoRA
LoRA dataset export script
Deployment scripts for web backend, vLLM, Nginx, and systemd
Project Structure
gemma4_learning_agent/
├── app/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── rag.py
│   │   │   ├── providers.py
│   │   │   ├── schemas.py
│   │   │   ├── storage.py
│   │   │   └── conversation_store.py
│   │   ├── data/
│   │   │   └── knowledge/
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── build_lora_dataset_v1.py
│   │
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   │
│   ├── deploy/
│   │   ├── 01_setup_web_env.sh
│   │   ├── 02_setup_vllm_env.sh
│   │   ├── 03_download_gemma4_12b.sh
│   │   ├── 04_start_dev.sh
│   │   ├── 05_start_vllm_with_lora.sh
│   │   ├── nginx.conf
│   │   ├── gemma4-learning-api.service
│   │   ├── gemma4-vllm.service
│   │   └── install_systemd.sh
│   │
│   ├── training/
│   │   ├── lora_train_gemma4_qlora.py
│   │   └── requirements-training.txt
│   │
│   ├── README_DEPLOY_CN.md
│   └── MIGRATION_MAP.md
│
├── .gitignore
└── README.md
Main Modules
1. RAG Knowledge Assistant

The system supports a local Markdown-based knowledge base. Documents are loaded, chunked, indexed, and retrieved according to user questions. Retrieved evidence is passed into the model context to improve grounding.

Example knowledge files are stored in:

app/backend/data/knowledge/

The current sample knowledge base focuses on beginner-level artificial intelligence learning, including:

AI fundamentals
Python learning path
Data processing
Machine learning
Deep learning
Large language models
RAG
LoRA
AI learning agent design
AI ethics and learning boundaries
2. Local Gemma4 Inference

The backend is designed to connect to a local OpenAI-compatible vLLM service.

Typical vLLM service:

vllm serve /path/to/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192

The FastAPI backend calls this local model endpoint for generation.

3. Web Frontend

The frontend provides a browser-based learning assistant interface.

Supported functions include:

Chat with the learning assistant
Knowledge-based question answering
Conversation history
Conversation rename and delete
Conversation export
Response quality rating
Marking answers as LoRA training candidates

Frontend files:

app/frontend/index.html
app/frontend/app.js
app/frontend/styles.css
4. Persistent Conversation Memory

Conversation history is stored with SQLite.

The system stores:

Conversation metadata
User messages
Assistant messages
Retrieved evidence
Model information
Response quality feedback
LoRA training selection flags

The runtime database is ignored by Git and should not be uploaded publicly.

5. Feedback-Driven LoRA Data Collection

The system supports response-level feedback.

A user can mark high-quality answers as training candidates. These selected samples can then be exported into a JSONL dataset for LoRA fine-tuning.

Export script:

app/backend/build_lora_dataset_v1.py

Typical output format:

{
  "messages": [
    {
      "role": "user",
      "content": "What is RAG?"
    },
    {
      "role": "assistant",
      "content": "RAG means retrieval-augmented generation..."
    }
  ],
  "meta": {
    "rating": 5,
    "agent_mode": "qa",
    "conversation_id": "..."
  }
}
Quick Start
1. Clone the repository
git clone https://github.com/qianmo123321/gemma4-learning-agent.git
cd gemma4-learning-agent
2. Create backend environment
cd app/backend
pip install -r requirements.txt
3. Configure environment variables

Copy the example environment file:

cp .env.example .env

Then edit .env according to your local model path, vLLM endpoint, database path, and knowledge base settings.

nano .env
4. Start vLLM

Start your local Gemma4 vLLM service separately. Example:

vllm serve /path/to/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 8192
5. Start backend
cd app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
6. Serve frontend

The frontend can be served through Nginx or any static file server. For deployment, see:

app/README_DEPLOY_CN.md
app/deploy/nginx.conf
Deployment

Deployment scripts are provided in:

app/deploy/

They include environment setup, vLLM startup, development startup, Nginx configuration, and systemd service examples.

For Chinese deployment notes, see:

app/README_DEPLOY_CN.md
LoRA Dataset Export

After collecting conversation feedback, run:

cd app/backend
python build_lora_dataset_v1.py

The script exports high-quality selected samples from the SQLite conversation database.

Recommended dataset sizes:

Small functional validation: 50-100 high-quality samples
Initial LoRA training:       300-1000 high-quality samples
Stable behavior tuning:      1000+ high-quality samples
What Is Not Included

For security and storage reasons, this repository does not include:

Gemma4 model weights
Hugging Face cache
Conda environments
Runtime SQLite database
Private .env files
API keys or access tokens
Large training checkpoints

These files are ignored by .gitignore.

Intended Use

This project is intended for:

AI education demonstrations
Private learning assistant prototypes
RAG-based course material question answering
Feedback-driven LoRA data collection
Lightweight local LLM application development

It is currently an educational prototype rather than a production-ready multi-user system.

Repository

GitHub:

https://github.com/qianmo123321/gemma4-learning-agent
License

No license has been specified yet. Please contact the repository owner before using this project for commercial purposes.
