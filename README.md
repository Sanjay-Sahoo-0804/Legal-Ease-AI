# Legal-Ease AI - Enterprise Document RAG Platform

Legal-Ease AI is an enterprise-grade Retrieval-Augmented Generation (RAG) platform designed to handle messy, dense legal agreements, compliance contracts, and case files safely and accurately. By utilizing a hybrid retrieval strategy and layout-aware processing pipelines, the system eliminates traditional LLM hallucinations and provides auditable, clause-level attribution back to original legal source documents.

---

## 🚀 Key Features

- **End-to-End RAG Architecture:** Orchestrates layout-aware raw document ingestion, vector parsing, and LLM orchestration layers.
- **Hybrid Retrieval Strategy:** Combines deep semantic dense embeddings with precise token/keyword indexing to cleanly pull statutory references, specific article counts, or custom clauses.
- **Document Preprocessing & Pipeline:** Advanced text cleaning, noise reduction, and structural boundary extraction designed specifically for complex legal templates.
- **Model Fine-Tuning Sandbox:** Modular tools to adapt baseline open-source LLMs (like Llama or Gemma) specifically on custom downstream corporate legal corpuses.
- **Streamlit Frontend Dashboard:** An interactive workspace enabling teams to drop multi-page PDFs, initiate chats, view structural risk highlights, and review clear per-passage citations.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend Interface:** Streamlit (Python-driven interactive workspace UI)
- **Backend Framework:** FastAPI / Flask (Represented via modular server configurations)
- **Vector Index Engine:** Chromadb / FAISS (Flexible semantic dense matrix handling)
- **Data Pipelines:** PyPDF / PDFPlumber, Tokenizers, Hugging Face Transformers
- **Infrastructure Containers:** Docker, Docker-Compose

---

## 📁 Project Architecture & Directory Structure

Based on your repository snapshot, here is the exact architectural layout of the platform:

```text
Legal-Ease-AI-main/
├── backend/                # Server orchestration & retrieval core
│   ├── app.py              # Main API server initializing chat pipelines and endpoints
│   ├── config.py           # Global platform settings, model paths, and API keys
│   └── vector_store.py     # Hybrid embedding insertions, indexing, and lookup controls
├── data_pipeline/          # Document processing & optimization engines
│   ├── finetune.py         # Training loops and parameter configs to adapt open models
│   └── preprocess.py       # Sentence segmentation, table cleanup, and tokenization tools
├── frontend/               # User engagement dashboard
│   └── main.py             # Streamlit application layout (Uploaders, chat, and citation UI)
├── .gitignore              # Standard file exclusion rules for Git
├── Dockerfile              # Containerization configuration for the core services
├── docker-compose.yml      # Multi-container multi-tier execution conductor
└── requirements.txt        # Python dependency manifest for local environment setup
```

---

## ⚙️ Installation & Local Setup

Follow these sequential steps to set up and launch your environment locally:

### 1. Clone & Navigate into the Project
```bash
git clone <your-repository-url>
cd Legal-Ease-AI-main
```

### 2. Native Local Deployment
Ensure you have **Python 3.10+** running locally:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install application dependencies
pip install -r requirements.txt

# Start the Backend Server
python backend/app.py

# Start the Frontend Dashboard (in a separate terminal)
streamlit run frontend/main.py
```

### 3. Containerized Docker Deployment
If you prefer to spin up the platform within isolated multi-tier containers:

```bash
# Build and stand up the backend and frontend components automatically
docker-compose up --build
```
Once initialized, access your local workspace via the browser addresses displayed in your terminal (typically `http://localhost:8501` for the Streamlit dashboard).

---

## 🛣️ API & Module Blueprints

### `backend/`
- **`app.py`**: Defines operational routing infrastructure for processing text queries, uploading legal packets, and serving response tokens paired with source citation contexts.
- **`vector_store.py`**: Controls vector-space segmentation, database connection handshakes, and hybrid similarity score blending formulas.

### `data_pipeline/`
- **`preprocess.py`**: Scrubs incoming text files of formatting artifacts, structural inconsistencies, or scanning noise while ensuring legal structures remain logically grouped.
- **`finetune.py`**: Ingests your specialized compliance playbooks and contract history datasets to optimize domain-specific LLM fine-tuning runs.

