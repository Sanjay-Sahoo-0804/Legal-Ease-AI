Legal-Ease AI - Enterprise Document RAG Platform
Legal-Ease AI is an enterprise-grade Retrieval-Augmented Generation (RAG) platform tailored for legal professionals, compliance officers, and corporate legal departments. The system provides secure, highly precise, and completely auditable analysis of dense legal documents, contracts, and case files. Built using an advanced AI architecture, the platform enforces strict grounding, multi-layered document ingestion, and verifiable citation mechanics to eliminate LLM hallucinations while parsing intricate legal layouts.
🚀 Key Features
• Advanced Ingestion & Parsing: Extract structured content from complex, multi-column legal documents and scanned PDFs using OCR while preserving logical reading orders and tabular layouts.
• Hybrid Retrieval Mechanism: Combines semantic vector search for high-level conceptual queries with BM25 keyword matching to accurately pinpoint exact statutory codes, clause numbers, and legal citations.
• Strict Grounding & Provenance: Enforces deterministic text extraction prompts that prevent the LLM from synthesizing outside knowledge, requiring per-passage, auditable citations mapped straight to the document section.
• Interactive Document Chat: Deep interrogation of multi-page agreements to instantly isolate indemnification liabilities, termination clauses, or regulatory obligations.
• Automated Risk Assessment: Agentic compliance workflows that cross-examine contract clauses against predefined corporate playbooks to automatically flag high-risk terms or missing protections.
• Drafting Studio: Template-driven compilation tools to rapidly generate first-pass legal notices, formal responses, and compliance briefs directly grounded in case files.
🛠️ Tech Stack & Dependencies
• AI/LLM Execution: LangChain/LlamaIndex (Orchestration Framework), OpenAI GPT / Anthropic Claude / Llama-3 (Legal Fine-Tuned)
• Vector Search & Storage: Pinecone / Qdrant / Milvus (High-performance hybrid vector indexes)
• Document Processing: Unstructured.io / PyPDF / Tesseract OCR (Advanced layout parsing)
• Backend Infrastructure: Python, FastAPI
• Data Validation & Orchestration: Pydantic (Schema enforcement), Celery (Asynchronous background document ingestion jobs)
• Security & Caching: Redis (Session state & query caching), JWT (Role-Based Access Control)
📁 Project Architecture & Directory Structure
text
Legal-Ease-AI/
├── agents/                 # Autonomous workflows and evaluation logic
│   ├── compliance.py       # Risk-checking engine running playbook comparisons
│   └── drafter.py          # Document template formatting and text generation hooks
├── config/                 # Platform setups and prompt definitions
│   ├── prompts.py          # Grounded legal prompts with strict citation rules
│   └── settings.py         # System configuration management (FastAPI, Vector DB)
├── controllers/            # API endpoint logic (translates HTTP to services)
│   ├── auth.py             # User access controls and enterprise SSO hooks
│   ├── document.py         # Document lifecycle handling (Upload, Process, Delete)
│   └── query.py            # Chat execution and citation compiler pipelines
├── ingestion/              # Ingestion engine and pipeline logic
│   ├── ocr_engine.py       # Layout-aware document OCR text parsing
│   └── splitter.py         # Legal clause-aware chunking components
├── models/                 # Database schemas and validation data shapes
│   ├── document.py         # Document registry, ownership metadata, and processing states
│   └── user.py             # RBAC roles, team assignments, and security profiles
├── retrieval/              # Query routing and vector infrastructure
│   ├── hybrid_search.py    # Merging layers for BM25 and Dense Vector indexes
│   └── reranker.py         # Cross-encoder scoring layers maximizing precision
├── routers/                # Modular FastAPI routing endpoints
│   ├── api_v1.py           # Core workspace orchestrations and routes
│   └── health.py           # Infrastructure performance tracking diagnostics
├── utils/                  # Core helpers and cross-cutting systems
│   ├── exceptions.py       # Custom asynchronous API error handling classes
│   └── logger.py           # Auditable system transaction trace structures
├── .env                    # AI keys, DB connection endpoints, and secrets (gitignored)
├── main.py                 # Core app initialization and API lifecycle runner
└── requirements.txt        # Manifest file managing project dependencies
Use code with caution.
⚙️ Installation & Local Setup
Follow these sequential steps to set up the development environment locally:
1. Prerequisites
Ensure you have Python 3.10+, a running instance of Redis, and credentials for your chosen Vector database cluster.
2. Clone the Repository
bash
git clone https://github.com
cd legal-ease-ai
Use code with caution.
3. Initialize a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Use code with caution.
4. Install Dependencies
bash
pip install -r requirements.txt
Use code with caution.
5. Configure Environment Variables
Create a file named .env in the root folder of the project and plug in your system integrations:
env
OPENAI_API_KEY=your_openai_or_anthropic_api_key
VECTOR_DB_URL=your_vector_database_endpoint
VECTOR_DB_API_KEY=your_vector_database_secret_key
DATABASE_URL=your_relational_metadata_db_connection
REDIS_URL=redis://localhost:6379/0
Use code with caution.
6. Boot the Application Server
bash
uvicorn main:app --reload --port 8000
Use code with caution.
The server will bind onto http://localhost:8000. You can explore the interactive API specification dashboard via http://localhost:8000/docs.
🛣️ API & Route Endpoint Blueprint
Document Lifecycle Routes (/api/v1/documents)
• GET /documents - Fetch all accessible corporate legal assets within the workspace.
• POST /documents/upload - Securely ingest multi-page contract documents and trigger the extraction pipeline.
• GET /documents/:id - Fetch target document processing metadata, ownership details, and text state indicators.
• DELETE /documents/:id - Purge a legal file from local storage, its relational footprints, and corresponding vector segments.
RAG & Query Routes (/api/v1/query)
• POST /query/chat - Dispatch contextual questions against isolated files with grounding parameters.
• POST /query/evaluate - Execute automatic compliance audits comparing uploaded text arrays to strict legal checklists.
• POST /query/draft - Generate template responses based on specified extraction contexts.
Enterprise Identity Routes (/api/v1/auth)
• POST /auth/register - Create user identities with granular permission controls.
• POST /auth/token - Authenticate system profiles to yield secure bearer JWT keys.
• POST /auth/logout - Invalidate target session bearer contexts to securely log out the active profile.
