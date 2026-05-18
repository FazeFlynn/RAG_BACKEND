# RAG System

A full-featured Retrieval-Augmented Generation (RAG) system that supports:

- **Document Q&A**: Upload PDFs, TXT, CSV, XLSX, DOCX files and ask questions about them
- **Web Search**: Automatically searches the web, scrapes results, and synthesizes answers
- **Smart Routing**: Automatically detects whether to search documents or the web
- **Conversational Memory**: Maintains context across multiple turns

## Architecture

```
app/
├── core/
│   └── config.py              # Settings from .env
├── models/
│   └── schemas.py             # Pydantic models
├── services/
│   ├── file_loader.py         # PDF, CSV, DOCX, etc. parsing
│   ├── chunker.py             # Text splitting with overlap
│   ├── embeddings.py          # Sentence-transformers embeddings
│   ├── vector_store.py        # ChromaDB vector storage
│   ├── retriever.py           # Vector search + reranking
│   ├── reranker.py            # Cross-encoder reranking
│   ├── llm.py                 # Ollama LLM interface
│   ├── web_search.py          # DuckDuckGo search + web scraping
│   ├── router.py              # Query intent classification
│   ├── conversation.py        # Chat memory
│   └── rag_pipeline.py        # Main orchestrator
├── api/routes/
│   ├── health.py              # Health check
│   ├── documents.py           # Upload/list/delete documents
│   └── chat.py                # Chat endpoint
└── ui/
    └── streamlit_app.py       # Web UI
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.ai/
# Then pull a model:
ollama pull qwen2.5:0.5b
# ollama pull llama3.1:8b
```

### 2. Set up the project

```bash
# Clone and enter the project
cd RAG

# Create virtual environment
python -m venv .venv
py -3.11 -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example env file
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/Mac

# Edit .env if needed (defaults work for local setup)
```

### 4. Start the API server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.
API docs at `http://localhost:8000/docs`.

### 5. Start the UI (separate terminal)

```bash
streamlit run app/ui/streamlit_app.py
```

The UI will be available at `http://localhost:8501`.

## Docker Setup

```bash
docker-compose up --build
```

Then pull the model inside the Ollama container:

```bash
# docker exec -it rag-ollama-1 ollama pull llama3.1:8b
docker exec -it rag-open-ollama-1 ollama pull tinyllama
```

## API Endpoints

### Chat

```bash
# Ask a question (auto-detects mode)
curl -X POST http://localhost:8011/chat/ -H "Content-Type: application/json" -d "{\"query\": \"Who is the founder of Anthropic?\"}"


curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'

# Force document Q&A mode
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document", "query_type": "document_qa"}'

# Force web search mode
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest AI news", "query_type": "web_search"}'
```

### Documents

```bash
# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@myfile.pdf"

# List indexed documents
curl http://localhost:8000/documents/

# Delete a document
curl -X DELETE http://localhost:8000/documents/myfile.pdf
```

### Health

```bash
curl http://localhost:8000/health
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

```bash
python eval/evaluate.py
```

## Configuration

All settings can be configured via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHUNK_SIZE` | `512` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of documents to retrieve |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results per query |

## Supported File Types

- PDF (`.pdf`)
- Plain text (`.txt`)
- Markdown (`.md`)
- CSV (`.csv`)
- Excel (`.xls`, `.xlsx`)
- Word (`.docx`)
- HTML (`.html`, `.htm`)
- JSON (`.json`)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| UI | React.js |
| Embeddings | sentence-transformers |
| Vector Store | ChromaDB |
| Reranker | cross-encoder |
| LLM | Ollama (Llama 3.1) |
| Web Search | DuckDuckGo | SearXNG
| Web Scraping | Trafilatura |
