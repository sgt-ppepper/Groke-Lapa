# Mriia AI Tutor

AI-powered educational tutor for Ukrainian 8-9 grade students. Built for Lapathon 2026.

## 🎯 Features

- **Topic Routing**: Match teacher queries to textbook topics
- **Content Generation**: Create lecture materials with Lapa LLM
- **Practice Generation**: Generate 8-12 test questions with Mamay LLM
- **Self-Validation**: Solver loop to verify question correctness
- **Personalization**: Adapt content based on student trajectory
- **Recommendations**: Provide next steps based on performance

## 🛠️ Tech Stack

- **LangGraph** - Agent orchestration
- **ChromaDB** - Vector database
- **Phoenix** - Observability & tracing
- **FastAPI** - API server
- **Lapa LLM** - Content generation (12B Instruct)
- **MamayLM** - Routing, practice, solving (12B Gemma-3)
- **Qwen** - Text embeddings

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Copy environment file
cp .env.example .env

# Add your API key
echo "LAPATHON_API_KEY=your_key_here" >> .env
```

### 2. Run with Docker

```bash
docker-compose up --build
```

This starts:
- API server at http://localhost:8000
- Phoenix UI at http://localhost:6006

### 3. Run Locally (Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Create data symlink
ln -s ../Lapathon2026\ Mriia\ public\ files data

# Run the server
python -m src.main
```

## 📡 API Endpoints

### POST /tutor/query
Full tutoring cycle: topic routing → content → practice tests

```bash
curl -X POST http://localhost:8000/tutor/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Складні речення та їх ознаки",
    "grade": 9,
    "subject": "Українська мова"
  }'
```

### POST /tutor/check-answers
Check student answers and get recommendations

### POST /benchmark/solve
Solve benchmark questions for CodaBench evaluation

## 📁 Project Structure

```
Groke-Lapa/
├── src/
│   ├── main.py           # FastAPI entry point
│   ├── config.py         # Settings
│   ├── tracing.py        # Phoenix tracing
│   ├── agents/           # LangGraph workflow
│   │   ├── graph.py      # Workflow definition
│   │   └── state.py      # State schema
│   ├── llm/              # LLM clients
│   │   ├── lapa.py       # Lapa LLM
│   │   ├── mamay.py      # MamayLM
│   │   └── embeddings.py # Qwen embeddings
│   ├── Lapathon2026_Mriia_public_files/  # Data files
│   └── benchmark/        # Benchmark solver (TODO)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LAPATHON_API_KEY` | API key for LLMs | required |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix collector URL | http://localhost:6006/v1/traces |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | ./chroma_db |
| `DATA_DIR` | Path to parquet files | Groke-Lapa/Lapathon2026_Mriia_public_files |

## 📊 Tracing

Phoenix UI available at http://localhost:6006 when running with Docker.

Traces include:
- LLM call latency and tokens
- Agent node execution times
- Retrieval performance

## 👥 Team

Groke-Lapa Team - Lapathon 2026
