# AGENTS.md - Mriia AI Tutor

Guidelines for AI coding agents working in this repository.

## Project Overview

Python-based AI tutoring system for Ukrainian 8th-9th grade students. Built with FastAPI, LangGraph, and ChromaDB.

**Tech Stack:**
- Python 3.11
- FastAPI (async REST API)
- LangGraph (agent orchestration)
- ChromaDB (vector database)
- Pydantic (data validation)
- OpenTelemetry + Phoenix (observability)

## Build & Run Commands

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

### Run

```bash
# Run locally (development)
python -m src.main

# Run with Docker
docker-compose up --build

# API: http://localhost:8000
# Phoenix UI: http://localhost:6006
```

### Testing

No test infrastructure is currently configured. To add tests:

```bash
# Install test dependencies (add to requirements.txt first)
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run single test file
pytest tests/test_main.py

# Run single test function
pytest tests/test_main.py::test_function_name -v

# Run with verbose output
pytest -v
```

### Linting (Recommended Setup)

No linting is configured. Recommended tools:

```bash
# Install linting tools
pip install ruff black mypy

# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## Project Structure

```
src/
├── main.py           # FastAPI entry point & endpoints
├── config.py         # Pydantic settings management
├── tracing.py        # Phoenix OpenTelemetry setup
├── agents/
│   ├── graph.py      # LangGraph workflow definition
│   └── state.py      # TypedDict state schema + dataclasses
├── llm/
│   ├── lapa.py       # Lapa LLM (content generation)
│   ├── mamay.py      # Mamay LLM (routing, practice, solving)
│   └── embeddings.py # Qwen text embeddings
├── data/             # Data loaders (placeholder)
└── benchmark/        # Benchmark solver (placeholder)
```

## Code Style Guidelines

### Imports

Organize imports in three groups with blank line separators:

```python
# 1. Standard library
from typing import TypedDict, Optional, List, Literal
from dataclasses import dataclass, field

# 2. Third-party
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 3. Local (relative imports)
from ..config import get_settings
from .state import TutorState
```

- Use `from X import Y` style
- Use relative imports for local modules (`.` and `..`)
- Sort alphabetically within each group

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `TutorState`, `LapaLLM` |
| Functions | snake_case | `create_tutor_graph`, `get_settings` |
| Variables | snake_case | `teacher_query`, `matched_topics` |
| Files/Modules | snake_case | `embeddings.py`, `graph.py` |
| Packages | snake_case, singular | `llm/`, `agents/`, `data/` |

### Type Annotations

Always use explicit type hints:

```python
# Function signatures
def get_settings() -> Settings:

# Optional values
student_id: Optional[int] = None

# Constrained strings
mode: Literal["demo", "benchmark", "practice"]

# Collections
matched_topics: List[dict]
```

**Use the right tool for each job:**
- `TypedDict` for state objects passed through workflows
- `@dataclass` for internal data models
- `pydantic.BaseModel` for API request/response models

### Documentation

Use Google-style docstrings:

```python
def generate_practice(
    self,
    topic: str,
    subtopics: List[str],
    grade: int = 9,
) -> str:
    """Generate practice questions for a topic.
    
    Args:
        topic: Main topic name
        subtopics: List of subtopics to cover
        grade: Grade level (8 or 9)
        
    Returns:
        Generated questions in structured format
    """
```

Module docstrings should describe purpose:

```python
"""Lapa LLM client for content generation."""
```

Use section separators for large files:

```python
# === Request/Response Models ===
# === Endpoints ===
```

### Error Handling

FastAPI endpoints:

```python
try:
    result = app.state.tutor_graph.invoke(state)
    return TutorResponse(...)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

Guard clauses for early returns:

```python
if not texts:
    return []
```

### Formatting

- 4-space indentation
- Line length: ~88-100 characters (Black-compatible)
- Two blank lines between top-level definitions
- One blank line between class methods
- Trailing commas in multi-line structures
- f-strings for string formatting

### Package Exports

Define `__all__` in `__init__.py`:

```python
from .lapa import LapaLLM
from .mamay import MamayLLM
from .embeddings import QwenEmbeddings

__all__ = ["LapaLLM", "MamayLLM", "QwenEmbeddings"]
```

### Common Patterns

**Cached settings:**
```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

**Async context manager for lifecycle:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic
```

**Pydantic model with Field descriptions:**
```python
class TutorRequest(BaseModel):
    query: str = Field(..., description="Teacher's query")
    grade: int = Field(9, description="Grade level (8 or 9)")
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tutor/query` | Full tutoring pipeline |
| POST | `/tutor/check-answers` | Answer evaluation |
| POST | `/benchmark/solve` | Benchmark question solving |

## Environment Variables

See `.env.example` for required variables:
- `LAPATHON_API_KEY` - API key for LLM services
- `DATA_DIR` - Path to data files (parquet)
- Model configuration settings
