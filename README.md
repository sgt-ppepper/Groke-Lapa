# Mriia AI Tutor

AI-powered educational tutor for Ukrainian 8-9 grade students. Built for Lapathon 2026.

## 🎯 Features

- **Topic Routing**: Match teacher queries to textbook topics using RAG (Retrieval-Augmented Generation)
- **Content Generation**: Create lecture materials (conspect) with Lapa LLM based on textbook pages
- **Practice Generation**: Generate 8-12 test questions with Mamay LLM
- **Self-Validation**: Solver loop to verify question correctness
- **Personalization**: Adapt content based on student trajectory
- **Recommendations**: Provide next steps based on performance

## 🛠️ Tech Stack

- **LangGraph** - Agent orchestration
- **ChromaDB** - Vector database for textbook topics and pages
- **Phoenix** - Observability & tracing
- **FastAPI** - API server
- **React + Vite** - Frontend UI
- **Lapa LLM** - Content generation (12B Instruct)
- **MamayLM** - Routing, practice, solving (12B Gemma-3)
- **Qwen** - Text embeddings

## 🚀 How to Run

### Prerequisites

- Docker and Docker Compose installed
- Node.js and npm (for frontend development)
- Python 3.10+ (for local development)
- API key for Lapathon LLM service

### Step 1: Setup Environment Variables

Create a `.env` file in the project root directory:

```bash
# Windows PowerShell
New-Item -Path .env -ItemType File

# Linux/Mac
touch .env
```

Add your API key to the `.env` file:

```env
LAPATHON_API_KEY=your_api_key_here
```

**Important:** Without this API key, the application will not be able to generate content.

### Step 2: Initialize ChromaDB Collections

Before running the application, you need to populate the ChromaDB database with textbook data.

#### Option A: Using Docker (Recommended)

```bash
# Start the containers first
docker-compose up -d

# Initialize TOC collection
docker-compose exec api python scripts/setup/setup_chroma_toc.py

# Initialize pages collection
docker-compose exec api python scripts/setup/setup_chroma_pages.py
```

#### Option B: Using Local Python

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize ChromaDB collections
python scripts/setup/setup_chroma_toc.py
python scripts/setup/setup_chroma_pages.py
```

**Note:** ChromaDB collections need to be initialized only once. The data will persist in the `chroma_db` directory (or Docker volume).

### Step 3: Start the Application

#### Option A: Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

This starts:
- **API server** at http://localhost:8000
- **Phoenix UI** at http://localhost:6006 (for tracing and observability)

#### Option B: Local Development

**Terminal 1 - Backend:**

```bash
# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Run FastAPI server
python -m src.main
```

The API will be available at http://localhost:8000

**Terminal 2 - Frontend:**

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173 (or another port shown in terminal)

### Step 4: Verify Installation

1. **Check API Health:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy","service":"mriia-tutor"}`

2. **Test a Query:**
   - Open http://localhost:5173 (or your frontend URL)
   - Or use curl:
   ```bash
   curl -X POST http://localhost:8000/tutor/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Складні речення та їх ознаки",
       "grade": 9,
       "subject": "Українська мова"
     }'
   ```

3. **Check Phoenix Tracing:**
   - Open http://localhost:6006
   - You should see traces of LLM calls and agent execution

## 📡 API Endpoints

### POST /tutor/query
Full tutoring cycle: topic routing → content generation → practice tests

**Request:**
```json
{
  "query": "Складні речення та їх ознаки",
  "grade": 9,
  "subject": "Українська мова",
  "student_id": 8  // optional
}
```

**Response:**
```json
{
  "lecture_content": "# Конспект...",
  "matched_topics": [{"topic": "...", "grade": 9, "subject": "..."}],
  "matched_pages": [{"content": "..."}],
  "practice_questions": [...],
  "control_questions": [],
  "sources": [],
  "recommendations": "",
  "error": null
}
```

### GET /students/list
Get list of available students

**Query Parameters:**
- `subject` (optional): Filter by subject
- `grade` (optional): Filter by grade

### GET /students/{student_id}/info
Get detailed information about a student

**Query Parameters:**
- `subject` (optional): Filter by subject

### POST /tutor/check-answers
Check student answers and get recommendations

### POST /benchmark/solve
Solve benchmark questions for CodaBench evaluation

## 📁 Project Structure

```
Groke-Lapa/
├── src/                          # Backend source code
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Settings and configuration
│   ├── tracing.py                # Phoenix tracing setup
│   ├── agents/                   # LangGraph workflow
│   │   ├── graph.py              # Workflow definition (topic router → content → practice)
│   │   ├── state.py              # State schema
│   │   └── topic_router.py       # RAG-based topic routing
│   ├── llm/                      # LLM clients
│   │   ├── lapa.py               # Lapa LLM for content generation
│   │   ├── mamay.py              # MamayLM for routing and practice
│   │   └── embeddings.py         # Qwen embeddings
│   └── personalization_engine.py # Student personalization
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── App.jsx               # Main app component
│   │   └── components/
│   │       ├── QueryForm.jsx     # Main query form
│   │       ├── BenchmarkSolver.jsx
│   │       └── AnswerCheck.jsx
│   └── package.json
├── scripts/                      # Utility scripts
│   ├── setup/                    # ChromaDB initialization
│   │   ├── setup_chroma_toc.py   # Setup TOC collection
│   │   └── setup_chroma_pages.py # Setup pages collection
│   └── examples/                 # Example scripts
├── Lapathon2026_Mriia_public_files/  # Data files
│   └── text-embedding-qwen/
│       ├── toc_for_hackathon_with_subtopics.parquet
│       └── pages_for_hackathon.parquet
├── data/                         # Runtime data
│   ├── chroma_db/                # ChromaDB storage (created after setup)
│   ├── benchmark_scores.parquet
│   └── benchmark_absences.parquet
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LAPATHON_API_KEY` | API key for Lapathon LLM service | - | **Yes** |
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix collector URL | http://localhost:6006/v1/traces | No |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | ./chroma_db | No |
| `DATA_DIR` | Path to parquet files | /data (Docker) or ./data (local) | No |

### ChromaDB Collections

After running setup scripts, two collections are created:

1. **`toc_topics`** - Table of Contents topics for semantic search
2. **`pages`** - Textbook pages linked to topics for content retrieval

## 📊 How It Works

### Workflow

1. **Topic Router**: Matches teacher query to textbook topics using RAG
   - Uses Qwen embeddings for semantic search
   - Finds relevant topics from TOC collection
   
2. **Context Retriever**: Retrieves actual textbook pages for matched topics
   - Queries ChromaDB pages collection by `book_topic_id`
   - Retrieves up to 10 pages for context

3. **Content Generator**: Generates lecture content (conspect) using Lapa LLM
   - Uses retrieved pages as context
   - Generates structured Markdown content with formulas

4. **Practice Generator**: Generates practice questions using Mamay LLM
   - Creates 8-12 multiple-choice questions
   - Validates answers using solver loop

5. **Personalization**: Adapts content based on student performance (if `student_id` provided)

### RAG (Retrieval-Augmented Generation)

The system uses RAG to ground responses in textbook content:

- **Embeddings**: Qwen text embeddings for semantic search
- **Vector Store**: ChromaDB with two collections:
  - TOC topics for topic routing
  - Pages for content retrieval
- **Retrieval**: Retrieves relevant pages based on matched topics
- **Generation**: LLMs generate content using retrieved pages as context

## 🐛 Troubleshooting

### ChromaDB Collection Not Found

**Error:** `ChromaDB collection 'toc_topics' not found`

**Solution:**
```bash
# Run setup scripts
docker-compose exec api python scripts/setup/setup_chroma_toc.py
docker-compose exec api python scripts/setup/setup_chroma_pages.py
```

### API Key Not Set

**Error:** `LAPATHON_API_KEY is not set`

**Solution:**
1. Create `.env` file in project root
2. Add `LAPATHON_API_KEY=your_key_here`
3. Restart Docker containers: `docker-compose restart`

### Empty Lecture Content

**Symptoms:** `lecture_content` is empty in API response

**Possible Causes:**
1. No pages retrieved from ChromaDB - check if `matched_pages` is empty
2. LLM API error - check server logs for `[Content Generator] Error`
3. Topic routing failed - check if `matched_topics` is empty

**Solution:**
- Check server logs for error messages
- Verify ChromaDB collections are populated
- Ensure API key is valid

### Port Already in Use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:**
```bash
# Stop other containers using port 8000
docker ps  # Find container using port 8000
docker stop <container_id>

# Or change port in docker-compose.yml
```

## 📊 Tracing & Observability

Phoenix UI is available at http://localhost:6006 when running with Docker.

Traces include:
- LLM call latency and tokens
- Agent node execution times
- Retrieval performance
- Topic routing matches
- Content generation success/failure

## 🧪 Testing

### Test Topic Router

```bash
python scripts/examples/test_topic_router.py
```

### Test API Endpoint

```bash
curl -X POST http://localhost:8000/tutor/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Поясни формулу дискриміната",
    "grade": 9,
    "subject": "Алгебра"
  }'
```

## 📝 Notes

- ChromaDB data persists in Docker volume `chroma_data` or local `chroma_db/` directory
- First startup may take longer while ChromaDB collections are being created
- Frontend connects to backend at `http://localhost:8000` by default
- Set `VITE_API_URL` environment variable to change frontend API URL

## 👥 Team

Groke-Lapa Team - Lapathon 2026

## 📄 License

[Add your license here]
