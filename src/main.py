"""FastAPI entry point for Mriia AI Tutor."""
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .tracing import init_tracing
from .agents import create_tutor_graph
from .agents.state import create_initial_state


# === Request/Response Models ===

class TutorRequest(BaseModel):
    """Request for the main tutoring endpoint."""
    query: str = Field(..., description="Teacher's query or topic request")
    grade: int = Field(9, description="Grade level (8 or 9)")
    subject: str = Field("Українська мова", description="Subject name")
    student_id: Optional[int] = Field(None, description="Student ID for personalization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Складні речення та їх ознаки",
                "grade": 9,
                "subject": "Українська мова",
                "student_id": None
            }
        }


class AnswerCheckRequest(BaseModel):
    """Request to check student answers."""
    query: str = Field(..., description="Original query")
    grade: int = Field(9, description="Grade level")
    subject: str = Field(..., description="Subject name")
    student_id: Optional[int] = Field(None, description="Student ID")
    student_answers: List[str] = Field(..., description="List of student answers")


class BenchmarkQuestion(BaseModel):
    """Single benchmark question."""
    question_id: str
    question_text: str
    answers: List[str]
    subject: str
    grade: int


class BenchmarkSolveRequest(BaseModel):
    """Request to solve benchmark questions."""
    questions: List[BenchmarkQuestion]


class TutorResponse(BaseModel):
    """Response from the tutoring endpoint."""
    lecture_content: str = ""
    control_questions: List[str] = []
    practice_questions: List[dict] = []
    sources: List[str] = []
    recommendations: str = ""
    error: Optional[str] = None


class AnswerCheckResponse(BaseModel):
    """Response from answer checking."""
    evaluation_results: List[dict]
    recommendations: str
    next_topics: List[str]


class BenchmarkSolveResponse(BaseModel):
    """Response from benchmark solving."""
    answers: List[dict]  # question_id -> answer_index mapping


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting Mriia AI Tutor")
    print(f"   LLM API: {settings.llm_base_url}")
    print(f"   Data dir: {settings.data_dir}")
    
    # Initialize tracing
    try:
        init_tracing()
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize tracing: {e}")
    
    # Create the LangGraph workflow
    app.state.tutor_graph = create_tutor_graph()
    print("✓ LangGraph workflow initialized")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Mriia AI Tutor")


# === FastAPI App ===

app = FastAPI(
    title="Mriia AI Tutor",
    description="AI-powered educational tutor for Ukrainian 8-9 grade students",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mriia-tutor"}


@app.post("/tutor/query", response_model=TutorResponse)
async def process_query(request: TutorRequest):
    """Process a teacher query through the full tutoring pipeline.
    
    This endpoint handles:
    1. Topic routing
    2. Context retrieval
    3. Personalization (if student_id provided)
    4. Content generation
    5. Practice question generation
    6. Self-validation
    """
    try:
        # Create initial state
        state = create_initial_state(
            teacher_query=request.query,
            grade=request.grade,
            subject=request.subject,
            student_id=request.student_id,
            mode="demo"
        )
        
        # Run the graph
        result = app.state.tutor_graph.invoke(state)
        
        return TutorResponse(
            lecture_content=result.get("lecture_content", ""),
            control_questions=result.get("control_questions", []),
            practice_questions=result.get("practice_questions", []),
            sources=result.get("sources", []),
            recommendations=result.get("recommendations", ""),
            error=result.get("error")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tutor/check-answers", response_model=AnswerCheckResponse)
async def check_answers(request: AnswerCheckRequest):
    """Check student answers and provide recommendations.
    
    This endpoint:
    1. Runs the full pipeline with student answers
    2. Evaluates each answer
    3. Generates personalized recommendations
    """
    try:
        # Create state with student answers
        state = create_initial_state(
            teacher_query=request.query,
            grade=request.grade,
            subject=request.subject,
            student_id=request.student_id,
            mode="practice"
        )
        state["student_answers"] = request.student_answers
        
        # Run the graph
        result = app.state.tutor_graph.invoke(state)
        
        return AnswerCheckResponse(
            evaluation_results=result.get("evaluation_results", []),
            recommendations=result.get("recommendations", ""),
            next_topics=result.get("next_topics", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark/solve", response_model=BenchmarkSolveResponse)
async def solve_benchmark(request: BenchmarkSolveRequest):
    """Solve benchmark questions for CodaBench evaluation.
    
    This endpoint is used for automated benchmark scoring.
    """
    from .llm import MamayLLM
    
    try:
        mamay = MamayLLM()
        answers = []
        
        for q in request.questions:
            result = mamay.solve_question(
                question_text=q.question_text,
                answers=q.answers,
                subject=q.subject
            )
            answers.append({
                "question_id": q.question_id,
                "answer_index": result["answer_index"],
                "answer_text": result["answer_text"]
            })
        
        return BenchmarkSolveResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Run directly ===

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
