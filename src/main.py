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
    lecture_content: str = Field(default="", description="Generated lecture content")
    control_questions: List[str] = Field(default_factory=list, description="Control questions")
    practice_questions: List[dict] = Field(default_factory=list, description="Practice questions")
    sources: List[str] = Field(default_factory=list, description="Source references")
    recommendations: str = Field(default="", description="Learning recommendations")
    error: Optional[str] = Field(default=None, description="Error message if any")
    # Debug/RAG information
    matched_topics: List[dict] = Field(default_factory=list, description="RAG extracted topics")
    matched_pages: List[dict] = Field(default_factory=list, description="Retrieved page texts for debugging")
    
    class Config:
        # Ensure all fields are included in JSON even if they have default values
        exclude_unset = False


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

# Optional: Serve static files from frontend build
# Uncomment to serve frontend from FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pathlib import Path
# 
# frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
# if frontend_build.exists():
#     app.mount("/static", StaticFiles(directory=str(frontend_build)), name="static")
#     
#     @app.get("/{full_path:path}")
#     async def serve_frontend(full_path: str):
#         """Serve frontend for all non-API routes."""
#         if full_path.startswith("api") or full_path.startswith("docs"):
#             return  # Let FastAPI handle these
#         file_path = frontend_build / full_path
#         if file_path.exists() and file_path.is_file():
#             return FileResponse(str(file_path))
#         # Serve index.html for SPA routing
#         return FileResponse(str(frontend_build / "index.html"))


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
        
        # Debug: print what we're getting from the graph
        lecture_content = result.get('lecture_content', '')
        matched_topics = result.get('matched_topics', [])
        matched_pages = result.get('matched_pages', [])
        
        print(f"[API] lecture_content length: {len(lecture_content)}")
        print(f"[API] lecture_content preview: {lecture_content[:100] if lecture_content else 'EMPTY'}...")
        print(f"[API] matched_topics count: {len(matched_topics)}")
        if matched_topics:
            print(f"[API] matched_topics[0]: {matched_topics[0]}")
        print(f"[API] matched_pages count: {len(matched_pages)}")
        import sys
        sys.stdout.flush()
        
        # Ensure all fields are explicitly set, even if empty
        matched_topics = result.get("matched_topics")
        if matched_topics is None:
            matched_topics = []
        elif not isinstance(matched_topics, list):
            matched_topics = []
        
        matched_pages = result.get("matched_pages")
        if matched_pages is None:
            matched_pages = []
        elif not isinstance(matched_pages, list):
            matched_pages = []
        
        lecture_content = result.get("lecture_content", "")
        if lecture_content is None:
            lecture_content = ""
        
        return TutorResponse(
            lecture_content=lecture_content,
            control_questions=result.get("control_questions") or [],
            practice_questions=result.get("practice_questions") or [],
            sources=result.get("sources") or [],
            recommendations=result.get("recommendations") or "",
            error=result.get("error"),
            matched_topics=matched_topics,
            matched_pages=matched_pages
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


# === Student Endpoints ===
# Note: /students/list must come before /students/{student_id} to avoid route conflicts

@app.get("/students/list")
async def list_students(subject: Optional[str] = None, grade: Optional[int] = None):
    """Get list of available student IDs with basic information.
    
    Args:
        subject: Optional subject filter
        grade: Optional grade filter
    """
    try:
        import pandas as pd
        import traceback
        settings = get_settings()
        
        # Check if file exists
        scores_path = settings.scores_parquet_path
        if not scores_path.exists():
            error_msg = f"Parquet file not found: {scores_path}"
            print(f"❌ Error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"📂 Loading scores from: {scores_path}")
        
        # Load scores data
        try:
            scores_df = pd.read_parquet(scores_path)
            print(f"✓ Loaded {len(scores_df)} records from parquet")
        except Exception as e:
            error_msg = f"Failed to read parquet file: {str(e)}"
            print(f"❌ Error: {error_msg}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Check if required columns exist
        required_cols = ['student_id', 'score_numeric', 'discipline_name', 'grade']
        missing_cols = [col for col in required_cols if col not in scores_df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}. Available columns: {list(scores_df.columns)}"
            print(f"❌ Error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Filter by subject and grade if provided
        if subject:
            scores_df = scores_df[scores_df['discipline_name'] == subject]
            print(f"  Filtered by subject '{subject}': {len(scores_df)} records")
        if grade:
            scores_df = scores_df[scores_df['grade'] == grade]
            print(f"  Filtered by grade {grade}: {len(scores_df)} records")
        
        # Convert score_numeric to numeric if it's not already
        scores_df['score_numeric'] = pd.to_numeric(scores_df['score_numeric'], errors='coerce')
        scores_df = scores_df.dropna(subset=['score_numeric'])
        
        # Get unique students with basic stats
        students_data = []
        unique_students = scores_df['student_id'].unique()
        print(f"  Found {len(unique_students)} unique students")
        
        for student_id in unique_students:
            student_scores = scores_df[scores_df['student_id'] == student_id]
            avg_score = student_scores['score_numeric'].mean()
            total_lessons = len(student_scores)
            
            # Get subjects for this student
            subjects = student_scores['discipline_name'].unique().tolist()
            grades = student_scores['grade'].unique().tolist()
            
            students_data.append({
                "student_id": int(student_id),
                "average_score": round(float(avg_score), 2),
                "total_lessons": int(total_lessons),
                "subjects": subjects,
                "grades": [int(g) for g in grades]
            })
        
        # Sort by student_id
        students_data.sort(key=lambda x: x['student_id'])
        
        print(f"✓ Returning {len(students_data)} students")
        return {"students": students_data, "total": len(students_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error in list_students: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/students/{student_id}/info")
async def get_student_info(
    student_id: int, 
    subject: Optional[str] = None
):
    """Get detailed information about a specific student.
    
    Args:
        student_id: Student ID
        subject: Optional subject filter
    """
    try:
        import pandas as pd
        import traceback
        settings = get_settings()
        
        # Load data with error checking
        scores_path = settings.scores_parquet_path
        absences_path = settings.absences_parquet_path
        
        if not scores_path.exists():
            raise HTTPException(status_code=500, detail=f"Scores file not found: {scores_path}")
        if not absences_path.exists():
            raise HTTPException(status_code=500, detail=f"Absences file not found: {absences_path}")
        
        try:
            scores_df = pd.read_parquet(scores_path)
            absences_df = pd.read_parquet(absences_path)
        except Exception as e:
            error_msg = f"Failed to read parquet files: {str(e)}"
            print(f"❌ Error: {error_msg}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Convert score_numeric to numeric
        scores_df['score_numeric'] = pd.to_numeric(scores_df['score_numeric'], errors='coerce')
        scores_df = scores_df.dropna(subset=['score_numeric'])
        
        # Filter by student_id
        student_scores = scores_df[scores_df['student_id'] == student_id].copy()
        student_absences = absences_df[absences_df['student_id'] == student_id].copy()
        
        if student_scores.empty:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        # Filter by subject if provided
        if subject:
            student_scores = student_scores[student_scores['discipline_name'] == subject]
            student_absences = student_absences[student_absences['discipline_name'] == subject]
        
        # Calculate statistics per subject
        subjects_info = []
        for subj in student_scores['discipline_name'].unique():
            subj_scores = student_scores[student_scores['discipline_name'] == subj]
            subj_absences = student_absences[student_absences['discipline_name'] == subj]
            
            avg_score = subj_scores['score_numeric'].mean()
            min_score = subj_scores['score_numeric'].min()
            max_score = subj_scores['score_numeric'].max()
            
            # Topic breakdown
            topic_breakdown = subj_scores.groupby('topic_name')['score_numeric'].mean().round(1).to_dict()
            weak_topics = [topic for topic, score in topic_breakdown.items() if score < 6]
            strong_topics = [topic for topic, score in topic_breakdown.items() if score > 9]
            
            subjects_info.append({
                "subject": subj,
                "average_score": round(float(avg_score), 2),
                "min_score": float(min_score),
                "max_score": float(max_score),
                "total_lessons": int(len(subj_scores)),
                "total_absences": int(len(subj_absences)),
                "topic_breakdown": topic_breakdown,
                "weak_topics": weak_topics,
                "strong_topics": strong_topics
            })
        
        # Overall statistics
        overall_avg = student_scores['score_numeric'].mean()
        total_absences_count = len(student_absences)
        
        return {
            "student_id": student_id,
            "overall_average_score": round(float(overall_avg), 2),
            "total_lessons": int(len(student_scores)),
            "total_absences": total_absences_count,
            "subjects": subjects_info,
            "available_subjects": student_scores['discipline_name'].unique().tolist(),
            "available_grades": [int(g) for g in student_scores['grade'].unique()]
        }
        
    except HTTPException:
        raise
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
