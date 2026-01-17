"""Shared state schema for LangGraph workflow."""
from typing import TypedDict, Optional, List, Literal


class TutorState(TypedDict, total=False):
    """Shared state passed through the LangGraph workflow.
    
    This represents the complete state at any point in the tutoring pipeline:
    Teacher Query → Topic Routing → Context → Content → Practice → Validation → Recommendations
    """
    
    # === Input ===
    teacher_query: str
    student_id: Optional[int]
    mode: Literal["demo", "benchmark", "practice"]
    grade: int  # 8 or 9
    subject: str  # Українська мова, Алгебра, Історія України
    
    # === Topic Routing ===
    # Each topic dict contains: topic, retrieved_docs, grade, subject
    matched_topics: List[dict]
    
    # === Student Personalization ===
    # Full PersonalizationEngine output: meta, metrics, enrichment, attendance, prompt_injection
    student_profile: Optional[dict]
    
    # === Generated Content ===
    lecture_content: str
    control_questions: List[str]
    
    # === Practice Generation ===
    topic: Optional[str]  # Override for explicit topic
    subtopics: Optional[List[str]]  # Override for explicit subtopics
    practice_count: Optional[int]  # Number of questions to generate
    practice_questions: List[dict]  # Each has: question, options, correct_answer, explanation
    
    # === Validation (Self-check) ===
    validation_results: List[dict]
    validation_passed: bool
    validation_feedback: Optional[str]
    regeneration_count: int
    
    # === Student Answers (if provided) ===
    student_answers: Optional[List[str]]  # ["A", "B", "C", ...]
    evaluation_results: Optional[List[dict]]  # is_correct, explanation, etc.
    
    # === Recommendations ===
    recommendations: str
    next_topics: List[str]
    
    # === Metadata & Grounding ===
    sources: List[str]  # Page references for grounding
    error: Optional[str]


def create_initial_state(
    teacher_query: str,
    grade: int = 9,
    subject: str = "Українська мова",
    student_id: Optional[int] = None,
    mode: Literal["demo", "benchmark", "practice"] = "demo"
) -> TutorState:
    """Create initial state for the workflow.
    
    Args:
        teacher_query: The teacher's request
        grade: Grade level (8 or 9)
        subject: Subject name
        student_id: Optional student ID for personalization
        mode: Workflow mode
        
    Returns:
        Initial TutorState
    """
    return TutorState(
        teacher_query=teacher_query,
        student_id=student_id,
        mode=mode,
        grade=grade,
        subject=subject,
        matched_topics=[],
        student_profile=None,
        lecture_content="",
        control_questions=[],
        topic=None,
        subtopics=None,
        practice_count=None,
        practice_questions=[],
        validation_results=[],
        validation_passed=False,
        validation_feedback=None,
        regeneration_count=0,
        student_answers=None,
        evaluation_results=None,
        recommendations="",
        next_topics=[],
        sources=[],
        error=None
    )
