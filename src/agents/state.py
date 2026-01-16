"""Shared state schema for LangGraph workflow."""
from typing import TypedDict, Optional, List, Literal, Any
from dataclasses import dataclass, field


@dataclass
class Topic:
    """Matched topic from TOC."""
    book_id: str
    topic_title: str
    section_title: str
    topic_text: str
    subtopics: List[str]
    grade: int
    subject: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@dataclass
class Page:
    """Retrieved page from knowledge base."""
    book_id: str
    page_number: int
    page_text: str
    topic_title: str
    section_title: str
    subject: str
    grade: int


@dataclass
class Question:
    """Generated or benchmark question."""
    question_id: str
    question_text: str
    answers: List[str]
    correct_answer_index: int
    subject: str
    grade: int
    topic: str
    difficulty: str = "середня"
    question_type: str = "single_choice"


@dataclass
class StudentProfile:
    """Student's performance profile."""
    student_id: int
    average_score: float
    weak_topics: List[str]
    missed_topics: List[str]
    total_lessons: int
    attended_lessons: int


@dataclass 
class EvalResult:
    """Result of evaluating a student answer."""
    question_id: str
    is_correct: bool
    student_answer: str
    correct_answer: str
    explanation: str


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
    matched_topics: List[dict]  # List of Topic as dict
    
    # === Context Retrieval ===
    matched_pages: List[dict]  # List of Page as dict
    
    # === Student Personalization ===
    student_profile: Optional[dict]  # StudentProfile as dict
    
    # === Generated Content ===
    lecture_content: str
    control_questions: List[str]
    
    # === Practice Generation ===
    practice_questions: List[dict]  # List of Question as dict
    
    # === Validation (Self-check) ===
    validation_results: List[dict]
    validation_passed: bool
    regeneration_count: int
    
    # === Student Answers (if provided) ===
    student_answers: Optional[List[str]]
    evaluation_results: Optional[List[dict]]  # List of EvalResult as dict
    
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
        matched_pages=[],
        student_profile=None,
        lecture_content="",
        control_questions=[],
        practice_questions=[],
        validation_results=[],
        validation_passed=False,
        regeneration_count=0,
        student_answers=None,
        evaluation_results=None,
        recommendations="",
        next_topics=[],
        sources=[],
        error=None
    )
