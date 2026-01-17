"""Personalization integration for the LangGraph workflow.

This module provides the bridge between the PersonalizationEngine
and the LangGraph workflow, handling data loading.
"""
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_settings
from ..personalization_engine import PersonalizationEngine


# Singleton instance for PersonalizationEngine
_engine_instance: Optional[PersonalizationEngine] = None


def get_personalization_engine() -> Optional[PersonalizationEngine]:
    """Get or create PersonalizationEngine instance.
    
    Loads data from parquet files on first call and caches the instance.
    Returns None if data files are not available.
    """
    global _engine_instance
    
    if _engine_instance is not None:
        return _engine_instance
    
    settings = get_settings()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    scores_path = project_root / settings.scores_parquet_path
    absences_path = project_root / settings.absences_parquet_path
    
    # Check if files exist
    if not scores_path.exists():
        print(f"[Personalization] Warning: scores file not found at {scores_path}")
        return None
    
    if not absences_path.exists():
        print(f"[Personalization] Warning: absences file not found at {absences_path}")
        return None
    
    try:
        # Load data from parquet files
        print(f"[Personalization] Loading scores from {scores_path}")
        df_scores = pd.read_parquet(scores_path)
        
        print(f"[Personalization] Loading absences from {absences_path}")
        df_absences = pd.read_parquet(absences_path)
        
        # Initialize the engine
        _engine_instance = PersonalizationEngine(df_scores, df_absences)
        
        return _engine_instance
        
    except Exception as e:
        print(f"[Personalization] Error loading data: {e}")
        return None


def get_student_personalization(
    student_id: int,
    subject: str,
    topic: str
) -> Dict[str, Any]:
    """Get personalization context for a student.
    
    This is the main function to call. It returns the full output from
    PersonalizationEngine.get_student_context() which includes:
    
    - meta: {student_id, subject, scope_type, is_fallback}
    - metrics: {average_score, min_score, max_score, grades_count}
    - enrichment: {full_topic_breakdown, weak_topics, strong_topics}
    - attendance: {missed_count, last_missed_date, missed_lessons_details}
    - prompt_injection: Ready-to-use prompt string for LLM
    
    Args:
        student_id: Student ID
        subject: Subject name (Алгебра, Українська мова, Історія України)
        topic: Current topic from router
        
    Returns:
        Full personalization context dict from PersonalizationEngine
    """
    engine = get_personalization_engine()
    
    if engine is None:
        return {
            "error": "Personalization engine not available",
            "prompt_injection": "Немає даних про учня. Використай стандартний підхід."
        }
    
    try:
        return engine.get_student_context(
            student_id=student_id,
            subject=subject,
            topic_from_router=topic
        )
        
    except Exception as e:
        print(f"[Personalization] Error getting context: {e}")
        return {
            "error": str(e),
            "prompt_injection": "Помилка отримання даних. Використай стандартний підхід."
        }