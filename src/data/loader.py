"""Data loaders for parquet files."""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from ..config import get_settings


@lru_cache(maxsize=1)
def load_pages(use_qwen: bool = True) -> pd.DataFrame:
    """Load pages parquet with text and embeddings.
    
    Args:
        use_qwen: If True, use Qwen embeddings, else use Gemini
        
    Returns:
        DataFrame with 1318 pages
    """
    settings = get_settings()
    
    if use_qwen:
        path = Path(settings.data_dir) / "text-embedding-qwen" / "pages_for_hackathon.parquet"
    else:
        path = Path(settings.data_dir) / "gemini-embedding-001" / "pages_for_hackathon.parquet"
    
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_toc(use_qwen: bool = True) -> pd.DataFrame:
    """Load TOC parquet with topics/subtopics and embeddings.
    
    Args:
        use_qwen: If True, use Qwen embeddings, else use Gemini
        
    Returns:
        DataFrame with 237 topics
    """
    settings = get_settings()
    
    if use_qwen:
        path = Path(settings.data_dir) / "text-embedding-qwen" / "toc_for_hackathon_with_subtopics.parquet"
    else:
        path = Path(settings.data_dir) / "gemini-embedding-001" / "toc_for_hackathon_with_subtopics.parquet"
    
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_questions() -> pd.DataFrame:
    """Load benchmark questions parquet.
    
    Returns:
        DataFrame with 141 questions
    """
    settings = get_settings()
    path = Path(settings.data_dir) / "lms_questions_dev.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_scores() -> pd.DataFrame:
    """Load student scores parquet.
    
    Returns:
        DataFrame with ~1M scores
    """
    settings = get_settings()
    path = Path(settings.data_dir) / "benchmark_scores.parquet"
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_absences() -> pd.DataFrame:
    """Load student absences parquet.
    
    Returns:
        DataFrame with ~298K absences
    """
    settings = get_settings()
    path = Path(settings.data_dir) / "benchmark_absences.parquet"
    return pd.read_parquet(path)


def get_student_scores(student_id: int) -> pd.DataFrame:
    """Get scores for a specific student.
    
    Args:
        student_id: The student ID
        
    Returns:
        DataFrame filtered to student
    """
    scores = load_scores()
    return scores[scores["student_id"] == student_id]


def get_student_absences(student_id: int) -> pd.DataFrame:
    """Get absences for a specific student.
    
    Args:
        student_id: The student ID
        
    Returns:
        DataFrame filtered to student
    """
    absences = load_absences()
    return absences[absences["student_id"] == student_id]


def get_subject_id(subject_name: str) -> int:
    """Convert subject name to ID.
    
    Args:
        subject_name: Subject name in Ukrainian
        
    Returns:
        Subject ID (72=Алгебра, 107=Історія України, 131=Українська мова)
    """
    mapping = {
        "Алгебра": 72,
        "Історія України": 107,
        "Українська мова": 131
    }
    return mapping.get(subject_name, 0)


def get_subject_name(subject_id: int) -> str:
    """Convert subject ID to name.
    
    Args:
        subject_id: Subject ID
        
    Returns:
        Subject name in Ukrainian
    """
    mapping = {
        72: "Алгебра",
        107: "Історія України",
        131: "Українська мова"
    }
    return mapping.get(subject_id, "Unknown")
