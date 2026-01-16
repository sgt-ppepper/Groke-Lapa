"""Content Generator agent node."""
import json
import logging
from typing import Dict, Any, List

from ..llm.lapa import LapaLLM
from .state import TutorState

logger = logging.getLogger(__name__)

def content_generator_node(state: TutorState) -> Dict[str, Any]:
    """
    Generate lecture content and control questions based on retrieved context.
    
    Args:
        state: Current graph state
        
    Returns:
        Update for TutorState with lecture_content, control_questions, sources
    """
    logger.info("Content Generator: Starting content generation")
    
    llm = LapaLLM()
    
    # Extract state variables
    query = state.get("teacher_query", "")
    grade = state.get("grade", 9)
    subject = state.get("subject", "Українська мова")
    matched_pages = state.get("matched_pages", [])
    student_profile = state.get("student_profile")
    
    # 1. Format Context
    context_text = ""
    sources = []
    
    for page in matched_pages:
        # Handle both dict and dataclass (if passed directly)
        if isinstance(page, dict):
            book_id = page.get("book_id", "Unknown")
            page_num = page.get("page_number", "?")
            text = page.get("page_text", "")
            topic = page.get("topic_title", "")
        else:
            book_id = getattr(page, "book_id", "Unknown")
            page_num = getattr(page, "page_number", "?")
            text = getattr(page, "page_text", "")
            topic = getattr(page, "topic_title", "")
            
        ref = f"[{book_id}, стор. {page_num}]"
        context_text += f"--- Джерело: {ref}, Тема: {topic} ---\n{text}\n\n"
        
        if ref not in sources:
            sources.append(ref)
            
    if not context_text:
        logger.warning("Content Generator: No context provided")
        context_text = "Інформація з підручника відсутня."
        
    # 2. Load System Prompt
    try:
        with open("src/prompts/content_generator_system_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        system_prompt = (
            f"Ти - професійний репетитор з предмету '{subject}' для учнів {grade}-го класу. "
            "Відповідай виключно у форматі JSON з полями 'lecture_content' та 'control_questions'."
        )
    
    # 3. Construct User Prompt
    personalization_note = ""
    if student_profile:
        weak_topics = student_profile.get("weak_topics", [])
        if weak_topics:
            personalization_note = (
                f"\nУчень має прогалини в таких темах: {', '.join(weak_topics)}. "
                "Спробуй пояснити матеріал детальніше, звертаючи увагу на базові поняття."
            )
    
    user_prompt = (
        f"Запит вчителя/учня: {query}\n"
        f"Предмет: {subject}, {grade} клас.\n\n"
        f"Матеріали з підручника:\n{context_text}\n"
        f"{personalization_note}\n"
        "Створи лекцію та 3 контрольні запитання."
    )
    
    # 4. Call LLM
    try:
        # We expect JSON output directly from the model now, driven by the system prompt
        response_json = llm.generate(user_prompt, system=system_prompt, max_tokens=5000)
        
        # Clean JSON markdown if present
        cleaned_json = response_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]
        cleaned_json = cleaned_json.strip()
            
        # Use strict=False to allow control characters (newlines) inside strings
        data = json.loads(cleaned_json, strict=False)
        
        # Extract content based on structure (sometimes wrapped in 'arguments')
        if "arguments" in data:
            args = data["arguments"]
            if isinstance(args, str):
                content_data = json.loads(args, strict=False)
            else:
                content_data = args
        elif "lecture_content" in data:
            content_data = data
        else:
            # Maybe it returned the function call object wrapper
            if "function" in data and "arguments" in data["function"]: # Just in case
                 args = data["function"]["arguments"]
                 if isinstance(args, str):
                    content_data = json.loads(args, strict=False)
                 else:
                    content_data = args
            else:
                # Fallback
                logger.warning(f"Unexpected JSON structure: {cleaned_json[:100]}...")
                content_data = {"lecture_content": str(response_json), "control_questions": []}
                
        lecture = content_data.get("lecture_content", "")
        questions = content_data.get("control_questions", [])
        
    except Exception as e:
        logger.error(f"Content Generator Error: {e}")
        # Fallback to simple generation without JSON constraint
        simple_system = f"Ти - професійний репетитор з предмету '{subject}' для учнів {grade}-го класу."
        
        fallback_prompt = (
            f"Запит: {query}\n\n"
            f"Матеріали:\n{context_text}\n\n"
            "Напиши лекцію та 3 контрольні запитання. Формат: Markdown. Не використовуй JSON."
        )
        lecture = llm.generate(fallback_prompt, system=simple_system)
        questions = []
    
    logger.info("Content generation completed")
    
    return {
        "lecture_content": lecture,
        "control_questions": questions,
        "sources": sources
    }
