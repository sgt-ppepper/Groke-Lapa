"""LangGraph workflow definition for Mriia AI Tutor.

This module defines the complete tutoring pipeline as a LangGraph graph:
    START → Topic Router → Context Retriever → Personalization → 
    Content Generator → Practice Generator → Solver/Validator → 
    (loop if invalid) → Check Answers (if provided) → Recommendations → END
"""
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END

from .state import TutorState
from .topic_router import TopicRouter, get_discipline_id
from .content_generator import generate_content as _generate_content_impl
from .personalization import get_student_personalization
from .practice_generator import (
    generate_practice as _generate_practice_impl,
    validate_questions as _validate_questions_impl,
    topic_to_text,
    build_validation_feedback,
)

# Initialize TopicRouter instance (singleton pattern)
_router_instance: TopicRouter = None


def get_topic_router() -> TopicRouter:
    """Get or create TopicRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = TopicRouter()
    return _router_instance


# === Node Functions (Placeholder implementations) ===
# These will be replaced with actual implementations from separate modules

def topic_router(state: TutorState) -> TutorState:
    """Route teacher query to relevant topics from TOC.
    
    Uses MamayLM or embedding similarity to find matching topics.
    Grade and subject are inferred from the query if not provided.
    """
    try:
        router = get_topic_router()
        query = state.get("teacher_query", "")
        
        # Get grade and subject from state if available, otherwise infer
        grade = state.get("grade")
        subject = state.get("subject")
        discipline_id = None
        
        if subject:
            # Get discipline ID from subject name if provided
            discipline_id = get_discipline_id(subject)
        
        # Route the query - will infer grade/subject if not provided
        result = router.route(
            query=query,
            grade=grade,
            discipline_id=discipline_id,
            top_k=5
        )
        
        # Update state with inferred values if they were missing
        if not state.get("grade") and result.get("grade"):
            state["grade"] = result["grade"]
        if not state.get("subject") and result.get("subject"):
            state["subject"] = result["subject"]
        
        # Store results in state
        state["matched_topics"] = [{
            "topic": result["topic"],
            "retrieved_docs": result["retrieved_docs"],
            "grade": result.get("grade"),
            "subject": result.get("subject"),
            "source_info": result.get("source_info", {})
        }]
        
        print(f"[Topic Router] Matched topic: {result['topic']}")
        print(f"[Topic Router] Inferred grade: {result.get('grade')}, subject: {result.get('subject')}")
        print(f"[Topic Router] Retrieved {len(result['retrieved_docs'])} documents")
        
    except Exception as e:
        print(f"[Topic Router] Error: {e}")
        state["error"] = f"Topic routing failed: {str(e)}"
        state["matched_topics"] = []
    
    return state


def context_retriever(state: TutorState) -> TutorState:
    """Retrieve relevant pages from knowledge base.
    
    Note: Context retrieval is handled by topic_router which returns retrieved_docs.
    This node is kept for potential future enhancements (multi-stage retrieval).
    """
    print(f"[Context Retriever] {len(state.get('matched_topics', []))} topics available")
    return state


def personalization_engine(state: TutorState) -> TutorState:
    """Load and apply student profile for personalization.
    
    Uses benchmark_scores and benchmark_absences data via PersonalizationEngine.
    Stores the full context in student_profile - use prompt_injection for LLM.
    """
    student_id = state.get("student_id")
    
    if student_id is None:
        print("[Personalization] No student_id provided, skipping")
        state["student_profile"] = None
        return state
    
    # Get subject from state
    subject = state.get("subject", "Українська мова")
    
    # Extract topic from matched_topics
    topic = ""
    matched_topics = state.get("matched_topics", [])
    if matched_topics:
        first_match = matched_topics[0]
        if isinstance(first_match, dict):
            topic = first_match.get("topic", "")
        elif isinstance(first_match, str):
            topic = first_match
    
    if not topic:
        topic = state.get("teacher_query", "Загальна тема")
    
    # Get personalization context - returns full dict with prompt_injection
    context = get_student_personalization(
        student_id=student_id,
        subject=subject,
        topic=topic
    )
    
    state["student_profile"] = context
    
    # Log key info (with safe access)
    metrics = context.get("metrics") if isinstance(context, dict) else None
    if metrics:
        avg_score = metrics.get("average_score", 0)
        print(f"[Personalization] Student {student_id}: avg score {avg_score:.1f}/12")
    else:
        print(f"[Personalization] Student {student_id}: no metrics available")
    
    return state


def content_generator(state: TutorState) -> TutorState:
    """Generate lecture content using Lapa LLM.
    
    Creates structured explanation with control questions.
    Uses the ContentGenerator from content_generator.py module.
    """
    result = _generate_content_impl(state)
    
    state["lecture_content"] = result["lecture_content"]
    state["control_questions"] = result["control_questions"]
    state["sources"] = result["sources"]
    
    if result.get("error"):
        state["error"] = result["error"]
    
    return state


def practice_generator(state: TutorState) -> TutorState:
    """Generate practice questions using MamayLLM.
    
    Creates 8-12 questions in various formats.
    Uses the practice_generator module.
    """
    matched_topics = state.get("matched_topics", [])
    normalized_topics = [topic_to_text(t) for t in matched_topics]
    
    # Get topic
    explicit_topic = state.get("topic")
    main_topic = (
        explicit_topic
        or (normalized_topics[0] if normalized_topics else None)
        or state.get("teacher_query", "Загальна тема")
    )
    
    # Get subtopics
    explicit_subtopics = state.get("subtopics")
    subtopics = (
        explicit_subtopics
        if explicit_subtopics is not None
        else (normalized_topics[1:] if len(normalized_topics) > 1 else [])
    )
    
    # Get personalization prompt
    student_profile = state.get("student_profile")
    personalization_prompt = None
    if student_profile and isinstance(student_profile, dict):
        personalization_prompt = student_profile.get("prompt_injection")
    
    # Build validation feedback if regenerating
    validator_feedback = build_validation_feedback(state)
    
    # Generate questions
    result = _generate_practice_impl(
        topic=main_topic,
        grade=state.get("grade", 9),
        subject=state.get("subject", "Українська мова"),
        lecture_content=state.get("lecture_content", ""),
        subtopics=subtopics,
        personalization_prompt=personalization_prompt,
        practice_count=state.get("practice_count", 8),
        validator_feedback=validator_feedback
    )
    
    state["practice_questions"] = result["practice_questions"]
    state["regeneration_count"] = state.get("regeneration_count", 0) + 1
    
    if result.get("error"):
        state["error"] = result["error"]
    
    return state


def solver_validator(state: TutorState) -> TutorState:
    """Validate generated questions by solving them.
    
    Self-correction loop: if answer doesn't match key, regenerate.
    Uses the practice_generator module for validation.
    """
    questions = state.get("practice_questions", [])
    subject = state.get("subject", "")
    
    result = _validate_questions_impl(questions, subject)
    
    state["practice_questions"] = result["questions"]
    state["validation_results"] = result["validation_results"]
    state["validation_passed"] = result["all_valid"]
    
    return state


def check_answers(state: TutorState) -> TutorState:
    """Check student answers and provide feedback.
    
    Compares student_answers with practice_questions[i]["correct_answer"]
    and generates evaluation_results.
    """
    print("[Check Answers] Evaluating student responses...")
    
    student_answers = state.get("student_answers")
    practice_questions = state.get("practice_questions", [])
    
    # No answers to check
    if not student_answers:
        print("[Check Answers] No student answers provided")
        state["evaluation_results"] = []
        return state
    
    evaluation_results = []
    correct_count = 0
    
    for i, question in enumerate(practice_questions):
        # Get student answer (handle index out of range)
        student_answer = student_answers[i] if i < len(student_answers) else None
        
        # Get correct answer from question
        correct_answer = question.get("correct_answer", "")
        
        # Compare (case-insensitive)
        is_correct = False
        if student_answer and correct_answer:
            is_correct = student_answer.upper().strip() == correct_answer.upper().strip()
        
        if is_correct:
            correct_count += 1
        
        # Build evaluation result
        result = {
            "question_index": i,
            "question_text": question.get("question", "")[:100],  # Truncate for readability
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.get("explanation", "") if not is_correct else ""
        }
        evaluation_results.append(result)
    
    state["evaluation_results"] = evaluation_results
    
    # Log summary
    total = len(practice_questions)
    if total > 0:
        score_pct = (correct_count / total) * 100
        print(f"[Check Answers] Score: {correct_count}/{total} ({score_pct:.0f}%)")
    
    return state


def recommendations_generator(state: TutorState) -> TutorState:
    """Generate learning recommendations based on performance.
    
    Analyzes evaluation_results and student_profile to provide:
    - Personalized recommendations text
    - Suggested next topics
    """
    print("[Recommendations] Generating learning recommendations...")
    
    evaluation_results = state.get("evaluation_results", [])
    student_profile = state.get("student_profile")
    matched_topics = state.get("matched_topics", [])
    
    # If no evaluation results, skip
    if not evaluation_results:
        print("[Recommendations] No evaluation results, skipping")
        state["recommendations"] = ""
        state["next_topics"] = []
        return state
    
    # Calculate score
    total = len(evaluation_results)
    correct = sum(1 for r in evaluation_results if r.get("is_correct"))
    score_pct = (correct / total * 100) if total > 0 else 0
    
    # Identify weak areas (wrong questions)
    wrong_questions = [r for r in evaluation_results if not r.get("is_correct")]
    
    # Build recommendations
    rec_parts = []
    
    # Score-based feedback
    if score_pct >= 90:
        rec_parts.append(f"🎉 Відмінний результат: {correct}/{total} ({score_pct:.0f}%)!")
        rec_parts.append("Ти чудово опанував цю тему. Можеш переходити до наступної.")
    elif score_pct >= 70:
        rec_parts.append(f"👍 Добрий результат: {correct}/{total} ({score_pct:.0f}%).")
        rec_parts.append("Рекомендую повторити деякі моменти перед наступною темою.")
    elif score_pct >= 50:
        rec_parts.append(f"📚 Середній результат: {correct}/{total} ({score_pct:.0f}%).")
        rec_parts.append("Потрібно більше практики з цієї теми.")
    else:
        rec_parts.append(f"⚠️ Потребує уваги: {correct}/{total} ({score_pct:.0f}%).")
        rec_parts.append("Рекомендую перечитати матеріал та спробувати знову.")
    
    # Add info about wrong answers
    if wrong_questions:
        rec_parts.append("")
        rec_parts.append(f"❌ Помилки у {len(wrong_questions)} питаннях:")
        for i, q in enumerate(wrong_questions[:3], 1):  # Show first 3
            q_text = q.get("question_text", "")[:50]
            rec_parts.append(f"   {i}. {q_text}...")
        if len(wrong_questions) > 3:
            rec_parts.append(f"   ... та ще {len(wrong_questions) - 3}")
    
    # Add personalization-based tips
    if student_profile and isinstance(student_profile, dict):
        enrichment = student_profile.get("enrichment", {})
        weak_topics = enrichment.get("weak_topics", [])
        if weak_topics:
            rec_parts.append("")
            rec_parts.append(f"📌 Також зверни увагу на: {', '.join(weak_topics[:2])}")
    
    state["recommendations"] = "\n".join(rec_parts)
    
    # Suggest next topics (from matched_topics if available)
    next_topics = []
    for topic in matched_topics[1:4]:  # Skip current, take next 3
        if isinstance(topic, dict):
            topic_name = topic.get("topic", "") or topic.get("topic_title", "")
        else:
            topic_name = str(topic)
        if topic_name:
            next_topics.append(topic_name)
    
    state["next_topics"] = next_topics
    
    print(f"[Recommendations] Score: {score_pct:.0f}%, {len(next_topics)} next topics suggested")
    
    return state


def response_finalizer(state: TutorState) -> TutorState:
    """Finalize and format the response.
    
    Compiles all generated content into final output:
    - Ensures all required fields are present
    - Formats sources for grounding
    - Logs summary of generated content
    """
    print("[Finalizer] Preparing final response...")
    
    # Ensure required fields have defaults
    if "lecture_content" not in state or state["lecture_content"] is None:
        state["lecture_content"] = ""
    
    if "control_questions" not in state or state["control_questions"] is None:
        state["control_questions"] = []
    
    if "practice_questions" not in state or state["practice_questions"] is None:
        state["practice_questions"] = []
    
    if "sources" not in state or state["sources"] is None:
        state["sources"] = []
    
    if "recommendations" not in state or state["recommendations"] is None:
        state["recommendations"] = ""
    
    # Format sources for display
    formatted_sources = []
    
    # 1. Primary Source from Topic Router (Best Quality)
    matched_topics = state.get("matched_topics", [])
    has_primary = False
    
    if matched_topics and isinstance(matched_topics[0], dict):
        source_info = matched_topics[0].get("source_info")
        if source_info:
            subject = source_info.get("subject", "Підручник")
            grade = source_info.get("grade", "")
            topic = source_info.get("topic_title", "")
            start = source_info.get("start_page")
            end = source_info.get("end_page")
            
            # Construct clear citation
            # e.g. "📘 Алгебра 9 клас: § 11. Функція... (стор. 75-82)"
            book_ref = f"{subject}"
            if grade:
                book_ref += f" {grade} клас"
                
            citation = f"📘 {book_ref}"
            if topic:
                citation += f": {topic}"
            
            if start:
                p_range = f"{start}-{end}" if (end and end != start) else str(start)
                citation += f" (стор. {p_range})"
            
            formatted_sources.append(citation)
            has_primary = True
    
    # 2. Add specific page references from content generation
    # Only act if they look like specific page citations, ignore messy text previews
    existing_sources = state.get("sources", [])
    
    for i, source in enumerate(existing_sources, 1):
        if isinstance(source, str):
            # Clean up formatting
            clean_source = source.strip()
            
            # Check for page references: "Сторінка 75" or "Page 75"
            if "сторінка" in clean_source.lower() or "page" in clean_source.lower():
                # Extract just the page info if possible
                if clean_source.lower().startswith("сторінка") or clean_source.lower().startswith("page"):
                    formatted_sources.append(f"   • {clean_source}")
                elif ":" in clean_source:
                    # "Джерело 1: Сторінка 5" -> "Сторінка 5"
                    parts = clean_source.split(":", 1)
                    if "сторінка" in parts[1].lower():
                        formatted_sources.append(f"   • {parts[1].strip()}")
            
            # If we don't have a primary source, allow generic ones but clean them
            elif not has_primary and len(formatted_sources) < 3:
                # Truncate long messy text
                if len(clean_source) > 60:
                     clean_source = clean_source[:57] + "..."
                formatted_sources.append(f"📄 {clean_source}")
                
        elif isinstance(source, dict):
            # Dict format (if used)
            page = source.get("page_number")
            topic = source.get("topic_title")
            if page:
                 formatted_sources.append(f"   • Сторінка {page}" + (f" ({topic})" if topic else ""))

    # Deduplicate while preserving order
    seen = set()
    unique_sources = []
    for s in formatted_sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)
            
    state["sources"] = unique_sources
    
    # Log summary
    lecture_len = len(state.get("lecture_content", ""))
    control_count = len(state.get("control_questions", []))
    practice_count = len(state.get("practice_questions", []))
    source_count = len(state.get("sources", []))
    
    print(f"[Finalizer] ✓ Lecture: {lecture_len} chars")
    print(f"[Finalizer] ✓ Control questions: {control_count}")
    print(f"[Finalizer] ✓ Practice questions: {practice_count}")
    print(f"[Finalizer] ✓ Sources: {source_count}")
    
    if state.get("error"):
        print(f"[Finalizer] ⚠ Error: {state['error']}")
    
    if state.get("student_profile"):
        metrics = state["student_profile"].get("metrics", {})
        avg = metrics.get("average_score", 0)
        print(f"[Finalizer] ✓ Personalized for student (avg: {avg:.1f}/12)")
    
    return state


# === Conditional Edges ===

def should_regenerate(state: TutorState) -> Literal["practice_generator", "check_or_finalize"]:
    """Check if questions need regeneration due to validation failure."""
    if not state.get("validation_passed", False):
        regen_count = state.get("regeneration_count", 0)
        if regen_count < 3:  # Max 3 regeneration attempts
            return "practice_generator"
    return "check_or_finalize"


def has_student_answers(state: TutorState) -> Literal["check_answers", "finalizer"]:
    """Check if student answers were provided."""
    if state.get("student_answers"):
        return "check_answers"
    return "finalizer"


# === Graph Builder ===

def create_tutor_graph() -> StateGraph:
    """Create the LangGraph workflow for the tutor.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph with state schema
    graph = StateGraph(TutorState)
    
    # Add nodes
    graph.add_node("topic_router", topic_router)
    graph.add_node("context_retriever", context_retriever)
    graph.add_node("personalization", personalization_engine)
    graph.add_node("content_generator", content_generator)
    graph.add_node("practice_generator", practice_generator)
    graph.add_node("solver_validator", solver_validator)
    graph.add_node("check_answers", check_answers)
    graph.add_node("recommendations", recommendations_generator)
    graph.add_node("finalizer", response_finalizer)
    
    # Define edges (linear flow with conditional branches)
    graph.add_edge(START, "topic_router")
    graph.add_edge("topic_router", "context_retriever")
    graph.add_edge("context_retriever", "personalization")
    graph.add_edge("personalization", "content_generator")
    graph.add_edge("content_generator", "practice_generator")
    graph.add_edge("practice_generator", "solver_validator")
    
    # Conditional: regenerate if validation fails
    graph.add_conditional_edges(
        "solver_validator",
        should_regenerate,
        {
            "practice_generator": "practice_generator",
            "check_or_finalize": "check_or_finalize_router"
        }
    )
    
    # Router node for check/finalize decision
    graph.add_node("check_or_finalize_router", lambda s: s)
    graph.add_conditional_edges(
        "check_or_finalize_router",
        has_student_answers,
        {
            "check_answers": "check_answers",
            "finalizer": "finalizer"
        }
    )
    
    graph.add_edge("check_answers", "recommendations")
    graph.add_edge("recommendations", "finalizer")
    graph.add_edge("finalizer", END)
    
    return graph.compile()
