"""LangGraph workflow definition for Mriia AI Tutor.

This module defines the complete tutoring pipeline as a LangGraph graph:
    START → Topic Router → Context Retriever → Personalization → 
    Content Generator → Practice Generator → Solver/Validator → 
    (loop if invalid) → Check Answers (if provided) → Recommendations → END
"""
import ast
import math
import re
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, START, END

from .state import TutorState
from ..llm.mamay import MamayLLM
from .topic_router import TopicRouter, get_discipline_id

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
            "subject": result.get("subject")
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
    
    Uses ChromaDB with Qwen embeddings for semantic search.
    """
    # TODO: Implement with actual ChromaDB retrieval
    print(f"[Context Retriever] Finding pages for {len(state.get('matched_topics', []))} topics...")
    return state


def personalization_engine(state: TutorState) -> TutorState:
    """Load and apply student profile for personalization.
    
    Uses benchmark_scores and benchmark_absences data.
    """
    # TODO: Implement with actual student data
    student_id = state.get('student_id')
    print(f"[Personalization] Student ID: {student_id or 'None (anonymous)'}")
    return state


def content_generator(state: TutorState) -> TutorState:
    """Generate lecture content using Lapa LLM.
    
    Creates structured explanation with control questions.
    """
    # TODO: Implement with LapaLLM
    print("[Content Generator] Generating lecture content...")
    return state


def parse_practice_questions(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output into structured practice questions."""
    questions: List[Dict[str, Any]] = []
    blocks = text.split("---")

    for block in blocks:
        if not block.strip():
            continue
        try:
            q_match = re.search(
                r"ПИТАННЯ\s*\d+:\s*(.*?)(?=\n[A-D]\.)",
                block,
                re.DOTALL,
            )
            question_text = q_match.group(1).strip() if q_match else ""

            options: List[str] = []
            for letter in ["A", "B", "C", "D"]:
                opt_match = re.search(
                    rf"{letter}\.\s*(.*?)(?=\n[A-D]\.|(?:\nВІДПОВІДЬ:)|$)",
                    block,
                    re.DOTALL,
                )
                if opt_match:
                    options.append(opt_match.group(1).strip())

            ans_match = re.search(r"ВІДПОВІДЬ:\s*([A-D])", block)
            correct_letter = ans_match.group(1) if ans_match else "A"

            expl_match = re.search(r"ПОЯСНЕННЯ:\s*(.*)", block, re.DOTALL)
            explanation = expl_match.group(1).strip() if expl_match else ""

            if question_text and len(options) == 4:
                questions.append(
                    {
                        "question": question_text,
                        "options": options,
                        "correct_answer": correct_letter,
                        "explanation": explanation,
                        "is_validated": False,
                    }
                )
        except Exception as exc:
            print(f"[Practice Parser] Error parsing block: {exc}")
            continue

    return questions


def extract_python_code(text: str) -> str:
    """Extract python code from a fenced block or return raw text."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _is_allowed_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in {"abs", "pow", "round", "min", "max"}
    if isinstance(node.func, ast.Attribute):
        return isinstance(node.func.value, ast.Name) and node.func.value.id == "math"
    return False


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.Raise,
                ast.Lambda,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Global,
                ast.Nonlocal,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise ValueError("Disallowed Python construct in validator code.")
        if isinstance(node, ast.Call) and not _is_allowed_call(node):
            raise ValueError("Disallowed function call in validator code.")


def run_validator_code(code: str) -> Optional[int]:
    """Execute limited python code and return ANSWER_INDEX if present."""
    if not code:
        return None
    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)

    safe_builtins = {"abs": abs, "pow": pow, "round": round, "min": min, "max": max}
    globals_dict = {"__builtins__": safe_builtins, "math": math}
    locals_dict: Dict[str, Any] = {}
    exec(compile(tree, filename="<validator>", mode="exec"), globals_dict, locals_dict)

    answer_index = locals_dict.get("ANSWER_INDEX")
    if isinstance(answer_index, int) and 0 <= answer_index <= 3:
        return answer_index
    answer_letter = locals_dict.get("ANSWER_LETTER")
    if isinstance(answer_letter, str) and answer_letter.upper() in {"A", "B", "C", "D"}:
        return ord(answer_letter.upper()) - ord("A")
    return None


def _topic_to_text(topic: Any) -> str:
    if isinstance(topic, str):
        return topic
    if isinstance(topic, dict):
        return (
            topic.get("topic_title")
            or topic.get("section_title")
            or topic.get("topic")
            or topic.get("title")
            or str(topic)
        )
    return str(topic)


def _build_validation_feedback(state: TutorState) -> str:
    feedback = state.get("validation_feedback")
    if feedback:
        return feedback

    results = state.get("validation_results") or []
    invalids = [r for r in results if not r.get("is_valid")]
    if not invalids:
        return ""

    parts: List[str] = []
    for r in invalids[:5]:
        idx = r.get("question_index")
        expected = r.get("expected_index")
        validator = r.get("validator_index")
        error = r.get("error")
        if isinstance(idx, int):
            q_num = idx + 1
        else:
            q_num = "?"
        if error:
            parts.append(f"Питання {q_num}: помилка перевірки ({error})")
        else:
            exp_letter = chr(65 + expected) if isinstance(expected, int) else "?"
            val_letter = chr(65 + validator) if isinstance(validator, int) else "?"
            parts.append(
                f"Питання {q_num}: очікувана відповідь {exp_letter}, перевірка дала {val_letter}"
            )

    if len(invalids) > 5:
        parts.append("Є й інші помилки, виправ загалом.")
    return "\n".join(parts)


def practice_generator(state: TutorState) -> TutorState:
    """Generate practice questions using MamayLM.
    
    Creates 8-12 questions in various formats.
    """
    print("[Practice Generator] Generating practice questions...")
    mamay = MamayLLM()

    explicit_topic = state.get("topic")
    explicit_subtopics = state.get("subtopics")
    matched_topics = state.get("matched_topics", [])
    normalized_topics = [_topic_to_text(t) for t in matched_topics]
    main_topic = (
        explicit_topic
        or (normalized_topics[0] if normalized_topics else None)
        or state.get("teacher_query", "Загальна тема")
    )
    subtopics = (
        explicit_subtopics
        if explicit_subtopics is not None
        else (normalized_topics[1:] if len(normalized_topics) > 1 else [])
    )

    grade = state.get("grade", 9)
    subject = state.get("subject", "Українська мова")
    student_level = state.get("student_level")
    practice_recommendations = state.get("practice_recommendations")
    practice_count = state.get("practice_count", 8)

    lecture_content = state.get("lecture_content", "")
    try:
        subtopics_text = ", ".join(subtopics) if subtopics else "загальні аспекти теми"
        level_text = student_level or "відповідно до класу"
        recommendations_text = practice_recommendations or "Немає додаткових рекомендацій."
        validator_feedback = _build_validation_feedback(state)

        prompt = f"""Створи {practice_count} тестових питань для учнів {grade} класу з предмету "{subject}".

Тема: {main_topic}
Підтеми: {subtopics_text}
Рівень учня: {level_text}
РЕКОМЕНДАЦІЇ ДО ПРАКТИКИ: {recommendations_text}

Вимоги до якості:
- Кожне питання має ТІЛЬКИ одну правильну відповідь.
- 4 варіанти відповіді (A, B, C, D).
- Не повторюй питання або формулювання.
"""

        if lecture_content:
            prompt += f"""
Використовуй ТІЛЬКИ цей конспект як джерело фактів:
\"\"\"
{lecture_content}
\"\"\"
"""

        if validator_feedback:
            prompt += f"""
ПОПЕРЕДНІ ЗАУВАЖЕННЯ ВАЛІДАТОРА:
{validator_feedback}
Виправ помилки і перегенеруй питання.
"""

        prompt += """
Для кожного питання надай:
1. Текст питання
2. 4 варіанти відповіді (A, B, C, D)
3. Правильну відповідь
4. Коротке пояснення

Формат кожного питання:
---
ПИТАННЯ [номер]:
[текст питання]

A. [варіант A]
B. [варіант B]
C. [варіант C]
D. [варіант D]

ВІДПОВІДЬ: [літера]
ПОЯСНЕННЯ: [коротке пояснення]
---"""

        raw_practice = mamay.generate(prompt, temperature=0.8, max_tokens=4000)

        parsed_questions = parse_practice_questions(raw_practice)
        print(f"[Practice Generator] Parsed {len(parsed_questions)} questions")

        return {
            **state,
            "practice_questions": parsed_questions,
            "regeneration_count": state.get("regeneration_count", 0) + 1,
        }
    except Exception as exc:
        print(f"[Practice Generator] Error: {exc}")
        return {**state, "error": str(exc)}


def solver_validator(state: TutorState) -> TutorState:
    """Validate generated questions by solving them.
    
    Self-correction loop: if answer doesn't match key, regenerate.
    """
    print("[Solver/Validator] Validating questions...")
    subject = (state.get("subject") or "").lower()
    questions = state.get("practice_questions", [])

    # Only run code-based validation for Algebra.
    if "алгебра" not in subject:
        return {**state, "validation_passed": True}

    if not questions:
        return {**state, "validation_passed": False}

    mamay = MamayLLM()
    validation_results: List[Dict[str, Any]] = []
    updated_questions: List[Dict[str, Any]] = []
    all_valid = True

    for idx, q in enumerate(questions):
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_letter = (q.get("correct_answer") or "A").upper()
        expected_index = ord(correct_letter) - ord("A") if correct_letter in "ABCD" else None

        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"""Ти перевіряєш тестове питання з алгебри.
Згенеруй короткий Python-код, який обчислює правильний варіант відповіді.

Вимоги:
- Поверни ТІЛЬКИ Python-код (без пояснень).
- Не використовуй імпорти.
- Присвой результат у змінну ANSWER_INDEX (0..3) або ANSWER_LETTER ("A".."D").

Питання:
{question_text}

Варіанти:
{options_text}
"""
        response = mamay.generate(prompt, temperature=0.0, max_tokens=300)
        code = extract_python_code(response)
        answer_index = None
        error = None
        try:
            answer_index = run_validator_code(code)
        except Exception as exc:
            error = str(exc)

        is_valid = (
            answer_index is not None
            and expected_index is not None
            and answer_index == expected_index
        )
        if not is_valid:
            all_valid = False

        updated_q = dict(q)
        updated_q["is_validated"] = is_valid
        updated_q["validator_answer_index"] = answer_index
        updated_q["validator_answer_letter"] = (
            chr(65 + answer_index) if isinstance(answer_index, int) else None
        )
        updated_questions.append(updated_q)

        validation_results.append(
            {
                "question_index": idx,
                "expected_index": expected_index,
                "validator_index": answer_index,
                "is_valid": is_valid,
                "error": error,
            }
        )

    return {
        **state,
        "practice_questions": updated_questions,
        "validation_results": validation_results,
        "validation_passed": all_valid,
    }


def check_answers(state: TutorState) -> TutorState:
    """Check student answers and provide feedback.
    
    Evaluates each answer and generates explanations.
    """
    # TODO: Implement answer checking
    print("[Check Answers] Evaluating student responses...")
    return state


def recommendations_generator(state: TutorState) -> TutorState:
    """Generate learning recommendations using Lapa LLM.
    
    Creates personalized next steps based on performance.
    """
    # TODO: Implement with LapaLLM
    print("[Recommendations] Generating learning recommendations...")
    return state


def response_finalizer(state: TutorState) -> TutorState:
    """Finalize and format the response.
    
    Compiles all generated content into final output.
    """
    print("[Finalizer] Preparing final response...")
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


# Compiled graph instance
tutor_graph = create_tutor_graph()
