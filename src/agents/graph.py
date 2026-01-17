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
            query=query, grade=grade, discipline_id=discipline_id, top_k=10
        )

        # Update state with inferred values if they were missing
        if not state.get("grade") and result.get("grade"):
            state["grade"] = result["grade"]
        if not state.get("subject") and result.get("subject"):
            state["subject"] = result["subject"]

        # Store results in state - now we have multiple topics
        topics_list = result.get("topics", [])
        if not topics_list:
            # Fallback: create single topic from old format if needed
            if result.get("topic"):
                topics_list = [{
                    "topic": result["topic"],
                    "retrieved_docs": result.get("retrieved_docs", []),
                    "grade": result.get("grade"),
                    "subject": result.get("subject"),
                    "discipline_id": result.get("discipline_id"),
                    "book_topic_id": result.get("book_topic_id")
                }]
        
        state["matched_topics"] = topics_list

        print(f"[Topic Router] Matched {len(topics_list)} topics:")
        for i, topic in enumerate(topics_list[:5], 1):
            print(f"  {i}. {topic.get('topic', 'Невідома тема')}")
        print(
            f"[Topic Router] Inferred grade: {result.get('grade')}, subject: {result.get('subject')}"
        )

        # Return new dict to ensure LangGraph properly merges the state
        return {
            **state,
            "matched_topics": state.get("matched_topics", []),
            "grade": state.get("grade"),
            "subject": state.get("subject"),
        }

    except ValueError as e:
        # ChromaDB collection not found - continue with fallback
        error_msg = str(e)
        if "ChromaDB collection" in error_msg:
            print(f"[Topic Router] Warning: {error_msg}")
            print(
                "[Topic Router] Continuing without topic routing - using query directly"
            )
            # Use query as topic fallback
            query = state.get("teacher_query", "")
            matched_topics = [
                {
                    "topic": query if query else "Загальна тема",
                    "retrieved_docs": [],
                    "grade": state.get("grade"),
                    "subject": state.get("subject"),
                }
            ]
            error_msg = "ChromaDB not initialized - topic routing disabled. Please run: python scripts/setup/setup_chroma_toc.py"
        else:
            print(f"[Topic Router] Error: {e}")
            error_msg = f"Topic routing failed: {str(e)}"
            matched_topics = []

        # Return new dict to ensure LangGraph properly merges the state
        return {**state, "matched_topics": matched_topics, "error": error_msg}
    except Exception as e:
        print(f"[Topic Router] Error: {e}")
        error_msg = f"Topic routing failed: {str(e)}"
        matched_topics = []

        # Return new dict to ensure LangGraph properly merges the state
        return {**state, "matched_topics": matched_topics, "error": error_msg}


def context_retriever(state: TutorState) -> TutorState:
    """Retrieve relevant pages from knowledge base.

    Uses ChromaDB pages collection to get actual textbook pages.
    """
    matched_topics = state.get("matched_topics", [])
    print(f"[Context Retriever] Finding pages for {len(matched_topics)} topics...")

    # If no topics matched, we can't retrieve context
    if not matched_topics or len(matched_topics) == 0:
        print("[Context Retriever] No topics matched - skipping context retrieval")
        state["matched_pages"] = []
        return state

    try:
        router = get_topic_router()

        # Retrieve pages for each matched topic (3-5 topics, 10 pages each)
        all_pages = []
        max_topics = min(5, len(matched_topics))  # Take up to 5 topics
        
        for topic_data in matched_topics[:max_topics]:
            topic_name = topic_data.get("topic", "")
            grade = topic_data.get("grade")
            discipline_id = topic_data.get("discipline_id")
            book_topic_id = topic_data.get("book_topic_id")

            pages_retrieved = False

            # Retrieve pages using book_topic_id if available
            if book_topic_id and router.pages_collection:
                try:
                    pages = router._retrieve_pages_for_topic(
                        book_topic_id=book_topic_id,
                        grade=grade,
                        discipline_id=discipline_id,
                        max_pages=10,  # 10 pages per topic
                    )
                    if pages:
                        all_pages.extend(pages)
                        pages_retrieved = True
                        print(
                            f"[Context Retriever] Retrieved {len(pages)} pages for topic '{topic_name}' (book_topic_id: {book_topic_id})"
                        )
                except Exception as e:
                    print(
                        f"[Context Retriever] Warning: Failed to retrieve pages for book_topic_id {book_topic_id}: {e}"
                    )

            # Fallback: use retrieved_docs from topic_router if no pages were retrieved for this topic
            if not pages_retrieved:
                retrieved_docs = topic_data.get("retrieved_docs", [])
                if retrieved_docs:
                    all_pages.extend(retrieved_docs)
                    print(
                        f"[Context Retriever] Using {len(retrieved_docs)} documents from topic router for '{topic_name}'"
                    )

        # Store pages in state (convert strings to dict format)
        if all_pages:
            state["matched_pages"] = [{"content": page} for page in all_pages]
        else:
            state["matched_pages"] = []

        print(
            f"[Context Retriever] Retrieved {len(state['matched_pages'])} total page/document chunks"
        )

    except Exception as e:
        print(f"[Context Retriever] Error: {e}")
        # Fallback to using retrieved_docs from matched_topics
        matched_pages = []
        for topic_data in matched_topics:
            retrieved_docs = topic_data.get("retrieved_docs", [])
            if retrieved_docs:
                matched_pages.extend([{"content": doc} for doc in retrieved_docs])
        state["matched_pages"] = matched_pages
        print(
            f"[Context Retriever] Fallback: Using {len(matched_pages)} documents from topic router"
        )

    return state


def personalization_engine(state: TutorState) -> TutorState:
    """Load and apply student profile for personalization.

    Uses benchmark_scores and benchmark_absences data.
    """
    # TODO: Implement with actual student data
    student_id = state.get("student_id")
    print(f"[Personalization] Student ID: {student_id or 'None (anonymous)'}")
    return state


def content_generator(state: TutorState) -> TutorState:
    """Generate lecture content using Lapa LLM.

    Creates structured explanation based on retrieved textbook pages.
    """
    import sys
    import traceback

    print("[Content Generator] Generating lecture content...")
    sys.stdout.flush()

    # Initialize fallback values
    try:
        teacher_query = state.get("teacher_query", "")
        topic_name = ""
        matched_topics = state.get("matched_topics", [])
        if matched_topics:
            # Use first topic as main, but show we have multiple
            topic_name = matched_topics[0].get("topic", "")
            if len(matched_topics) > 1:
                topic_name += f" (та {len(matched_topics)-1} інших тем)"

        print(
            f"[Content Generator] Topics: {len(matched_topics)}, Main: {topic_name}, Query: {teacher_query[:50] if teacher_query else 'EMPTY'}"
        )
        sys.stdout.flush()

        # Default fallback content
        fallback_content = f"# {topic_name or teacher_query or 'Загальна тема'}\n\nКонспект готується на основі матеріалу з підручника.\n\n**Запит:** {teacher_query}"

        # Set fallback immediately to ensure we always have something
        state["lecture_content"] = fallback_content
        print(f"[Content Generator] Set initial fallback content")
        sys.stdout.flush()
    except Exception as e:
        print(f"[Content Generator] Error in initialization: {e}")
        print(f"[Content Generator] Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        teacher_query = state.get("teacher_query", "")
        state["lecture_content"] = (
            f"# {teacher_query or 'Загальна тема'}\n\nКонспект готується..."
        )
        return state

    try:
        from ..llm.lapa import LapaLLM

        lapa = LapaLLM()

        # Get context from retrieved pages
        matched_pages = state.get("matched_pages", [])
        print(f"[Content Generator] Found {len(matched_pages)} pages for context")
        import sys

        sys.stdout.flush()

        # Combine all page content into context
        context_parts = []
        for page_data in matched_pages:
            if isinstance(page_data, dict):
                content = page_data.get("content", "")
            else:
                content = str(page_data)
            if content:
                context_parts.append(content)

        context = "\n\n".join(context_parts)
        
        # Log context size
        print(f"[Content Generator] Context size: {len(context)} chars from {len(context_parts)} pages")
        if context:
            print(f"[Content Generator] Context preview (first 200 chars): {context[:200]}...")
        sys.stdout.flush()

        # If no context, use the query itself
        if not context:
            context = teacher_query
            print(
                "[Content Generator] Warning: No pages retrieved, using query as context"
            )

        # Build prompt for content generation
        query_text = teacher_query if teacher_query else f"Поясни тему: {topic_name}"

        system_prompt = """Ти - досвідчений вчитель, який створює ВИЧЕРПНІ та ЗРОЗУМІЛІ навчальні конспекти для учнів.

ТВОЯ ЗАДАЧА: Створити ДЕТАЛЬНИЙ, структурований конспект, який повною мірою розкриває тему на основі матеріалу з підручника.

СТРУКТУРА КОНСПЕКТУ (обов'язково дотримуйся):
1. Головний заголовок (# Назва теми)
2. ВСТУПНИЙ РОЗДІЛ (2-4 речення):
   - Що таке ця тема
   - Чому вона важлива
   - Зв'язок з попередніми темами (якщо є)
3. ОСНОВНІ РОЗДІЛИ (## Підзаголовки) - КОЖЕН розділ має містити:
   - ПОВНИЙ абзац введення (3-5 речень), що пояснює концепцію
   - Детальні пояснення з прикладами
   - Визначення важливих термінів (**жирним**)
   - Конкретні приклади з реального життя або навчальні приклади
   - Пояснення зв'язків між поняттями та причинно-наслідкових зв'язків
   - Як це застосовується на практиці
4. ПРАКТИЧНІ ПРИКЛАДИ (якщо доречно):
   - Конкретні випадки використання
   - Типові помилки та як їх уникати
5. ВИСНОВОК або ПІДСУМОК:
   - Основні моменти теми (2-3 речення)
   - Що важливо запам'ятати

ВИМОГИ ДО СТИЛЮ:
- Пиши ДЕТАЛЬНО та ВИЧЕРПНО - кожна ідея має бути розкрита повністю
- Використовуй ПОВНІ АБЗАЦИ (мінімум 3-4 речення на абзац), а НЕ тільки списки
- Пояснюй матеріал так, ніби ти на справжньому уроці - детально та зрозуміло
- Використовуй прості слова та конструкції - матеріал має бути зрозумілим для учнів
- Додавай ПРИКЛАДИ та АНАЛОГІЇ для кращого розуміння
- Пояснюй НЕ ТІЛЬКИ "що", але й "чому" та "як"
- Створюй ЛОГІЧНУ послідовність викладення - кожен розділ випливає з попереднього

ПРО СПИСКИ:
- Використовуй списки ТІЛЬКИ для переліків або послідовностей кроків
- ПЕРЕД кожним списком обов'язково має бути АБЗАЦ-ВСТУП (2-3 речення)
- ПІСЛЯ списку додай АБЗАЦ-ПОЯСНЕННЯ (якщо потрібно)

ФОРМАТУВАННЯ (Markdown):
- # для головного заголовка
- ## для основних розділів (розкривай кожен детально)
- ### для підрозділів
- **жирний** для важливих термінів та визначень
- *курсив* для акцентів
- $формула$ для інлайн формул
- $$формула$$ для окремих формул
- Нумеровані списки (1., 2., 3.) тільки для послідовностей
- Марковані списки (- або *) тільки для переліків

ПАМ'ЯТАЙ: Твій конспект має бути ТАКИМ ПОВНИМ, щоб учень міг зрозуміти тему тільки з нього, без додаткових джерел."""

        print(f"[Content Generator] Calling Lapa LLM with query: {query_text[:100]}...")
        sys.stdout.flush()

        # Generate content
        try:
            lecture_content = lapa.generate_with_context(
                query=query_text,
                context=context,
                system=system_prompt,
                temperature=0.7,
                max_tokens=5000,  # Збільшено для вичерпного та детального конспекту
            )

            # Handle None response
            if lecture_content is None:
                print("[Content Generator] Warning: LLM returned None, using fallback")
                lecture_content = fallback_content
        except Exception as llm_error:
            print(f"[Content Generator] LLM call failed: {llm_error}")
            import traceback

            print(f"[Content Generator] LLM error traceback: {traceback.format_exc()}")
            sys.stdout.flush()
            lecture_content = fallback_content

        print(
            f"[Content Generator] LLM returned {len(lecture_content) if lecture_content else 0} chars"
        )

        # Ensure we have content (even if empty, set a fallback)
        if not lecture_content or lecture_content.strip() == "":
            print(
                "[Content Generator] Warning: Generated content is empty, using fallback"
            )
            lecture_content = fallback_content
        else:
            # Ensure content starts with a heading
            if not lecture_content.strip().startswith("#"):
                lecture_content = (
                    f"# {topic_name or teacher_query}\n\n{lecture_content}"
                )

        state["lecture_content"] = lecture_content
        print(
            f"[Content Generator] Final lecture content: {len(lecture_content)} chars"
        )
        print(f"[Content Generator] Content preview: {lecture_content[:100]}...")
        import sys

        sys.stdout.flush()
        print(
            f"[Content Generator] State after setting: lecture_content in state = {'lecture_content' in state}"
        )
        sys.stdout.flush()

    except Exception as e:
        import traceback
        import sys

        print(f"[Content Generator] Error in try block: {e}")
        print(f"[Content Generator] Traceback: {traceback.format_exc()}")
        sys.stdout.flush()

        # Fallback: create simple content from query
        state["lecture_content"] = fallback_content
        if not state.get("error"):
            state["error"] = f"Content generation failed: {str(e)}"
        print(
            f"[Content Generator] Using fallback content: {len(fallback_content)} chars"
        )
        sys.stdout.flush()

    # Final safety check - ensure lecture_content is never empty
    import sys

    if not state.get("lecture_content") or state["lecture_content"].strip() == "":
        print(
            "[Content Generator] CRITICAL: lecture_content is still empty, using emergency fallback"
        )
        sys.stdout.flush()
        state["lecture_content"] = (
            f"# {teacher_query or 'Загальна тема'}\n\nКонспект готується..."
        )

    final_content = state.get("lecture_content", "")
    print(
        f"[Content Generator] Returning state with lecture_content length: {len(final_content)}"
    )
    print(
        f"[Content Generator] Final content preview: {final_content[:100] if final_content else 'EMPTY'}..."
    )
    sys.stdout.flush()

    # Return new dict to ensure LangGraph properly merges the state
    return {**state, "lecture_content": state.get("lecture_content", "")}


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
        recommendations_text = (
            practice_recommendations or "Немає додаткових рекомендацій."
        )
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
        expected_index = (
            ord(correct_letter) - ord("A") if correct_letter in "ABCD" else None
        )

        options_text = "\n".join(
            [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
        )
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

    # Log what we have in state
    lecture_content = state.get("lecture_content", "")
    matched_topics = state.get("matched_topics", [])
    matched_pages = state.get("matched_pages", [])

    print(
        f"[Finalizer] lecture_content length: {len(lecture_content) if lecture_content else 0}"
    )
    print(f"[Finalizer] matched_topics count: {len(matched_topics)}")
    print(f"[Finalizer] matched_pages count: {len(matched_pages)}")
    import sys

    sys.stdout.flush()

    # Log matched_topics details
    if matched_topics:
        for i, topic in enumerate(matched_topics):
            print(f"[Finalizer] matched_topics[{i}]: {topic.get('topic', 'N/A')}")
    sys.stdout.flush()

    # Ensure lecture_content exists
    if not lecture_content or lecture_content.strip() == "":
        print("[Finalizer] WARNING: lecture_content is empty, setting fallback")
        teacher_query = state.get("teacher_query", "")
        topic_name = ""
        if matched_topics:
            topic_name = matched_topics[0].get("topic", "")
        state["lecture_content"] = (
            f"# {topic_name or teacher_query or 'Загальна тема'}\n\nКонспект готується..."
        )
        print(
            f"[Finalizer] Set fallback lecture_content: {len(state['lecture_content'])} chars"
        )
        sys.stdout.flush()

    print(
        f"[Finalizer] Final state - lecture_content: {len(state.get('lecture_content', ''))} chars, matched_topics: {len(state.get('matched_topics', []))}"
    )
    sys.stdout.flush()

    # Return new dict to ensure all fields are properly included
    return {
        **state,
        "lecture_content": state.get("lecture_content", ""),
        "matched_topics": state.get("matched_topics", []),
        "matched_pages": state.get("matched_pages", []),
    }


# === Conditional Edges ===


def should_regenerate(
    state: TutorState,
) -> Literal["practice_generator", "check_or_finalize"]:
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
            "check_or_finalize": "check_or_finalize_router",
        },
    )

    # Router node for check/finalize decision
    graph.add_node("check_or_finalize_router", lambda s: s)
    graph.add_conditional_edges(
        "check_or_finalize_router",
        has_student_answers,
        {"check_answers": "check_answers", "finalizer": "finalizer"},
    )

    graph.add_edge("check_answers", "recommendations")
    graph.add_edge("recommendations", "finalizer")
    graph.add_edge("finalizer", END)

    return graph.compile()


# Compiled graph instance
tutor_graph = create_tutor_graph()
