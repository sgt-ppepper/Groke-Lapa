"""Practice Generator - generates practice questions using MamayLLM.

This module handles:
1. Generating practice questions based on topic and lecture content
2. Parsing LLM output into structured questions
3. Validating questions (for Algebra using Python REPL)
4. Building validation feedback for regeneration
"""
import ast
import math
import re
from typing import Any, Dict, List, Optional

from ..llm.mamay import MamayLLM


# === Parsing Utilities ===

def parse_practice_questions(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output into structured practice questions.
    
    Expected format:
    ---
    ПИТАННЯ 1:
    [question text]
    
    A. [option A]
    B. [option B]
    C. [option C]
    D. [option D]
    
    ВІДПОВІДЬ: [letter]
    ПОЯСНЕННЯ: [explanation]
    ---
    
    Returns:
        List of question dicts with keys:
        - question: str
        - options: List[str]
        - correct_answer: str (A, B, C, or D)
        - explanation: str
        - is_validated: bool
    """
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


# === Validation Utilities (for Algebra) ===

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
    """Check if a function call is allowed in validator code."""
    if isinstance(node.func, ast.Name):
        return node.func.id in {"abs", "pow", "round", "min", "max"}
    if isinstance(node.func, ast.Attribute):
        return isinstance(node.func.value, ast.Name) and node.func.value.id == "math"
    return False


def _validate_ast(tree: ast.AST) -> None:
    """Validate AST to ensure only safe constructs are used."""
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
    """Execute limited python code and return ANSWER_INDEX if present.
    
    The code should set either:
    - ANSWER_INDEX: int (0-3)
    - ANSWER_LETTER: str ("A", "B", "C", or "D")
    
    Returns:
        Answer index (0-3) or None if not found/invalid
    """
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


# === Helper Functions ===

def topic_to_text(topic: Any) -> str:
    """Convert topic (str or dict) to text representation."""
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


def build_validation_feedback(state: dict) -> str:
    """Build feedback string from validation results for regeneration."""
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


# === Main Generation Function ===

def generate_practice(
    topic: str,
    grade: int,
    subject: str,
    lecture_content: str = "",
    subtopics: Optional[List[str]] = None,
    personalization_prompt: Optional[str] = None,
    practice_count: int = 8,
    validator_feedback: str = ""
) -> Dict[str, Any]:
    """Generate practice questions for a topic.
    
    Args:
        topic: Main topic name
        grade: Grade level (8 or 9)
        subject: Subject name
        lecture_content: Optional lecture content to base questions on
        subtopics: Optional list of subtopics
        personalization_prompt: Optional prompt_injection from PersonalizationEngine
        practice_count: Number of questions to generate (default 8)
        validator_feedback: Feedback from previous validation for regeneration
        
    Returns:
        Dict with:
        - practice_questions: List of parsed question dicts
        - raw_output: Raw LLM output (for debugging)
        - error: Error message if any
    """
    print("[Practice Generator] Generating practice questions...")
    mamay = MamayLLM()
    
    subtopics_text = ", ".join(subtopics) if subtopics else "загальні аспекти теми"
    
    # Build personalization section
    personalization_text = ""
    if personalization_prompt:
        personalization_text = f"\nКОНТЕКСТ УЧНЯ:\n{personalization_prompt}\n"
    
    prompt = f"""Створи {practice_count} тестових питань для учнів {grade} класу з предмету "{subject}".

Тема: {topic}
Підтеми: {subtopics_text}
{personalization_text}
Вимоги до якості:
- Кожне питання має ТІЛЬКИ одну правильну відповідь.
- 4 варіанти відповіді (A, B, C, D).
- Не повторюй питання або формулювання.
- Математичні формули записуй у форматі LaTeX ($формула$).
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

    try:
        raw_output = mamay.generate(prompt, temperature=0.8, max_tokens=4000)
        parsed_questions = parse_practice_questions(raw_output)
        
        print(f"[Practice Generator] Parsed {len(parsed_questions)} questions")
        
        return {
            "practice_questions": parsed_questions,
            "raw_output": raw_output,
            "error": None
        }
        
    except Exception as e:
        print(f"[Practice Generator] Error: {e}")
        return {
            "practice_questions": [],
            "raw_output": "",
            "error": str(e)
        }


def validate_questions(
    questions: List[Dict[str, Any]],
    subject: str
) -> Dict[str, Any]:
    """Validate questions by attempting to solve them.
    
    For Algebra questions, uses Python REPL to verify calculations.
    For other subjects, validation is skipped (returns all valid).
    
    Args:
        questions: List of question dicts
        subject: Subject name
        
    Returns:
        Dict with:
        - questions: Updated questions with validation info
        - validation_results: Detailed validation results
        - all_valid: bool indicating if all questions passed
    """
    print("[Solver/Validator] Validating questions...")
    
    # Only run code-based validation for Algebra
    if "алгебра" not in subject.lower():
        return {
            "questions": questions,
            "validation_results": [],
            "all_valid": True
        }

    if not questions:
        return {
            "questions": [],
            "validation_results": [],
            "all_valid": False
        }

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
        "questions": updated_questions,
        "validation_results": validation_results,
        "all_valid": all_valid
    }
