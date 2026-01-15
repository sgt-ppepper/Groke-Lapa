import json
import re
from typing import Dict, List

from dotenv import load_dotenv

from src.agents.graph import practice_generator


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_practice(questions: List[Dict], subtopics: List[str]) -> Dict[str, float]:
    total = len(questions)
    if total == 0:
        return {
            "total": 0,
            "valid_options_ratio": 0.0,
            "valid_answer_ratio": 0.0,
            "non_empty_explanation_ratio": 0.0,
            "unique_question_ratio": 0.0,
            "subtopic_coverage_ratio": 0.0,
        }

    valid_options = sum(1 for q in questions if len(q.get("options", [])) == 4)
    valid_answers = sum(
        1
        for q in questions
        if (q.get("correct_answer") or "").strip().upper() in {"A", "B", "C", "D"}
    )
    non_empty_expl = sum(1 for q in questions if q.get("explanation"))

    normalized_questions = [_normalize_text(q.get("question", "")) for q in questions]
    unique_questions = len(set(normalized_questions))

    subtopics_text = [s for s in (subtopics or []) if s]
    if subtopics_text:
        coverage_hits = 0
        for subtopic in subtopics_text:
            token = _normalize_text(subtopic)
            if any(token in _normalize_text(q.get("question", "")) for q in questions):
                coverage_hits += 1
        subtopic_coverage_ratio = coverage_hits / len(subtopics_text)
    else:
        subtopic_coverage_ratio = 0.0

    return {
        "total": total,
        "valid_options_ratio": valid_options / total,
        "valid_answer_ratio": valid_answers / total,
        "non_empty_explanation_ratio": non_empty_expl / total,
        "unique_question_ratio": unique_questions / total,
        "subtopic_coverage_ratio": subtopic_coverage_ratio,
    }


def run_test():
    load_dotenv()

    examples = [
        {
            "subject": "Українська мова",
            "grade": 9,
            "topic": "Складне речення",
            "subtopics": ["Складносурядне", "Складнопідрядне"],
            "student_level": "олімпіадний",
            "practice_recommendations": "Можна давати завдання підвищеної складності (рівень олімпіади).",
            "lecture_content": (
                "Складне речення — це речення з двома і більше граматичними основами. "
                "Складносурядне речення поєднує частини сурядними сполучниками (і, але, або). "
                "Складнопідрядне речення має головну та підрядну частини, поєднані підрядними сполучниками (що, коли, тому що)."
            ),
        },
        {
            "subject": "Історія України",
            "grade": 9,
            "topic": "Національно-визвольна війна середини XVII ст.",
            "subtopics": ["Причини війни", "Богдан Хмельницький"],
            "student_level": "сильний",
            "practice_recommendations": "Питання можуть бути складніші, але з чіткою відповіддю.",
            "lecture_content": (
                "Національно-визвольна війна середини XVII ст. розпочалася 1648 року. "
                "Богдан Хмельницький став гетьманом і очолив козацьке військо. "
                "Причини війни включали соціальний і національний гніт, обмеження прав козацтва."
            ),
        },
        {
            "subject": "Алгебра",
            "grade": 8,
            "topic": "Квадратні рівняння",
            "subtopics": ["Дискримінант", "Формула коренів"],
            "student_level": "середній",
            "practice_recommendations": "Частину питань зроби з підстановкою чисел у формулу.",
            "lecture_content": (
                "Квадратне рівняння має вигляд ax^2 + bx + c = 0, де a ≠ 0. "
                "Дискримінант D = b^2 - 4ac. "
                "Якщо D > 0, маємо два корені: x1 = (-b + √D)/(2a), x2 = (-b - √D)/(2a)."
            ),
        },
    ]

    for example in examples:
        state = {
            "subject": example["subject"],
            "grade": example["grade"],
            "topic": example["topic"],
            "subtopics": example["subtopics"],
            "student_level": example["student_level"],
            "practice_recommendations": example["practice_recommendations"],
            "lecture_content": example["lecture_content"],
            "practice_count": 5,
            "validation_results": [],
            "validation_feedback": None,
            "regeneration_count": 0,
        }

        result = practice_generator(state)
        questions = result.get("practice_questions", [])
        metrics = evaluate_practice(questions, example["subtopics"])

        print("=" * 50)
        print(f"PRACTICE GENERATOR: {example['subject']}")
        print("=" * 50)
        print(f"Questions: {len(questions)}")
        print("Metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("Sample:", json.dumps(questions[:2], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_test()
