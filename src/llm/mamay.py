"""Mamay LLM client for topic routing, practice, and solving."""
from openai import OpenAI
from typing import Optional, List

from ..config import get_settings


class MamayLLM:
    """Client for MamayLM (12B Gemma-3 Instruct).
    
    Use cases:
    - Topic routing
    - Practice/test generation
    - Question solving and validation
    """
    
    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(
            api_key=settings.lapathon_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.mamay_model
    
    def generate(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text completion using Mamay.
        
        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def solve_question(
        self,
        question_text: str,
        answers: List[str],
        subject: str = "Загальний",
        concise: bool = False
    ) -> dict:
        """Solve a multiple-choice question.
        
        Args:
            question_text: The question text
            answers: List of answer options
            subject: Subject name for context
            concise: If True, output only the answer letter (faster, fewer tokens)
            
        Returns:
            Dict with 'answer_index', 'answer_text', and 'reasoning'
        """
        options = "\n".join([f"{chr(65+i)}. {a}" for i, a in enumerate(answers)])
        
        if concise:
            # Concise mode: prime model to think, but only output letter
            prompt = f"""Предмет: {subject}

Питання: {question_text}

Варіанти відповіді:
{options}

Використай наданий контекст з підручників. Проаналізуй кожен варіант, визнач правильну відповідь.
Виведи ТІЛЬКИ одну літеру (A, B, C або D):"""
            response = self.generate(prompt, temperature=0.0, max_tokens=50)
            
            # Parse single letter response
            answer_letter = None
            for char in response.strip():
                if char.upper() in 'ABCD':
                    answer_letter = char.upper()
                    break
        else:
            # Full reasoning mode
            prompt = f"""Предмет: {subject}

Питання: {question_text}

Варіанти відповіді:
{options}

Використай наданий контекст з підручників для відповіді.
Розв'яжи це питання крок за кроком, потім дай фінальну відповідь у форматі:
ВІДПОВІДЬ: [літера]"""
            response = self.generate(prompt, temperature=0.1)
            
            # Parse answer from reasoning
            answer_letter = None
            for line in response.split('\n'):
                if 'ВІДПОВІДЬ:' in line.upper():
                    for char in line:
                        if char.upper() in 'ABCD':
                            answer_letter = char.upper()
                            break
                    break
        
        answer_index = ord(answer_letter) - ord('A') if answer_letter else 0
        
        return {
            "answer_index": answer_index,
            "answer_text": answers[answer_index] if answer_index < len(answers) else "",
            "reasoning": response if not concise else f"Answer: {answer_letter}"
        }
    
    def generate_practice(
        self,
        topic: str,
        subtopics: List[str],
        grade: int,
        subject: str,
        count: int = 8,
        difficulty: str = "середня"
    ) -> str:
        """Generate practice questions for a topic.
        
        Args:
            topic: Main topic name
            subtopics: List of subtopics to cover
            grade: Grade level (8 or 9)
            subject: Subject name
            count: Number of questions to generate
            difficulty: Difficulty level (легка, середня, складна)
            
        Returns:
            Generated questions in structured format
        """
        subtopics_text = ", ".join(subtopics) if subtopics else "загальні аспекти теми"
        
        prompt = f"""Створи {count} тестових питань для учнів {grade} класу з предмету "{subject}".

Тема: {topic}
Підтеми: {subtopics_text}
Складність: {difficulty}

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

        return self.generate(prompt, temperature=0.8, max_tokens=4000)
