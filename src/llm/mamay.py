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
        api_key = settings.lapathon_api_key
        
        # Validate API key is set
        if not api_key or api_key == "":
            raise ValueError(
                "LAPATHON_API_KEY is not set. Please create a .env file with your API key. "
                "Example: LAPATHON_API_KEY=your_key_here"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url,
            timeout=120.0  # 2 minute timeout
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
        try:
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
        except Exception as e:
            error_msg = str(e)
            # Check if it's an authentication error
            if "401" in error_msg or "Authentication" in error_msg or "token" in error_msg.lower():
                api_key_preview = f"{self.client.api_key[:10]}..." if self.client.api_key else "NOT SET"
                raise ValueError(
                    f"API Authentication Error: Invalid or missing API key. "
                    f"Please check your LAPATHON_API_KEY in .env file. "
                    f"Current key starts with: {api_key_preview}"
                ) from e
            raise
    
    def solve_question(
        self,
        question_text: str,
        answers: List[str],
        subject: str = "Загальний"
    ) -> dict:
        """Solve a multiple-choice question.
        
        Args:
            question_text: The question text
            answers: List of answer options
            subject: Subject name for context
            
        Returns:
            Dict with 'answer_index', 'answer_text', and 'reasoning'
        """
        options = "\n".join([f"{chr(65+i)}. {a}" for i, a in enumerate(answers)])
        
        prompt = f"""Предмет: {subject}

Питання: {question_text}

Варіанти відповіді:
{options}

Розв'яжи це питання крок за кроком, потім дай фінальну відповідь у форматі:
ВІДПОВІДЬ: [літера]"""

        response = self.generate(prompt, temperature=0.1)
        
        # Parse answer
        answer_letter = None
        for line in response.split('\n'):
            if 'ВІДПОВІДЬ:' in line.upper():
                # Extract letter
                for char in line:
                    if char.upper() in 'ABCD':
                        answer_letter = char.upper()
                        break
                break
        
        answer_index = ord(answer_letter) - ord('A') if answer_letter else 0
        
        return {
            "answer_index": answer_index,
            "answer_text": answers[answer_index] if answer_index < len(answers) else "",
            "reasoning": response
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
