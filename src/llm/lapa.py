"""Lapa LLM client for content generation."""
from openai import OpenAI
from typing import Optional, List, Dict, Any

from ..config import get_settings


class LapaLLM:
    """Client for Lapa LLM (12B Instruct) - optimized for content generation.
    
    Use cases:
    - Generating lecture content
    - Creating explanations
    - Building recommendations
    """
    
    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(
            api_key=settings.lapathon_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.lapa_model
        self.function_calling_model = settings.lapa_function_calling_model
    
    def generate(
        self, 
        prompt: str, 
        system: Optional[str] = None, 
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text completion using Lapa.
        
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
    
    def function_call(
        self, 
        prompt: str, 
        tools: List[Dict[str, Any]]
    ) -> str:
        """Use lapa-function-calling for structured output.
        
        Args:
            prompt: User prompt with tool instructions
            tools: List of tool definitions
            
        Returns:
            Function call result as string
        """
        # Build function calling prompt
        fc_prompt = f"""Ти модель штучного інтелекту з викликом функцій. Тобі надаються підписи функцій всередині тегів XML. Ти можеш викликати одну або кілька функцій, щоб допомогти з запитом користувача.

Ось доступні інструменти: 
{tools}

Використовуй JSON для виклику функцій.

{prompt}"""
        
        response = self.client.chat.completions.create(
            model=self.function_calling_model,
            messages=[{"role": "user", "content": fc_prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate response with retrieved context (RAG pattern).
        
        Args:
            query: User query
            context: Retrieved context from knowledge base
            system: Optional system message
            temperature: Sampling temperature
            
        Returns:
            Generated response grounded in context
        """
        prompt = f"""Контекст з підручників:
{context}

Запит: {query}

Дай відповідь на основі наданого контексту. Якщо інформації недостатньо, скажи про це."""

        return self.generate(prompt, system=system, temperature=temperature)
    
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

