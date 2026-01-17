"""Content Generator agent - generates lecture content using Lapa LLM.

This agent:
1. Takes retrieved context from topic router
2. Uses Lapa LLM to generate structured lecture content
3. Creates control questions for comprehension check
4. Tracks sources for grounding
"""
from typing import Any, Dict, List, Optional

from ..llm.lapa import LapaLLM


class ContentGenerator:
    """Generator for structured lecture content using Lapa LLM.
    
    Produces:
    - Topic explanation appropriate for grade level
    - Key concepts and definitions
    - Examples and illustrations
    - Control questions for comprehension
    """
    
    def __init__(self):
        """Initialize ContentGenerator with Lapa LLM client."""
        self.lapa = LapaLLM()
    
    def generate(
        self,
        query: str,
        topic: str,
        retrieved_docs: List[str],
        grade: int = 9,
        subject: str = "Українська мова",
        personalization_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate lecture content based on query and retrieved context.
        
        Args:
            query: Original teacher query
            topic: Matched topic name
            retrieved_docs: List of retrieved document texts (context)
            grade: Grade level (8 or 9)
            subject: Subject name
            personalization_prompt: Optional prompt_injection from PersonalizationEngine
            
        Returns:
            Dict with:
                - "lecture_content": str - Generated lecture text
                - "control_questions": List[str] - Comprehension questions
                - "sources": List[str] - Source references for grounding
        """
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Build the prompt
        prompt = self._build_prompt(
            query=query,
            topic=topic,
            context=context,
            grade=grade,
            subject=subject,
            personalization_prompt=personalization_prompt
        )
        
        # Generate content using Lapa
        system_prompt = self._get_system_prompt(subject, grade)
        
        response = self.lapa.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.7,
            max_tokens=3000
        )
        
        # Parse the response
        lecture_content, control_questions = self._parse_response(response)
        
        # Extract sources from retrieved docs
        sources = self._extract_sources(retrieved_docs)
        
        return {
            "lecture_content": lecture_content,
            "control_questions": control_questions,
            "sources": sources
        }
    
    def _build_context(self, retrieved_docs: List[str]) -> str:
        """Combine retrieved documents into a single context string."""
        if not retrieved_docs:
            return ""
        
        # Join documents with clear separators
        return "\n\n".join(retrieved_docs)
    
    def _get_system_prompt(self, subject: str, grade: int) -> str:
        """Get system prompt for content generation."""
        return f"""Ти досвідчений вчитель {subject} для учнів {grade} класу української школи.

Твоя роль:
- Пояснювати матеріал чітко, зрозуміло та структуровано
- Використовувати приклади, що відповідають віку учнів
- Дотримуватись навчальної програми України
- Базувати пояснення ЛИШЕ на наданому контексті з підручника

Стиль:
- Використовуй українську мову
- Будь дружнім та підтримуючим
- Уникай надто складної термінології без пояснень"""

    def _build_prompt(
        self,
        query: str,
        topic: str,
        context: str,
        grade: int,
        subject: str,
        personalization_prompt: Optional[str] = None
    ) -> str:
        """Build the generation prompt."""
        personalization_text = ""
        if personalization_prompt:
            personalization_text = f"\n\nКОНТЕКСТ УЧНЯ:\n{personalization_prompt}"
        
        prompt = f"""Створи структурований конспект уроку для учня {grade} класу з предмету "{subject}".

Запит вчителя: {query}
Тема: {topic}{personalization_text}

КОНТЕКСТ З ПІДРУЧНИКА:
\"\"\"
{context}
\"\"\"

ВИМОГИ ДО КОНСПЕКТУ:

1. **Вступ** (2-3 речення):
   - Чому ця тема важлива
   - Що учень дізнається

2. **Основний матеріал**:
   - Чіткі визначення ключових понять
   - Покрокові пояснення
   - 2-3 приклади з життя або підручника

3. **Важливо запам'ятати**:
   - Головні факти (3-5 пунктів)

4. **Контрольні питання** (2-3 питання):
   - Для перевірки розуміння
   - Різного рівня складності

ФОРМАТ ВІДПОВІДІ:

## Вступ
[текст вступу]

## Основний матеріал
[детальне пояснення теми]

## Важливо запам'ятати
- [пункт 1]
- [пункт 2]
- [пункт 3]

## Контрольні питання
1. [питання 1]
2. [питання 2]
3. [питання 3]

ВАЖЛИВО: Використовуй ТІЛЬКИ інформацію з наданого контексту. Якщо якоїсь інформації немає - не вигадуй."""

        return prompt
    
    def _parse_response(self, response: str) -> tuple[str, List[str]]:
        """Parse LLM response into lecture content and control questions.
        
        Returns:
            Tuple of (lecture_content, control_questions)
        """
        control_questions = []
        
        # Try to extract control questions section
        if "## Контрольні питання" in response:
            parts = response.split("## Контрольні питання")
            lecture_content = parts[0].strip()
            questions_section = parts[1] if len(parts) > 1 else ""
            
            # Parse questions (numbered list)
            for line in questions_section.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering/bullet
                    question = line.lstrip("0123456789.-) ").strip()
                    if question:
                        control_questions.append(question)
        else:
            # No clear separation - return full response as content
            lecture_content = response
        
        return lecture_content, control_questions
    
    def _extract_sources(self, retrieved_docs: List[str]) -> List[str]:
        """Extract source references from retrieved documents.
        
        Parses document format: "Документ N (сторінка X): ..."
        Returns list of formatted source strings.
        """
        sources = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Try to extract page number from format: "Документ N (сторінка X): ..."
            if "(сторінка" in doc:
                import re
                match = re.search(r"\(сторінка\s*(\d+)\)", doc)
                if match:
                    page_num = match.group(1)
                    sources.append(f"Сторінка {page_num}")
                else:
                    sources.append(f"Джерело {i}")
            elif "Документ" in doc:
                # Try to extract any useful info from document header
                # Format might be "Документ 1: TOPIC: ..."
                import re
                topic_match = re.search(r"TOPIC:\s*([^\\n]+)", doc)
                if topic_match:
                    topic_name = topic_match.group(1).strip()[:50]
                    sources.append(f"Джерело {i}: {topic_name}")
                else:
                    # Try to get first meaningful content
                    doc_preview = doc.split("Документ")[1] if "Документ" in doc else doc
                    doc_preview = doc_preview.strip()[:50]
                    if doc_preview:
                        sources.append(f"Джерело {i}: {doc_preview}...")
                    else:
                        sources.append(f"Джерело {i}")
            else:
                sources.append(f"Джерело {i}")
        
        return sources


def generate_content(state: dict) -> dict:
    """Node function for LangGraph workflow.
    
    Args:
        state: TutorState dictionary
        
    Returns:
        Dict with lecture_content, control_questions, sources, and error (if any)
    """
    print("[Content Generator] Generating lecture content...")
    
    # Get data from state
    query = state.get("teacher_query", "")
    grade = state.get("grade", 9)
    subject = state.get("subject", "Українська мова")
    matched_topics = state.get("matched_topics", [])
    
    # Get personalization prompt from student_profile
    student_profile = state.get("student_profile")
    personalization_prompt = None
    if student_profile and isinstance(student_profile, dict):
        personalization_prompt = student_profile.get("prompt_injection")
    
    # Extract topic and retrieved docs from matched_topics
    topic = ""
    retrieved_docs = []
    
    if matched_topics:
        first_match = matched_topics[0]
        if isinstance(first_match, dict):
            topic = first_match.get("topic", "")
            retrieved_docs = first_match.get("retrieved_docs", [])
        elif isinstance(first_match, str):
            topic = first_match
    
    # Check if we have context to work with
    if not retrieved_docs:
        print("[Content Generator] Warning: No retrieved documents available")
        return {
            "lecture_content": "",
            "control_questions": [],
            "sources": [],
            "error": None
        }
    
    try:
        generator = ContentGenerator()
        result = generator.generate(
            query=query,
            topic=topic,
            retrieved_docs=retrieved_docs,
            grade=grade,
            subject=subject,
            personalization_prompt=personalization_prompt
        )
        
        print(f"[Content Generator] Generated {len(result['lecture_content'])} chars of content")
        print(f"[Content Generator] Created {len(result['control_questions'])} control questions")
        
        return {
            "lecture_content": result["lecture_content"],
            "control_questions": result["control_questions"],
            "sources": result["sources"],
            "error": None
        }
        
    except Exception as e:
        print(f"[Content Generator] Error: {e}")
        return {
            "lecture_content": "",
            "control_questions": [],
            "sources": [],
            "error": f"Content generation failed: {str(e)}"
        }


