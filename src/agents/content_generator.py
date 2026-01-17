"""Content Generator agent - generates lecture content using Mamay LLM.

This agent:
1. Takes retrieved context from topic router
2. Uses Mamay LLM to generate structured lecture content
3. Creates control questions for comprehension check
4. Tracks sources for grounding
"""
from typing import Any, Dict, List, Optional

from ..llm.mamay import MamayLLM


class ContentGenerator:
    """Generator for structured lecture content using Mamay LLM.
    
    Produces:
    - Topic explanation appropriate for grade level
    - Key concepts and definitions
    - Examples and illustrations
    - Control questions for comprehension
    """
    
    def __init__(self):
        """Initialize ContentGenerator with Mamay LLM client."""
        self.llm = MamayLLM()
    
    def generate(
        self,
        query: str,
        topic: str,
        retrieved_docs: List[str],
        grade: int = 9,
        subject: str = "Українська мова",
        personalization_prompt: Optional[str] = None,
        source_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate lecture content based on query and retrieved context.
        
        Args:
            query: Original teacher query
            topic: Matched topic name
            retrieved_docs: List of retrieved document texts (context)
            grade: Grade level (8 or 9)
            subject: Subject name
            personalization_prompt: Optional prompt_injection from PersonalizationEngine
            source_info: Optional source metadata for grounding
            
        Returns:
            Dict with:
                - "lecture_content": str - Generated lecture text
                - "control_questions": List[str] - Comprehension questions
                - "sources": List[str] - Source references for grounding
        """
        # Build context from retrieved documents (limit size to avoid timeout)
        context = self._build_context(retrieved_docs, max_chars=8000)
        print(f"[Content Generator] Context: {len(context)} chars from {len(retrieved_docs)} docs")
        
        # Log personalization data
        if personalization_prompt:
            print(f"[Content Generator] 🎯 Personalization prompt:")
            for line in personalization_prompt.split('\n')[:5]:  # Show first 5 lines
                if line.strip():
                    print(f"[Content Generator]    {line.strip()}")
        else:
            print("[Content Generator] No personalization data available")
        
        # Build the prompt
        prompt = self._build_prompt(
            query=query,
            topic=topic,
            context=context,
            grade=grade,
            subject=subject,
            personalization_prompt=personalization_prompt
        )
        
        # Generate content using Mamay
        system_prompt = self._get_system_prompt(subject, grade)
        
        print("[Content Generator] Calling Mamay LLM...")
        response = self.llm.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.7,
            max_tokens=4000
        )
        print("[Content Generator] LLM response received")
        
        # Parse the response
        lecture_content, control_questions = self._parse_response(response)
        
        # Extract and format sources
        sources = self._extract_sources(retrieved_docs, source_info)
        
        return {
            "lecture_content": lecture_content,
            "control_questions": control_questions,
            "sources": sources
        }
    
    def _build_context(self, retrieved_docs: List[str], max_chars: int = 6000) -> str:
        """Combine retrieved documents into a single context string.
        
        Limits total context size to avoid LLM timeout.
        Uses up to 5 docs, max 1500 chars each.
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        total_chars = 0
        
        # Use first 5 docs
        for doc in retrieved_docs[:5]:
            # Truncate individual docs to 1500 chars
            doc_text = doc[:1500] if len(doc) > 1500 else doc
            
            if total_chars + len(doc_text) > max_chars:
                break
            
            context_parts.append(doc_text)
            total_chars += len(doc_text) + 2
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self, subject: str, grade: int) -> str:
        """Get system prompt for content generation."""
        return f"""Ти досвідчений вчитель {subject} для учнів {grade} класу української школи з багаторічним досвідом.

Твоя місія:
- Створювати глибокі, змістовні та захоплюючі навчальні матеріали
- Пояснювати складні концепції простою мовою з яскравими прикладами
- Адаптувати пояснення під рівень та потреби конкретного учня
- Використовувати сучасні педагогічні підходи

Стиль викладання:
- Чітка структура з логічними переходами між темами
- Приклади з реального життя, близькі підліткам
- Математичні формули записуй у форматі LaTeX ($формула$)
- Використовуй таблиці та списки для кращого сприйняття
- Додавай короткі історичні факти або цікавинки де доречно
- Будь дружнім, підтримуючим та мотивуючим"""

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
        # Build personalization section with specific instructions
        personalization_section = ""
        if personalization_prompt:
            personalization_section = f"""

🎯 ПЕРСОНАЛІЗАЦІЯ ДЛЯ УЧНЯ:
{personalization_prompt}

ВРАХОВУЙ ЦІ ДАНІ ПРИ СТВОРЕННІ КОНТЕНТУ:
- Якщо учень має низький бал - використовуй простіші пояснення, більше прикладів
- Якщо учень має високий бал - додай складніші приклади та поглиблення теми
- Якщо є слабкі теми - наголоси на зв'язках з ними для кращого розуміння
"""
        else:
            personalization_section = "\n(Персоналізація не доступна - використовуй стандартний підхід для середнього учня)"
        
        prompt = f"""Створи ГЛИБОКИЙ ТА ДЕТАЛЬНИЙ конспект уроку для учня {grade} класу з предмету "{subject}".

📚 ТЕМА: {topic}
❓ ЗАПИТ ВЧИТЕЛЯ: {query}
{personalization_section}

📖 КОНТЕКСТ З ПІДРУЧНИКА:
\"\"\"
{context}
\"\"\"

═══════════════════════════════════════════════════════════
ВИМОГИ ДО КОНСПЕКТУ (створи РОЗГОРНУТИЙ матеріал):
═══════════════════════════════════════════════════════════

## 🎯 Вступ
- Чому ця тема важлива для учня та де вона застосовується
- Коротка історична довідка або цікавий факт (якщо доречно)
- Що учень зможе зробити після вивчення теми

## 📚 Основний матеріал

### Теоретична частина
- Чіткі визначення ВСІХ ключових понять з теми
- Покрокові пояснення з логічними переходами
- Важливі формули/правила (у форматі LaTeX для математики)
- Зв'язок з раніше вивченим матеріалом

### Приклади та пояснення
- 3-4 детальні приклади з покроковим розв'язком
- Приклади від простого до складного
- Типові помилки та як їх уникнути

## 💡 Важливо запам'ятати
- 5-7 ключових фактів, формул або правил
- Мнемонічні прийоми для запам'ятовування (якщо є)

## ✅ Контрольні питання
1. Питання на розуміння термінів
2. Питання на застосування правил
3. Питання підвищеної складності (з формулами в LaTeX, де потрібно)

═══════════════════════════════════════════════════════════
⚠️ ВАЖЛИВО:
- Базуйся ТІЛЬКИ на наданому контексті з підручника
- НЕ вигадуй факти, яких немає в контексті
- Пиши українською мовою
- Всі математичні вирази огортай у долари: $x^2$
- Використовуй емодзі для структурування (помірно)
═══════════════════════════════════════════════════════════"""

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
    
    def _extract_sources(
        self, 
        retrieved_docs: List[str], 
        source_info: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract source references from retrieved documents.
        
        Args:
            retrieved_docs: List of document texts
            source_info: Optional dict with topic metadata:
                - subject, grade, topic_title, start_page, end_page
        
        Returns:
            List of formatted source strings for grounding
        """
        import re
        sources = []
        
        # 1. Primary source from topic metadata (highest quality)
        if source_info:
            subject = source_info.get("subject", "Підручник")
            grade = source_info.get("grade", "")
            topic_title = source_info.get("topic_title", "")
            start_page = source_info.get("start_page")
            end_page = source_info.get("end_page")
            
            # Build citation: "📘 Алгебра 9 клас: § 11. Функція... (стор. 75-82)"
            book_ref = subject
            if grade:
                book_ref += f" {grade} клас"
            
            citation = f"📘 {book_ref}"
            if topic_title:
                citation += f": {topic_title}"
            
            if start_page:
                p_range = f"{start_page}-{end_page}" if (end_page and end_page != start_page) else str(start_page)
                citation += f" (стор. {p_range})"
            
            sources.append(citation)
        
        # 2. Extract page references from retrieved documents
        for i, doc in enumerate(retrieved_docs, 1):
            # Try to extract page number from format: "(сторінка X)" or "PAGE: X"
            page_match = re.search(r"\(сторінка\s*(\d+)\)", doc)
            if page_match:
                sources.append(f"   • Сторінка {page_match.group(1)}")
                continue
            
            page_match = re.search(r"PAGE:\s*(\d+)", doc)
            if page_match:
                sources.append(f"   • Сторінка {page_match.group(1)}")
                continue
            
            # Try to extract topic from document
            topic_match = re.search(r"TOPIC:\s*([^\n]+)", doc)
            if topic_match:
                topic_name = topic_match.group(1).strip()[:50]
                sources.append(f"   • {topic_name}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)
        
        return unique_sources[:10]  # Limit to 10 sources


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
    
    # Extract topic, retrieved docs, and source_info from matched_topics
    topic = ""
    retrieved_docs = []
    source_info = None
    
    if matched_topics:
        first_match = matched_topics[0]
        if isinstance(first_match, dict):
            topic = first_match.get("topic", "")
            retrieved_docs = first_match.get("retrieved_docs", [])
            source_info = first_match.get("source_info")
        elif isinstance(first_match, str):
            topic = first_match
    
    # Also include matched_pages if available
    matched_pages = state.get("matched_pages", [])
    for page in matched_pages:
        if isinstance(page, dict):
            content = page.get("content", "")
            if content and content not in retrieved_docs:
                retrieved_docs.append(content)
        elif isinstance(page, str) and page not in retrieved_docs:
            retrieved_docs.append(page)
    
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
            personalization_prompt=personalization_prompt,
            source_info=source_info
        )
        
        print(f"[Content Generator] Generated {len(result['lecture_content'])} chars of content")
        print(f"[Content Generator] Created {len(result['control_questions'])} control questions")
        print(f"[Content Generator] Extracted {len(result['sources'])} sources")
        
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
