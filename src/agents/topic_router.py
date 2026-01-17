"""TopicRouter agent - routes teacher queries to relevant topics from TOC.

This agent:
1. Uses Mamay model for understanding context and intent (reasoning)
2. Uses ChromaDB to search the TOC (table of contents) topics
3. Returns matched topic with retrieved documents
"""
from typing import Dict, List, Optional, Any
from collections import Counter
import chromadb
from chromadb.config import Settings

from ..llm.mamay import MamayLLM
from ..llm.embeddings import QwenEmbeddings
from ..config import get_settings


# Subject name to discipline_id mapping
# Based on actual data in ChromaDB collection
SUBJECT_TO_DISCIPLINE_ID = {
    "Історія України": 107,
    "Українська мова": 131,
    "Алгебра": 72,
}


def get_discipline_id(subject: str) -> int:
    """Get discipline ID from subject name.
    
    Args:
        subject: Subject name (e.g., "Історія України")
        
    Returns:
        Discipline ID (defaults to 107 for Історія України if not found)
    """
    return SUBJECT_TO_DISCIPLINE_ID.get(subject, 107)


class TopicRouter:
    """Router that matches teacher queries to topics from the textbook TOC.
    
    Uses:
    - Mamay LLM for reasoning and understanding query intent
    - ChromaDB for semantic search in TOC topics
    - Qwen embeddings for vector similarity
    """
    
    def __init__(self):
        """Initialize TopicRouter with ChromaDB and LLM clients."""
        settings = get_settings()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get TOC topics collection
        try:
            self.toc_collection = self.chroma_client.get_collection("toc_topics")
        except Exception:
            raise ValueError(
                "ChromaDB collection 'toc_topics' not found. "
                "Please ensure the collection is created and indexed."
            )
        
        # Get pages collection (optional - for retrieving page content)
        try:
            self.pages_collection = self.chroma_client.get_collection("pages")
        except Exception:
            # Pages collection is optional - router will work without it
            self.pages_collection = None
        
        # Initialize LLM clients
        self.mamay = MamayLLM()
        self.embeddings = QwenEmbeddings()
    
    def route(
        self,
        query: str,
        grade: Optional[int] = None,
        discipline_id: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Route a teacher query to relevant topics.
        
        If grade and/or discipline_id are not provided, they will be inferred from the query.
        
        Args:
            query: Teacher's query (e.g., "Поясни Руїну")
            grade: Optional grade level (8 or 9). If None, will be inferred.
            discipline_id: Optional global discipline ID. If None, will be inferred.
            top_k: Number of top topics to retrieve
            
        Returns:
            Dict with:
                - "topic": str - Name of the matched theme
                - "retrieved_docs": List[str] - Retrieved documents formatted as "Документ N: ..."
                - "grade": int - Inferred or provided grade
                - "subject": str - Inferred or provided subject name
                - "discipline_id": int - Inferred or provided discipline ID
        """
        # Step 1: Infer grade and subject if not provided
        if grade is None or discipline_id is None:
            inferred = self._infer_grade_and_subject(query)
            if grade is None:
                grade = inferred.get("grade")
            if discipline_id is None:
                discipline_id = inferred.get("discipline_id")
        
        # Step 2: Use Mamay to understand and potentially refine the query
        refined_query = self._refine_query_with_mamay(query, grade, discipline_id)
        
        # Step 3: Search ChromaDB for matching topics
        search_query = refined_query if refined_query else query
        
        # Generate embedding for the query using Qwen
        query_embedding = self.embeddings.embed(search_query)
        
        # Build where clause - only include provided filters
        where_clause = {}
        if grade is not None:
            where_clause["grade"] = grade
        if discipline_id is not None:
            where_clause["global_discipline_id"] = discipline_id
        
        # Query ChromaDB collection using explicit embeddings
        if where_clause:
            if len(where_clause) == 2:
                # Multiple conditions need $and
                where_clause = {
                    "$and": [
                        {"grade": grade},
                        {"global_discipline_id": discipline_id}
                    ]
                }
        else:
            # No filters - search across all topics
            where_clause = None
        
        results = self.toc_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more candidates for better selection
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Step 3: Use Mamay to select the best topic from candidates
        if results["ids"] and len(results["ids"][0]) > 0:
            best_topic = self._select_best_topic_with_mamay(
                query, 
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0]
            )
            
            # Extract topic name from best match
            topic_name = best_topic.get("topic_title")
            if not topic_name:
                # Fallback: try to extract from first document
                first_doc = results["documents"][0][0] if results["documents"][0] else ""
                for line in first_doc.split("\n"):
                    if line.startswith("TOPIC:"):
                        topic_name = line.replace("TOPIC:", "").strip()
                        break
                if not topic_name:
                    topic_name = "Невідома тема"
        else:
            # Fallback if no results
            return {
                "topic": "",
                "retrieved_docs": []
            }
        
        # Step 4: Extract topic_text from the best matching document
        # The document format is: router_text + "\n\n---TOPIC_CONTENT---\n" + topic_text
        best_doc_idx = 0
        for i, meta in enumerate(results["metadatas"][0]):
            if meta.get("book_topic_id") == best_topic.get("book_topic_id"):
                best_doc_idx = i
                break
        
        best_document = results["documents"][0][best_doc_idx] if best_doc_idx < len(results["documents"][0]) else ""
        topic_text = self._extract_topic_text_from_doc(best_document)
        
        # Step 5: Retrieve page content if topic_text is not available
        book_topic_id = best_topic.get("book_topic_id")
        page_content = []
        
        # Only query pages if we don't have topic_text
        if not topic_text and book_topic_id and self.pages_collection:
            page_content = self._retrieve_pages_for_topic(
                book_topic_id=book_topic_id,
                grade=grade,
                discipline_id=discipline_id,
                max_pages=top_k
            )
        
        # Step 6: Format output
        # Priority: topic_text > page_content > document extraction
        if topic_text:
            # Split topic_text into chunks for better presentation
            retrieved_docs = self._format_topic_text(topic_text, top_k)
        elif page_content:
            retrieved_docs = page_content
        else:
            # Fallback to topic documents
            retrieved_docs = []
            for i, doc in enumerate(results["documents"][0][:top_k], 1):
                # Extract meaningful content from document
                doc_text = self._extract_doc_content(doc)
                retrieved_docs.append(f"Документ {i}: {doc_text}")
        
        # Get subject name from best match
        subject_name = best_topic.get("global_discipline_name", "Невідомий предмет")
        inferred_grade = best_topic.get("grade", grade)
        inferred_discipline_id = best_topic.get("global_discipline_id", discipline_id)
        
        # Build source info for grounding
        start_page = best_topic.get("book_page_start") or best_topic.get("start_page")
        end_page = best_topic.get("book_page_end") or best_topic.get("end_page")
        source_info = {
            "topic_title": topic_name,
            "book_id": best_topic.get("book_id"),
            "start_page": start_page,
            "end_page": end_page,
            "subject": subject_name,
            "grade": inferred_grade
        }
        
        return {
            "topic": topic_name,
            "retrieved_docs": retrieved_docs,
            "grade": int(inferred_grade) if inferred_grade else None,
            "subject": subject_name,
            "discipline_id": int(inferred_discipline_id) if inferred_discipline_id else None,
            "source_info": source_info
        }
    
    def _refine_query_with_mamay(
        self, 
        query: str, 
        grade: Optional[int], 
        discipline_id: Optional[int]
    ) -> Optional[str]:
        """Use Mamay to refine/understand the query for better routing.
        
        Args:
            query: Original teacher query
            grade: Grade level
            discipline_id: Discipline ID
            
        Returns:
            Refined query or None if refinement not needed
        """
        system_prompt = """Ти допомагаєш зрозуміти запит вчителя для пошуку теми в підручнику.
Твоя задача - зрозуміти намір вчителя і можливо уточнити запит для кращого пошуку.
Якщо запит вже чіткий, поверни його без змін."""
        
        grade_info = f"Клас: {grade}" if grade else "Клас: не вказано"
        subject_info = f"Предмет ID: {discipline_id}" if discipline_id else "Предмет: не вказано"
        
        prompt = f"""Запит вчителя: "{query}"
{grade_info}
{subject_info}

Проаналізуй запит і, якщо потрібно, уточни його для кращого пошуку теми в підручнику.
Поверни тільки уточнений запит без додаткових пояснень."""
        
        try:
            refined = self.mamay.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=200
            )
            # Clean up the response
            refined = refined.strip().strip('"').strip("'")
            return refined if refined and refined != query else None
        except Exception:
            # If Mamay fails, return None to use original query
            return None
    
    def _select_best_topic_with_mamay(
        self,
        original_query: str,
        topic_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use Mamay to select the best matching topic from candidates.
        
        Args:
            original_query: Original teacher query
            topic_ids: List of topic IDs from ChromaDB
            documents: List of document texts
            metadatas: List of metadata dicts
            
        Returns:
            Best matching topic metadata dict
        """
        # Build candidate list for Mamay
        candidates_text = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            topic_title = meta.get("topic_title", "Невідома тема")
            section_title = meta.get("section_title", "")
            summary = meta.get("topic_summary", "")
            
            candidates_text.append(
                f"Кандидат {i}:\n"
                f"  Тема: {topic_title}\n"
                f"  Розділ: {section_title}\n"
                f"  Опис: {summary}\n"
                f"  Документ: {doc[:300]}..."
            )
        
        system_prompt = """Ти допомагаєш вибрати найбільш відповідну тему з підручника для запиту вчителя.
Проаналізуй кандидатів і вибери найкращий."""
        
        prompt = f"""Запит вчителя: "{original_query}"

Кандидати тем з підручника:
{chr(10).join(candidates_text)}

Проаналізуй кожного кандидата і вибери найбільш відповідний запиту вчителя.
Поверни тільки номер кандидата (1, 2, 3, ...) без додаткових пояснень."""
        
        try:
            response = self.mamay.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=50
            )
            
            # Extract number from response
            selected_idx = 0  # Default to first
            for char in response:
                if char.isdigit():
                    selected_idx = int(char) - 1
                    if 0 <= selected_idx < len(metadatas):
                        break
            
            return metadatas[selected_idx]
        except Exception:
            # If Mamay fails, return first candidate
            return metadatas[0] if metadatas else {}
    
    def _infer_grade_and_subject(self, query: str) -> Dict[str, Any]:
        """Infer grade and subject from query using semantic search and Mamay.
        
        Args:
            query: Teacher's query
            
        Returns:
            Dict with "grade" and "discipline_id" (or None if cannot be determined)
        """
        # Step 1: Do a broad semantic search across all topics
        query_embedding = self.embeddings.embed(query)
        
        # Get top candidates from all subjects and grades
        results = self.toc_collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Get top 10 candidates
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or len(results["ids"][0]) == 0:
            # Fallback: return None for both
            return {"grade": None, "discipline_id": None}
        
        # Step 2: Use Mamay to analyze candidates and determine subject/grade
        candidates_info = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0][:5], results["metadatas"][0][:5]), 1):
            topic_title = meta.get("topic_title", "Невідома тема")
            discipline_name = meta.get("global_discipline_name", "Невідомий предмет")
            grade = meta.get("grade")
            
            # Extract summary
            summary = ""
            for line in doc.split("\n"):
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()[:200]
                    break
            
            candidates_info.append(
                f"Кандидат {i}:\n"
                f"  Предмет: {discipline_name}\n"
                f"  Клас: {grade}\n"
                f"  Тема: {topic_title}\n"
                f"  Опис: {summary}"
            )
        
        system_prompt = """Ти допомагаєш визначити предмет та клас для запиту вчителя.
Проаналізуй запит та кандидатів і визнач найбільш відповідний предмет та клас."""
        
        prompt = f"""Запит вчителя: "{query}"

Кандидати тем з різних предметів:
{chr(10).join(candidates_info)}

Визнач:
1. Який предмет найбільш відповідає запиту? (Алгебра, Історія України, або Українська мова)
2. Який клас найбільш відповідає? (8 або 9)

Поверни відповідь у форматі:
ПРЕДМЕТ: [назва предмету]
КЛАС: [номер класу]"""
        
        try:
            response = self.mamay.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse response
            inferred_grade = None
            inferred_discipline_id = None
            
            # Extract subject
            subject_name = None
            for line in response.split("\n"):
                if "ПРЕДМЕТ:" in line.upper():
                    subject_name = line.split(":", 1)[1].strip() if ":" in line else None
                elif "КЛАС:" in line.upper():
                    grade_str = line.split(":", 1)[1].strip() if ":" in line else None
                    if grade_str:
                        try:
                            inferred_grade = int(grade_str)
                        except ValueError:
                            pass
            
            # Map subject name to discipline_id
            if subject_name:
                inferred_discipline_id = get_discipline_id(subject_name)
            
            # Fallback: use most common from top candidates
            if inferred_grade is None or inferred_discipline_id is None:
                from collections import Counter
                grades = [m.get("grade") for m in results["metadatas"][0][:5] if m.get("grade")]
                disciplines = [m.get("global_discipline_id") for m in results["metadatas"][0][:5] if m.get("global_discipline_id")]
                
                if inferred_grade is None and grades:
                    inferred_grade = Counter(grades).most_common(1)[0][0]
                if inferred_discipline_id is None and disciplines:
                    inferred_discipline_id = Counter(disciplines).most_common(1)[0][0]
            
            return {
                "grade": int(inferred_grade) if inferred_grade else None,
                "discipline_id": int(inferred_discipline_id) if inferred_discipline_id else None
            }
            
        except Exception as e:
            # Fallback: use most common from top candidates
            from collections import Counter
            grades = [m.get("grade") for m in results["metadatas"][0][:5] if m.get("grade")]
            disciplines = [m.get("global_discipline_id") for m in results["metadatas"][0][:5] if m.get("global_discipline_id")]
            
            inferred_grade = Counter(grades).most_common(1)[0][0] if grades else None
            inferred_discipline_id = Counter(disciplines).most_common(1)[0][0] if disciplines else None
            
            return {
                "grade": int(inferred_grade) if inferred_grade else None,
                "discipline_id": int(inferred_discipline_id) if inferred_discipline_id else None
            }
    
    def _retrieve_pages_for_topic(
        self,
        book_topic_id: str,
        grade: Optional[int] = None,
        discipline_id: Optional[int] = None,
        max_pages: int = 5
    ) -> List[str]:
        """Retrieve page content for a specific topic.
        
        Args:
            book_topic_id: The book_topic_id to search for
            grade: Optional grade filter
            discipline_id: Optional discipline ID filter
            max_pages: Maximum number of pages to retrieve
            
        Returns:
            List of formatted page documents
        """
        if not self.pages_collection:
            return []
        
        # Build where clause
        where_clause = {"book_topic_id": str(book_topic_id)}
        
        if grade is not None:
            where_clause["grade"] = grade
        if discipline_id is not None:
            where_clause["global_discipline_id"] = discipline_id
        
        # If multiple conditions, use $and
        if len(where_clause) > 1:
            conditions = [{"book_topic_id": str(book_topic_id)}]
            if grade is not None:
                conditions.append({"grade": grade})
            if discipline_id is not None:
                conditions.append({"global_discipline_id": discipline_id})
            where_clause = {"$and": conditions}
        
        try:
            # Query pages by book_topic_id
            results = self.pages_collection.get(
                where=where_clause,
                limit=max_pages * 2  # Get more to sort by page number
            )
            
            if not results["ids"] or len(results["ids"]) == 0:
                return []
            
            # Sort by page number
            pages_with_nums = []
            for i, meta in enumerate(results["metadatas"]):
                page_num = meta.get("book_page_number") or meta.get("page_number", 0)
                pages_with_nums.append({
                    "page_num": int(page_num) if page_num else 0,
                    "doc": results["documents"][i],
                    "meta": meta
                })
            
            # Sort by page number
            pages_with_nums.sort(key=lambda x: x["page_num"])
            
            # Format pages
            retrieved_docs = []
            for i, page_data in enumerate(pages_with_nums[:max_pages], 1):
                page_text = page_data["doc"]
                page_num = page_data["page_num"]
                
                # Limit page text length
                if len(page_text) > 1000:
                    page_text = page_text[:1000] + "..."
                
                retrieved_docs.append(
                    f"Документ {i} (сторінка {page_num}): {page_text}"
                )
            
            return retrieved_docs
            
        except Exception as e:
            # If query fails, return empty list
            print(f"Warning: Failed to retrieve pages for topic {book_topic_id}: {e}")
            return []
    
    def _extract_topic_text_from_doc(self, doc: str) -> str:
        """Extract full topic_text from ChromaDB document.
        
        The document format is: router_text + "\n\n---TOPIC_CONTENT---\n" + topic_text
        
        Args:
            doc: Full document text from ChromaDB
            
        Returns:
            Full topic_text if available, empty string otherwise
        """
        if "---TOPIC_CONTENT---" in doc:
            parts = doc.split("---TOPIC_CONTENT---", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return ""
    
    def _format_topic_text(self, topic_text: str, max_chunks: int = 5) -> List[str]:
        """Format topic_text into multiple document chunks.
        
        Args:
            topic_text: Full topic text content
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of formatted document strings
        """
        if not topic_text:
            return []
        
        # Split by paragraphs (double newlines) or sections
        chunks = []
        
        # Try splitting by double newlines first (paragraphs)
        paragraphs = [p.strip() for p in topic_text.split("\n\n") if p.strip()]
        
        if len(paragraphs) > 1:
            # Group paragraphs into chunks
            chunk_size = max(1, len(paragraphs) // max_chunks)
            for i in range(0, len(paragraphs), chunk_size):
                chunk = "\n\n".join(paragraphs[i:i+chunk_size])
                if chunk:
                    chunks.append(chunk)
        else:
            # Single paragraph or no clear separation - split by length
            chunk_length = max(1000, len(topic_text) // max_chunks)
            for i in range(0, len(topic_text), chunk_length):
                chunk = topic_text[i:i+chunk_length].strip()
                if chunk:
                    chunks.append(chunk)
        
        # Limit to max_chunks
        chunks = chunks[:max_chunks]
        
        # Format as documents
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            # Limit each chunk to reasonable length
            if len(chunk) > 2000:
                chunk = chunk[:2000] + "..."
            formatted.append(f"Документ {i}: {chunk}")
        
        return formatted if formatted else [f"Документ 1: {topic_text[:2000]}{'...' if len(topic_text) > 2000 else ''}"]
    
    def _extract_doc_content(self, doc: str) -> str:
        """Extract meaningful content from ChromaDB document.
        
        Args:
            doc: Full document text from ChromaDB
            
        Returns:
            Extracted content (first 500 chars or summary)
        """
        # First try to extract topic_text if available
        topic_text = self._extract_topic_text_from_doc(doc)
        if topic_text:
            return topic_text[:500] + "..." if len(topic_text) > 500 else topic_text
        
        # Document format from notebook: "SECTION: ...\nTOPIC: ...\nSUMMARY: ...\nTEXT: ..."
        # Extract the most relevant parts
        lines = doc.split("\n")
        content_parts = []
        
        for line in lines:
            if line.startswith("TOPIC:") or line.startswith("SUMMARY:") or line.startswith("TEXT:"):
                content_parts.append(line)
        
        if content_parts:
            result = " ".join(content_parts)
            # Limit length
            if len(result) > 500:
                result = result[:500] + "..."
            return result
        
        # Fallback: return first 500 chars
        return doc[:500] + "..." if len(doc) > 500 else doc

