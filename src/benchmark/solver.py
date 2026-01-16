"""Benchmark solver for lms_questions.

Solves multiple-choice questions using MamayLLM with optional RAG context retrieval.
Designed for CodaBench evaluation pipeline.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
from flashrank import RerankRequest

from ..config import get_settings
from ..llm import MamayLLM, LapaLLM
from ..data.vector_store import get_vector_store
from ..llm import QwenEmbeddings


@dataclass
class BenchmarkResult:
    """Result for a single benchmark question."""
    question_id: str
    predicted_answer_index: int
    predicted_answer_text: str
    correct_answer_index: Optional[int] = None
    is_correct: Optional[bool] = None
    reasoning: str = ""


class BenchmarkSolver:
    """Solver for benchmark questions using MamayLLM or LapaLLM.
    
    Supports:
    - Single question solving
    - Batch solving with progress tracking
    - RAG context injection (when vector store available)
    - Result export for CodaBench
    - Model selection (mamay or lapa)
    """
    
    SUPPORTED_MODELS = ["mamay", "lapa"]
    SUPPORTED_RETRIEVAL = ["baseline", "two_stage", "hyde", "hybrid", "hyde_two_stage", "hyde_hybrid"]
    
    def __init__(self, use_rag: bool = False, model: str = "mamay", retrieval_strategy: str = "baseline", concise: bool = False):
        """Initialize the solver.
        
        Args:
            use_rag: Whether to use RAG for context retrieval (requires vector store)
            model: LLM to use - 'mamay' or 'lapa'
            retrieval_strategy: 'baseline', 'two_stage', 'hyde', or 'hybrid' (BM25 + semantic + rerank)
            concise: If True, LLM outputs only answer letter (faster, fewer tokens)
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {self.SUPPORTED_MODELS}, got: {model}")
        if retrieval_strategy not in self.SUPPORTED_RETRIEVAL:
            raise ValueError(f"Retrieval must be one of {self.SUPPORTED_RETRIEVAL}, got: {retrieval_strategy}")
        
        self.model_name = model
        self.retrieval_strategy = retrieval_strategy
        self.concise = concise
        if model == "mamay":
            self.llm = MamayLLM()
        else:
            self.llm = LapaLLM()
        
        self.use_rag = use_rag
        self.settings = get_settings()
        
        # Vector store will be initialized lazily if use_rag is True
        self._vector_store = get_vector_store()
        self._embeddings = QwenEmbeddings()
        
        # Initialize FlashRank reranker for hybrid/hyde_hybrid retrieval
        self._reranker = None
        if retrieval_strategy in ["hybrid", "hyde_hybrid"]:
            from flashrank import Ranker
            self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    
    def _get_context(self, question_text: str, subject: str, grade: int) -> str:
        """Retrieve relevant context from knowledge base.
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string or empty if no RAG
        """
        if not self.use_rag:
            return ""
        
        # Lazy load vector store and embeddings (uses persisted ChromaDB)
        # if self._vector_store is None:
        #     try:
        #         # Check if indexed, if not - index now
        #         if not self._vector_store.is_indexed:
        #             print("Vector store not indexed, indexing now...")
        #             self._vector_store.index_all()
        #         else:
        #             print(f"✓ Loaded vector store ({self._vector_store.pages_collection.count()} pages, {self._vector_store.topics_collection.count()} topics)")
                    
        #     except Exception as e:
        #         print(f"Warning: Could not load vector store: {e}")
        #         self.use_rag = False
        #         return ""
        
        # Get query embedding
        query_embedding = self._embeddings.embed(question_text)
        
        # Search for relevant pages
        results = self._vector_store.search_pages(
            query_embedding=query_embedding,
            top_k=3,
            filter_dict={"grade": grade, "subject": subject}
        )
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return ""
        
        # Combine top results into context with source info
        context_parts = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            source_info = f"[{meta.get('book_name', 'Підручник')}, стор. {meta.get('page_number', '?')}]"
            context_parts.append(f"{source_info}\n{doc[:1500]}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_context_two_stage(self, question_text: str, subject: str, grade: int) -> str:
        """Two-stage retrieval: TOC first, then pages within matched topics.
        
        Stage 1: Search topics (237) using section_topic_embedding
        Stage 2: Get pages from matched topics, prioritize theory
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string with topic summary + relevant pages
        """
        if not self.use_rag:
            return ""
        
        # Get query embedding
        query_embedding = self._embeddings.embed(question_text)
        
        # ===== STAGE 1: Topic Retrieval =====
        topic_results = self._vector_store.search_topics(
            query_embedding=query_embedding,
            top_k=2,  # Get top 2 topics
            filter_dict={"grade": grade, "subject": subject}
        )
        
        if not topic_results or not topic_results.get("documents") or not topic_results["documents"][0]:
            # Fallback to baseline if no topics found
            return self._get_context(question_text, subject, grade)
        
        context_parts = []
        matched_topic_ids = []
        
        # Extract topic info and collect IDs
        for doc, meta in zip(topic_results["documents"][0], topic_results["metadatas"][0]):
            topic_title = meta.get("topic_title", "")
            section_title = meta.get("section_title", "")
            subtopics = meta.get("subtopics", "")
            
            # Add topic summary to context
            topic_context = f"📚 Тема: {topic_title}"
            if section_title:
                topic_context += f"\n📖 Розділ: {section_title}"
            if subtopics:
                topic_context += f"\n📝 Підтеми: {subtopics}"
            
            # Add topic text (first 800 chars as summary)
            if doc:
                topic_context += f"\n\n{doc[:800]}"
            
            context_parts.append(topic_context)
            
            # Collect topic title for page filtering
            if topic_title:
                matched_topic_ids.append(topic_title)
        
        # ===== STAGE 2: Page Retrieval within matched topics =====
        if matched_topic_ids:
            page_results = self._vector_store.search_pages(
                query_embedding=query_embedding,
                top_k=6,  # Get more, then filter
                filter_dict={"grade": grade, "subject": subject}
            )
            
            if page_results and page_results.get("documents") and page_results["documents"][0]:
                page_parts = []
                for doc, meta in zip(page_results["documents"][0], page_results["metadatas"][0]):
                    page_topic = meta.get("topic_title", "")
                    
                    # Prioritize pages from matched topics
                    if page_topic in matched_topic_ids:
                        source_info = f"[{meta.get('book_name', 'Підручник')}📄 стор. {meta.get('page_number', '?')}]"
                        page_parts.append(f"{source_info}\n{doc[:1000]}")
                        
                        if len(page_parts) >= 2:  # Max 2 pages from matched topics
                            break
                
                # If no pages from matched topics, take first 2 from results
                if not page_parts:
                    for doc, meta in zip(page_results["documents"][0][:2], page_results["metadatas"][0][:2]):
                        source_info = f"[{meta.get('book_name', 'Підручник')}📄 стор. {meta.get('page_number', '?')}]"
                        page_parts.append(f"{source_info}\n{doc[:1000]}")
                
                if page_parts:
                    context_parts.append("\n--- Сторінки підручника ---\n" + "\n\n".join(page_parts))
        
        return "\n\n".join(context_parts)
    
    def _get_context_hyde(self, question_text: str, subject: str, grade: int) -> str:
        """HyDE (Hypothetical Document Embedding) retrieval.
        
        1. Generate a hypothetical document that would contain the answer
        2. Embed that hypothetical document
        3. Search for real documents similar to the hypothetical
        
        This often retrieves more relevant content because we search for
        documents similar to what the answer SHOULD look like.
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string from retrieved documents
        """
        if not self.use_rag:
            return ""
        
        # Step 1: Generate hypothetical answer document
        hyde_prompt = f"""Предмет: {subject}
Клас: {grade}

Питання: {question_text}

Напиши короткий параграф (3-5 речень) з підручника, який містить інформацію для відповіді на це питання. 
Пиши як автор підручника, використовуй чіткі визначення та приклади."""

        hypothetical_doc = self.llm.generate(hyde_prompt, temperature=0.3, max_tokens=200)
        
        # Step 2: Embed the hypothetical document (not the question!)
        hyde_embedding = self._embeddings.embed(hypothetical_doc)
        
        # Step 3: Search using the hypothetical document embedding
        page_results = self._vector_store.search_pages(
            query_embedding=hyde_embedding,
            top_k=5,  # Get top 5 most similar real documents
            filter_dict={"grade": grade, "subject": subject}
        )
        
        if not page_results or not page_results.get("documents") or not page_results["documents"][0]:
            # Fallback to baseline if no results
            return self._get_context(question_text, subject, grade)
        
        # Format context
        context_parts = []
        for doc, meta in zip(page_results["documents"][0], page_results["metadatas"][0]):
            source_info = f"[{meta.get('book_name', 'Підручник')}, стор. {meta.get('page_number', '?')}]"
            context_parts.append(f"{source_info}\n{doc[:1500]}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_context_hybrid(self, question_text: str, subject: str, grade: int) -> str:
        """Hybrid retrieval: BM25 + Semantic search with FlashRank reranking.
        
        1. Get top candidates from BM25 (keyword matching)
        2. Get top candidates from semantic search (meaning matching)
        3. Union the results and remove duplicates
        4. Rerank all candidates using FlashRank cross-encoder
        5. Return top 5 after reranking
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string from reranked documents
        """
        if not self.use_rag:
            return ""
        
        from rank_bm25 import BM25Okapi
        
        # Load pages data for BM25
        from ..data.loader import load_pages
        pages_df = load_pages()
        
        # Filter pages by grade and subject
        filtered_df = pages_df[
            (pages_df["grade"] == grade) & 
            (pages_df["global_discipline_name"] == subject)
        ].copy()
        
        if filtered_df.empty:
            return self._get_context(question_text, subject, grade)
        
        # ===== BM25 Search =====
        # Tokenize documents (simple whitespace tokenization)
        corpus = filtered_df["page_text"].fillna("").tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top 15 from BM25
        bm25_top_indices = bm25_scores.argsort()[-15:][::-1]
        bm25_docs = [(filtered_df.iloc[i], corpus[i]) for i in bm25_top_indices if bm25_scores[i] > 0]
        
        # ===== Semantic Search =====
        query_embedding = self._embeddings.embed(question_text)
        semantic_results = self._vector_store.search_pages(
            query_embedding=query_embedding,
            top_k=15,
            filter_dict={"grade": grade, "subject": subject}
        )
        
        # Collect semantic docs
        semantic_docs = []
        if semantic_results and semantic_results.get("documents") and semantic_results["documents"][0]:
            for doc, meta in zip(semantic_results["documents"][0], semantic_results["metadatas"][0]):
                semantic_docs.append((meta, doc))
        
        # ===== Union and Deduplicate =====
        all_docs = {}  # Use text hash as key to dedupe
        
        for meta, text in bm25_docs:
            text_hash = hash(text[:200])
            if text_hash not in all_docs:
                all_docs[text_hash] = {
                    "text": text,
                    "meta": {
                        "book_name": meta.get("book_name", "Підручник"),
                        "page_number": meta.get("book_page_number", "?")
                    }
                }
        
        for meta, text in semantic_docs:
            text_hash = hash(text[:200])
            if text_hash not in all_docs:
                all_docs[text_hash] = {
                    "text": text,
                    "meta": meta
                }
        
        if not all_docs:
            return self._get_context(question_text, subject, grade)
        
        # ===== FlashRank Reranking =====
        
        # Prepare passages for reranking
        passages = [{"id": k, "text": v["text"][:2000]} for k, v in all_docs.items()]
        
        rerank_request = RerankRequest(query=question_text, passages=passages)
        reranked = self._reranker.rerank(rerank_request)
        
        # Get top 5 after reranking
        context_parts = []
        for result in reranked[:5]:
            doc_id = result["id"]
            doc_info = all_docs[doc_id]
            meta = doc_info["meta"]
            text = doc_info["text"]
            
            source_info = f"[{meta.get('book_name', 'Підручник')}, стор. {meta.get('page_number', '?')}]"
            context_parts.append(f"{source_info}\n{text[:1500]}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_context_hyde_two_stage(self, question_text: str, subject: str, grade: int) -> str:
        """HyDE + Two-Stage: Generate hypothetical topic AND doc for better matching.
        
        1. Generate hypothetical topic title → embed → search topics
        2. Generate hypothetical answer doc → embed → search pages in matched topics
        
        This uses topic-style embedding for topic search and
        content-style embedding for page search.
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string from topic-matched pages
        """
        if not self.use_rag:
            return ""
        
        import json as json_lib
        
        # Generate both hypothetical topic and doc in ONE call
        prompt = f"""Предмет: {subject}
Клас: {grade}

Питання: {question_text}

Згенеруй JSON з двома полями:
1. "topic" - назва теми з підручника, яка найкраще відповідає питанню (коротко, 5-15 слів)
2. "doc" - параграф з підручника (3-5 речень), який містить відповідь на питання

Відповідай ТІЛЬКИ валідним JSON, без додаткового тексту:
{{"topic": "...", "doc": "..."}}"""

        response = self.llm.generate(prompt, temperature=0.3, max_tokens=300)
        
        # Parse JSON response
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json_lib.loads(response[start:end])
                hypothetical_topic = data.get("topic", "")
                hypothetical_doc = data.get("doc", "")
            else:
                print("HyDE Two-Stage: FALLBACK 1!")
                # Fallback if JSON parsing fails
                hypothetical_topic = response[:100]
                hypothetical_doc = response
        except:
            print("HyDE Two-Stage: FALLBACK 2!")
            hypothetical_topic = response[:100]
            hypothetical_doc = response
        
        # Embed both
        topic_embedding = self._embeddings.embed(hypothetical_topic)
        doc_embedding = self._embeddings.embed(hypothetical_doc)
        
        # Step 3: Search topics with TOPIC embedding
        topic_results = self._vector_store.search_topics(
            query_embedding=topic_embedding,
            top_k=3,
            filter_dict={"grade": grade, "subject": subject}
        )
        
        if not topic_results or not topic_results.get("documents") or not topic_results["documents"][0]:
            return self._get_context_hyde(question_text, subject, grade)
        
        context_parts = []
        matched_topic_ids = []
        
        # Extract topic info
        for doc, meta in zip(topic_results["documents"][0], topic_results["metadatas"][0]):
            topic_title = meta.get("topic_title", "")
            section_title = meta.get("section_title", "")
            subtopics = meta.get("subtopics", "")
            
            topic_context = f"📚 Тема: {topic_title}"
            if section_title:
                topic_context += f"\n📖 Розділ: {section_title}"
            if subtopics:
                topic_context += f"\n📝 Підтеми: {subtopics}"
            if doc:
                topic_context += f"\n\n{doc[:800]}"
            
            context_parts.append(topic_context)
            if topic_title:
                matched_topic_ids.append(topic_title)
        
        # Step 4: Search pages with DOC embedding (within matched topics)
        if matched_topic_ids:
            page_results = self._vector_store.search_pages(
                query_embedding=doc_embedding,  # Use doc embedding for pages!
                top_k=8,
                filter_dict={"grade": grade, "subject": subject}
            )
            
            if page_results and page_results.get("documents") and page_results["documents"][0]:
                page_parts = []
                for doc, meta in zip(page_results["documents"][0], page_results["metadatas"][0]):
                    page_topic = meta.get("topic_title", "")
                    if page_topic in matched_topic_ids:
                        source_info = f"[{meta.get('book_name', 'Підручник')}📄 стор. {meta.get('page_number', '?')}]"
                        page_parts.append(f"{source_info}\n{doc[:1200]}")
                        if len(page_parts) >= 5:
                            break
                
                if not page_parts:
                    for doc, meta in zip(page_results["documents"][0][:5], page_results["metadatas"][0][:5]):
                        source_info = f"[{meta.get('book_name', 'Підручник')}📄 стор. {meta.get('page_number', '?')}]"
                        page_parts.append(f"{source_info}\n{doc[:1200]}")
                
                if page_parts:
                    context_parts.append("\n--- Сторінки підручника ---\n" + "\n\n".join(page_parts))
        
        return "\n\n".join(context_parts)
    
    def _get_context_hyde_hybrid(self, question_text: str, subject: str, grade: int) -> str:
        """HyDE + Hybrid: BM25 (keywords) + Semantic (HyDE embedding) + Rerank.
        
        1. Generate hypothetical document
        2. BM25 search with original question keywords
        3. Semantic search with HyDE embedding
        4. Union + deduplicate
        5. Rerank with FlashRank
        
        Args:
            question_text: The question to get context for
            subject: Subject name for filtering
            grade: Grade level for filtering
            
        Returns:
            Context string from reranked documents
        """
        if not self.use_rag:
            return ""
        
        from rank_bm25 import BM25Okapi
        from ..data.loader import load_pages
        
        # Step 1: Generate hypothetical document
        hyde_prompt = f"""Предмет: {subject}
Клас: {grade}

Питання: {question_text}

Напиши короткий параграф (3-5 речень) з підручника, який містить інформацію для відповіді на це питання."""

        hypothetical_doc = self.llm.generate(hyde_prompt, temperature=0.3, max_tokens=200)
        hyde_embedding = self._embeddings.embed(hypothetical_doc)
        
        # Step 2: BM25 search with QUESTION keywords
        pages_df = load_pages()
        filtered_df = pages_df[
            (pages_df["grade"] == grade) & 
            (pages_df["global_discipline_name"] == subject)
        ].copy()
        
        if filtered_df.empty:
            return self._get_context_hyde(question_text, subject, grade)
        
        corpus = filtered_df["page_text"].fillna("").tolist()
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        bm25_top_indices = bm25_scores.argsort()[-15:][::-1]
        bm25_docs = [(filtered_df.iloc[i], corpus[i]) for i in bm25_top_indices if bm25_scores[i] > 0]
        
        # Step 3: Semantic search with HYDE embedding
        semantic_results = self._vector_store.search_pages(
            query_embedding=hyde_embedding,
            top_k=15,
            filter_dict={"grade": grade, "subject": subject}
        )
        
        semantic_docs = []
        if semantic_results and semantic_results.get("documents") and semantic_results["documents"][0]:
            for doc, meta in zip(semantic_results["documents"][0], semantic_results["metadatas"][0]):
                semantic_docs.append((meta, doc))
        
        # Step 4: Union and deduplicate
        all_docs = {}
        
        for meta, text in bm25_docs:
            text_hash = hash(text[:200])
            if text_hash not in all_docs:
                all_docs[text_hash] = {
                    "text": text,
                    "meta": {
                        "book_name": meta.get("book_name", "Підручник"),
                        "page_number": meta.get("book_page_number", "?")
                    }
                }
        
        for meta, text in semantic_docs:
            text_hash = hash(text[:200])
            if text_hash not in all_docs:
                all_docs[text_hash] = {"text": text, "meta": meta}
        
        if not all_docs:
            return self._get_context_hyde(question_text, subject, grade)
        
        # Step 5: Rerank with FlashRank
        passages = [{"id": k, "text": v["text"][:2000]} for k, v in all_docs.items()]
        rerank_request = RerankRequest(query=question_text, passages=passages)
        reranked = self._reranker.rerank(rerank_request)
        
        # Get top 5
        context_parts = []
        for result in reranked[:5]:
            doc_id = result["id"]
            doc_info = all_docs[doc_id]
            meta = doc_info["meta"]
            text = doc_info["text"]
            
            source_info = f"[{meta.get('book_name', 'Підручник')}, стор. {meta.get('page_number', '?')}]"
            context_parts.append(f"{source_info}\n{text[:1500]}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _solve_with_llm(
        self, 
        question_text: str, 
        answers: List[str], 
        subject: str
    ) -> dict:
        """Solve question using the selected LLM.
        
        Both MamayLLM and LapaLLM have unified solve_question interface.
        """
        return self.llm.solve_question(
            question_text=question_text,
            answers=answers,
            subject=subject,
            concise=self.concise
        )
    
    def solve_question(
        self,
        question_id: str,
        question_text: str,
        answers: List[str],
        subject: str = "Загальний",
        grade: int = 9,
        correct_answer_index: Optional[int] = None
    ) -> BenchmarkResult:
        """Solve a single benchmark question.
        
        Args:
            question_id: Unique question identifier
            question_text: The question text
            answers: List of answer options
            subject: Subject name
            grade: Grade level
            correct_answer_index: Optional correct answer for evaluation
            
        Returns:
            BenchmarkResult with prediction and optional correctness check
        """
        # Get context based on retrieval strategy
        if self.retrieval_strategy == "two_stage":
            context = self._get_context_two_stage(question_text, subject, grade)
        elif self.retrieval_strategy == "hyde":
            context = self._get_context_hyde(question_text, subject, grade)
        elif self.retrieval_strategy == "hybrid":
            context = self._get_context_hybrid(question_text, subject, grade)
        elif self.retrieval_strategy == "hyde_two_stage":
            context = self._get_context_hyde_two_stage(question_text, subject, grade)
        elif self.retrieval_strategy == "hyde_hybrid":
            context = self._get_context_hyde_hybrid(question_text, subject, grade)
        else:
            context = self._get_context(question_text, subject, grade)
        
        # Build prompt with optional context
        if context:
            full_question = f"""Контекст з підручників:
{context}

{question_text}"""
        else:
            full_question = question_text
        
        # Solve using selected LLM
        result = self._solve_with_llm(
            question_text=full_question,
            answers=answers,
            subject=subject
        )
        
        # Create result
        benchmark_result = BenchmarkResult(
            question_id=question_id,
            predicted_answer_index=result["answer_index"],
            predicted_answer_text=result["answer_text"],
            correct_answer_index=correct_answer_index,
            reasoning=result["reasoning"]
        )
        
        # Check correctness if ground truth provided
        if correct_answer_index is not None:
            benchmark_result.is_correct = (
                result["answer_index"] == correct_answer_index
            )
        
        return benchmark_result
    
    def solve_batch(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[BenchmarkResult]:
        """Solve a batch of questions.
        
        Args:
            questions: List of question dicts with keys:
                - question_id, question_text, answers, subject, grade
                - optionally: correct_answer_indices
            show_progress: Whether to show progress bar
            
        Returns:
            List of BenchmarkResult
        """
        results = []
        iterator = tqdm(questions, desc="Solving questions") if show_progress else questions
        
        for q in iterator:
            # Get correct answer index if available
            correct_idx = None
            if "correct_answer_indices" in q and q["correct_answer_indices"]:
                correct_idx = q["correct_answer_indices"][0]
            
            result = self.solve_question(
                question_id=q["question_id"],
                question_text=q["question_text"],
                answers=q["answers"],
                subject=q.get("global_discipline_name", "Загальний"),
                grade=q.get("grade", 9),
                correct_answer_index=correct_idx
            )
            results.append(result)
        
        return results
    
    def solve_from_parquet(
        self,
        parquet_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """Solve questions from parquet file and optionally save results.
        
        Args:
            parquet_path: Path to questions parquet (default: from settings)
            output_path: Optional path to save results JSON
            evaluate: Whether to calculate accuracy metrics
            
        Returns:
            Dict with results and optional metrics
        """
        if parquet_path is None:
            parquet_path = self.settings.questions_parquet_path
        
        print(f"Loading questions from {parquet_path}")

        df = pd.read_parquet(parquet_path)
        
        # Take 50% of the dataset with fixed random state for reproducibility
        #questions = df.sample(frac=0.5, random_state=42).to_dict("records")
        questions = df.to_dict("records")
        print(f"Loaded {len(questions)} questions")
        
        # Solve all questions
        results = self.solve_batch(questions)
        
        # Prepare output
        output = {
            "results": [asdict(r) for r in results],
            "total_questions": len(results)
        }
        
        # Calculate metrics if evaluating
        if evaluate:
            correct = sum(1 for r in results if r.is_correct)
            output["metrics"] = {
                "accuracy": correct / len(results) if results else 0,
                "correct": correct,
                "total": len(results)
            }
            print(f"\nAccuracy: {output['metrics']['accuracy']:.2%} ({correct}/{len(results)})")
            
            # Per-subject breakdown
            subject_results = {}
            for r, q in zip(results, questions):
                subj = q.get("global_discipline_name", "Unknown")
                if subj not in subject_results:
                    subject_results[subj] = {"correct": 0, "total": 0}
                subject_results[subj]["total"] += 1
                if r.is_correct:
                    subject_results[subj]["correct"] += 1
            
            output["metrics"]["by_subject"] = {
                k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
                for k, v in subject_results.items()
            }
            
            print("\nBy subject:")
            for subj, m in output["metrics"]["by_subject"].items():
                print(f"  {subj}: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")
        
        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else x)
            print(f"\nResults saved to {output_path}")
        
        return output
    
    def export_for_codabench(
        self,
        results: List[BenchmarkResult],
        output_path: Path
    ) -> None:
        """Export results in CodaBench submission format.
        
        Args:
            results: List of BenchmarkResult
            output_path: Path to save submission file
        """
        # Format: question_id -> answer_letter
        submission = {}
        for r in results:
            answer_letter = chr(65 + r.predicted_answer_index)  # 0->A, 1->B, etc.
            submission[r.question_id] = answer_letter
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)
        
        print(f"CodaBench submission saved to {output_path}")


def main():
    """CLI entry point for benchmark solving."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve benchmark questions")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input parquet file (default: from config)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--codabench-output",
        type=Path,
        help="Output file for CodaBench submission"
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Use RAG for context retrieval"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation (no accuracy calculation)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["mamay", "lapa"],
        default="mamay",
        help="LLM model to use for solving (default: mamay)"
    )
    parser.add_argument(
        "--retrieval", "-r",
        type=str,
        choices=["baseline", "two_stage", "hyde", "hybrid", "hyde_two_stage", "hyde_hybrid"],
        default="baseline",
        help="Retrieval: baseline, two_stage, hyde, hybrid, hyde_two_stage, hyde_hybrid"
    )
    parser.add_argument(
        "--concise", "-c",
        action="store_true",
        help="Output only answer letter (faster, fewer tokens)"
    )
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}, retrieval: {args.retrieval}, concise: {args.concise}")
    
    # Create solver
    solver = BenchmarkSolver(
        use_rag=args.use_rag, 
        model=args.model, 
        retrieval_strategy=args.retrieval,
        concise=args.concise
    )
    
    # Solve questions
    output = solver.solve_from_parquet(
        parquet_path=args.input,
        output_path=args.output,
        evaluate=not args.no_eval
    )
    
    # Export for CodaBench if requested
    if args.codabench_output:
        results = [BenchmarkResult(**r) for r in output["results"]]
        solver.export_for_codabench(results, args.codabench_output)


if __name__ == "__main__":
    main()
