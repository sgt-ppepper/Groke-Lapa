"""ChromaDB vector store with indexing from parquet files."""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from ..config import get_settings
from .loader import load_pages, load_toc


class VectorStore:
    """ChromaDB vector store for semantic search.
    
    Stores:
    - pages: Book pages with page_text_embedding (1318 pages, 4096 dim)
    - topics: TOC topics with section_topic_embedding (237 topics, 4096 dim)
    
    Pre-computed Qwen embeddings are loaded from parquet files.
    """
    
    EMBEDDING_DIM = 4096  # Qwen embedding dimension
    
    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
        """
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collections
        self.pages_collection = self.client.get_or_create_collection(
            name="pages",
            metadata={"hnsw:space": "cosine", "dimension": self.EMBEDDING_DIM}
        )
        
        self.topics_collection = self.client.get_or_create_collection(
            name="topics",
            metadata={"hnsw:space": "cosine", "dimension": self.EMBEDDING_DIM}
        )
        
        self._indexed = False
    
    @property
    def is_indexed(self) -> bool:
        """Check if data has been indexed."""
        return (
            self.pages_collection.count() > 0 and 
            self.topics_collection.count() > 0
        )
    
    def index_all(self, force: bool = False) -> None:
        """Index all data from parquet files.
        
        Args:
            force: If True, reindex even if already indexed
        """
        if self.is_indexed and not force:
            print(f"✓ Already indexed ({self.pages_collection.count()} pages, {self.topics_collection.count()} topics)")
            return
        
        if force:
            print("Force reindexing - clearing existing data...")
            self.client.delete_collection("pages")
            self.client.delete_collection("topics")
            self.pages_collection = self.client.create_collection(
                name="pages",
                metadata={"hnsw:space": "cosine", "dimension": self.EMBEDDING_DIM}
            )
            self.topics_collection = self.client.create_collection(
                name="topics",
                metadata={"hnsw:space": "cosine", "dimension": self.EMBEDDING_DIM}
            )
        
        self._index_pages()
        self._index_topics()
        
        print(f"✓ Indexing complete: {self.pages_collection.count()} pages, {self.topics_collection.count()} topics")
    
    def _index_pages(self) -> None:
        """Index pages from parquet into ChromaDB."""
        print("Loading pages parquet...")
        df = load_pages(use_qwen=True)
        
        # Reset index to ensure we have unique row numbers
        df = df.reset_index(drop=True)
        
        print(f"Indexing {len(df)} pages...")
        
        # Process in batches
        batch_size = 100
        for i in tqdm(range(0, len(df), batch_size), desc="Indexing pages"):
            batch = df.iloc[i:i+batch_size]
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for idx, row in batch.iterrows():
                # Create unique ID using row index to guarantee uniqueness
                page_id = f"page_{idx}_{row['book_id']}_{row['book_page_number']}"
                ids.append(page_id)
                
                # Get embedding (convert numpy to list)
                embedding = row["page_text_embedding"]
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
                
                # Document text
                documents.append(row["page_text"][:10000] if row["page_text"] else "")
                
                # Metadata for filtering
                metadatas.append({
                    "book_id": str(row["book_id"]),
                    "book_name": str(row["book_name"]),
                    "grade": int(row["grade"]),
                    "page_number": int(row["book_page_number"]),
                    "section_title": str(row["section_title"]) if row["section_title"] else "",
                    "topic_title": str(row["topic_title"]) if row["topic_title"] else "",
                    "subject": str(row["global_discipline_name"]),
                    "subject_id": int(row["global_discipline_id"])
                })
            
            # Add to collection
            self.pages_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
    
    def _index_topics(self) -> None:
        """Index topics from TOC parquet into ChromaDB."""
        print("Loading TOC parquet...")
        df = load_toc(use_qwen=True)
        
        # Reset index to ensure unique row numbers
        df = df.reset_index(drop=True)
        
        print(f"Indexing {len(df)} topics...")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Indexing topics"):
            # Create unique ID using row index
            topic_id = f"topic_{idx}_{row['book_id']}_{row['book_topic_id']}"
            ids.append(topic_id)
            
            # Get combined section+topic embedding
            embedding = row["section_topic_embedding"]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            embeddings.append(embedding)
            
            # Combine topic text and summary
            doc_text = f"{row['topic_title']}\n\n{row.get('topic_summary', '')}\n\n{row.get('topic_text', '')}"
            documents.append(doc_text[:10000])
            
            # Extract subtopics list (can be list or numpy array)
            subtopics = row.get("subtopics", None)
            subtopics_str = ""
            if subtopics is not None:
                if isinstance(subtopics, np.ndarray):
                    subtopics = subtopics.tolist()
                if isinstance(subtopics, list) and len(subtopics) > 0:
                    subtopics_str = ", ".join(str(s) for s in subtopics[:10])
            
            # Safe int conversion for page numbers (may be NaN)
            start_page = row.get("topic_start_page")
            end_page = row.get("topic_end_page")
            
            metadatas.append({
                "book_id": str(row["book_id"]),
                "book_name": str(row["book_name"]),
                "grade": int(row["grade"]),
                "section_title": str(row["section_title"]),
                "topic_title": str(row["topic_title"]),
                "topic_type": str(row.get("topic_type", "")),
                "subject": str(row["global_discipline_name"]),
                "subject_id": int(row["global_discipline_id"]),
                "start_page": int(start_page) if start_page is not None and not np.isnan(start_page) else 0,
                "end_page": int(end_page) if end_page is not None and not np.isnan(end_page) else 0,
                "subtopics": subtopics_str
            })
        
        # Add all at once (small dataset)
        self.topics_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search_pages(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for relevant pages by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector (4096 dim)
            top_k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"grade": 9, "subject": "Алгебра"})
            
        Returns:
            Dict with ids, documents, metadatas, distances
        """
        # Build where clause from filter
        where = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                if v is not None:
                    conditions.append({k: v})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}
        
        results = self.pages_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def search_topics(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for relevant topics by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector (4096 dim)
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Dict with ids, documents, metadatas, distances
        """
        where = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                if v is not None:
                    conditions.append({k: v})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}
        
        results = self.topics_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    
    def get_pages_by_topic(
        self,
        topic_title: str,
        grade: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get all pages for a specific topic.
        
        Args:
            topic_title: Topic title to filter by
            grade: Optional grade filter
            
        Returns:
            Dict with matching pages
        """
        where = {"topic_title": topic_title}
        if grade:
            where = {"$and": [{"topic_title": topic_title}, {"grade": grade}]}
        
        return self.pages_collection.get(
            where=where,
            include=["documents", "metadatas"]
        )


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def init_vector_store(force_reindex: bool = False) -> VectorStore:
    """Initialize vector store and index data if needed.
    
    Args:
        force_reindex: If True, reindex all data
        
    Returns:
        Initialized VectorStore
    """
    store = get_vector_store()
    store.index_all(force=force_reindex)
    return store


if __name__ == "__main__":
    # CLI for indexing
    import argparse
    
    parser = argparse.ArgumentParser(description="Index data into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force reindex")
    parser.add_argument("--persist-dir", type=str, help="Custom persist directory")
    
    args = parser.parse_args()
    
    print("Initializing vector store...")
    if args.persist_dir:
        store = VectorStore(persist_dir=args.persist_dir)
    else:
        store = get_vector_store()
    
    store.index_all(force=args.force)
    print("Done!")
