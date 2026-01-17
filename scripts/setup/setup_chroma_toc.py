"""Setup script to create and index ChromaDB collection for TOC topics.

This script:
1. Loads the TOC parquet file
2. Creates ChromaDB collection 'toc_topics'
3. Indexes all topics with embeddings and metadata
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import chromadb
from chromadb.config import Settings

# Add src to path (go up two levels from scripts/setup/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.config import get_settings
from src.llm.embeddings import QwenEmbeddings


def to_float_list(x: Any) -> List[float]:
    """Ensure Python list[float] from parquet embedding field."""
    if x is None:
        raise ValueError("Embedding is None")
    if isinstance(x, list):
        return [float(v) for v in x]
    if hasattr(x, 'tolist'):  # numpy array
        return x.tolist()
    raise TypeError(f"Unsupported embedding type: {type(x)}")


def clean_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be flat scalar JSON types."""
    out = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


def build_router_text(row, topic_text_max_chars: int = 1500) -> str:
    """Build a compact text for topic routing.
    
    Uses:
      - topic_title
      - subtopics (list[str]) OR subtopics_with_text[].name
      - topic_summary
      - section_title
      - (optional) first N chars of topic_text
    """
    def _s(x):
        return "" if x is None else str(x).strip()

    def _join_list(xs):
        if isinstance(xs, list):
            return "; ".join([_s(v) for v in xs if _s(v)])
        return ""

    section = _s(row.get("section_title"))
    topic = _s(row.get("topic_title"))
    summary = _s(row.get("topic_summary"))

    # Prefer subtopics_with_text.name if available; fallback to subtopics list
    sub_names = ""
    swt = row.get("subtopics_with_text")
    if isinstance(swt, list) and len(swt) > 0:
        names = []
        for d in swt:
            if isinstance(d, dict):
                nm = _s(d.get("name"))
                if nm:
                    names.append(nm)
        sub_names = "; ".join(names)

    if not sub_names:
        sub_names = _join_list(row.get("subtopics"))

    # Optional short body (avoid huge text that can dilute embeddings)
    body = _s(row.get("topic_text"))
    if topic_text_max_chars and body:
        body = body[:topic_text_max_chars]

    parts = []
    if topic:
        parts.append(f"TOPIC: {topic}")
    if sub_names:
        parts.append(f"SUBTOPICS: {sub_names}")
    if summary:
        parts.append(f"SUMMARY: {summary}")
    if section:
        parts.append(f"SECTION: {section}")
    if body:
        parts.append(f"TEXT: {body}")

    return "\n".join(parts)


def recreate_collection(client, name: str, metadata: Dict[str, Any] | None = None):
    """Delete if exists, then create."""
    try:
        client.delete_collection(name)
        print(f"  Deleted existing collection '{name}'")
    except Exception:
        pass
    return client.create_collection(name=name, metadata=metadata or {})


def main():
    """Main function to set up ChromaDB TOC collection."""
    print("="*60)
    print("ChromaDB TOC Topics Setup")
    print("="*60)
    
    settings = get_settings()
    
    # Check if parquet file exists - try multiple locations
    toc_path = settings.toc_parquet_path
    
    # If relative path doesn't exist, try relative to script location and other common paths
    if not toc_path.exists():
        script_dir = Path(__file__).parent
        # Try multiple alternative paths (order matters - try most likely first)
        alt_paths = [
            Path("/app/Lapathon2026_Mriia_public_files/text-embedding-qwen/toc_for_hackathon_with_subtopics.parquet"),
            Path("/app") / "Lapathon2026_Mriia_public_files" / "text-embedding-qwen" / "toc_for_hackathon_with_subtopics.parquet",
            script_dir.parent / "Lapathon2026_Mriia_public_files" / "text-embedding-qwen" / "toc_for_hackathon_with_subtopics.parquet",
        ]
        
        found = False
        for alt_path in alt_paths:
            if alt_path.exists():
                toc_path = alt_path
                print(f"   Found data file at: {toc_path}")
                found = True
                break
        
        if not found:
            print(f"❌ Error: TOC parquet file not found at:")
            print(f"   - {settings.toc_parquet_path}")
            for alt_path in alt_paths:
                exists = "✓" if alt_path.exists() else "✗"
                print(f"   {exists} {alt_path}")
            print(f"\n   Please ensure the data files are in one of these locations:")
            print(f"   - {Path(settings.data_dir).resolve()}/text-embedding-qwen/")
            print(f"   - /app/Lapathon2026_Mriia_public_files/text-embedding-qwen/")
            sys.exit(1)
    
    print(f"\n1. Loading TOC data from: {toc_path}")
    toc_df = pd.read_parquet(toc_path)
    print(f"   ✓ Loaded {len(toc_df)} TOC topics")
    
    # Build router text for each topic
    print("\n2. Building router text for topics...")
    toc_df["router_text"] = toc_df.apply(build_router_text, axis=1)
    print(f"   ✓ Built router text for {len(toc_df)} topics")
    
    # Initialize ChromaDB client
    print(f"\n3. Initializing ChromaDB client at: {settings.chroma_persist_dir}")
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    print("   ✓ ChromaDB client initialized")
    
    # Create collection
    print("\n4. Creating 'toc_topics' collection...")
    TOC_EMBED_FIELD = "section_topic_embedding"
    
    toc_topics = recreate_collection(
        chroma_client, 
        "toc_topics", 
        {"source": "toc_for_hackathon_with_subtopics"}
    )
    print("   ✓ Collection created")
    
    # Prepare data for indexing
    print("\n5. Preparing data for indexing...")
    ids, docs, embs, metas = [], [], [], []
    seen_ids = set()  # Track IDs to ensure uniqueness
    
    for idx, row in toc_df.iterrows():
        # Generate unique ID - handle None values and duplicates
        book_topic_id = row.get("book_topic_id")
        if book_topic_id is None or pd.isna(book_topic_id):
            # Use index as fallback if book_topic_id is None
            topic_id = f"topic_{idx}"
        else:
            topic_id = str(book_topic_id)
        
        # Ensure uniqueness by appending suffix if duplicate
        original_topic_id = topic_id
        counter = 1
        while topic_id in seen_ids:
            topic_id = f"{original_topic_id}_{counter}"
            counter += 1
        seen_ids.add(topic_id)
        
        emb = to_float_list(row[TOC_EMBED_FIELD])
        
        # Build document: router_text for search, then full topic_text for retrieval
        router_text = row["router_text"]
        topic_text = str(row.get("topic_text", "") or "").strip()
        
        # Combine router_text and topic_text with clear separator
        # Router text is used for semantic search, topic_text is for content retrieval
        if topic_text:
            doc = f"{router_text}\n\n---TOPIC_CONTENT---\n{topic_text}"
        else:
            doc = router_text
        
        md = clean_metadata({
            "global_discipline_id": row.get("global_discipline_id"),
            "global_discipline_name": row.get("global_discipline_name"),
            "grade": row.get("grade"),
            "book_id": row.get("book_id"),
            "book_topic_id": row.get("book_topic_id"),
            "book_section_id": row.get("book_section_id"),
            "topic_type": row.get("topic_type"),
            "topic_title": row.get("topic_title"),
            "section_title": row.get("section_title"),
            "topic_start_page": row.get("topic_start_page"),
            "topic_end_page": row.get("topic_end_page"),
            # Store topic_text length for reference (actual text is in document)
            "has_topic_text": bool(topic_text),
        })
        
        ids.append(topic_id)
        docs.append(doc)
        embs.append(emb)
        metas.append(md)
    
    # Validate embedding dims
    dim = len(embs[0])
    assert all(len(e) == dim for e in embs), "Embedding dimension mismatch!"
    print(f"   ✓ Prepared {len(ids)} topics (embedding dim: {dim})")
    
    # Index in batches
    print("\n6. Indexing topics into ChromaDB...")
    batch_size = 100
    total = len(ids)
    
    for i in range(0, total, batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]
        batch_embs = embs[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        
        toc_topics.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas
        )
        print(f"   Indexed {min(i+batch_size, total)}/{total} topics...", end='\r')
    
    print(f"\n   ✓ Successfully indexed {total} topics")
    
    # Verify collection
    print("\n7. Verifying collection...")
    count = toc_topics.count()
    print(f"   ✓ Collection contains {count} documents")
    
    print("\n" + "="*60)
    print("✓ ChromaDB TOC collection setup complete!")
    print("="*60)
    print(f"\nCollection: 'toc_topics'")
    print(f"Location: {settings.chroma_persist_dir}")
    print(f"Documents: {count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

