"""Setup script to create and index ChromaDB collection for book pages.

This script:
1. Loads the pages parquet file
2. Creates ChromaDB collection 'pages'
3. Indexes all pages with embeddings and metadata
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


def recreate_collection(client, name: str, metadata: Dict[str, Any] | None = None):
    """Delete if exists, then create."""
    try:
        client.delete_collection(name)
        print(f"  Deleted existing collection '{name}'")
    except Exception:
        pass
    return client.create_collection(name=name, metadata=metadata or {})


def main():
    """Main function to set up ChromaDB pages collection."""
    print("="*60)
    print("ChromaDB Pages Setup")
    print("="*60)
    
    settings = get_settings()
    
    # Check if parquet file exists - try multiple locations
    pages_path = settings.pages_parquet_path
    
    # If relative path doesn't exist, try relative to script location
    if not pages_path.exists():
        script_dir = Path(__file__).parent
        # Try relative to script directory (Groke-Lapa/)
        alt_path = script_dir / "Lapathon2026_Mriia_public_files" / "text-embedding-qwen" / "pages_for_hackathon.parquet"
        if alt_path.exists():
            pages_path = alt_path
            print(f"   Found data file at: {pages_path}")
        else:
            # Try absolute path from workspace root
            workspace_root = script_dir.parent
            alt_path2 = workspace_root / "Groke-Lapa" / "Lapathon2026_Mriia_public_files" / "text-embedding-qwen" / "pages_for_hackathon.parquet"
            if alt_path2.exists():
                pages_path = alt_path2
                print(f"   Found data file at: {pages_path}")
            else:
                print(f"❌ Error: Pages parquet file not found at:")
                print(f"   - {settings.pages_parquet_path}")
                print(f"   - {alt_path}")
                print(f"   - {alt_path2}")
                print(f"\n   Please ensure the data files are in one of these locations:")
                print(f"   - {Path(settings.data_dir).resolve()}/text-embedding-qwen/")
                print(f"   - {script_dir}/Lapathon2026_Mriia_public_files/text-embedding-qwen/")
                print(f"   - {workspace_root}/Groke-Lapa/Lapathon2026_Mriia_public_files/text-embedding-qwen/")
                sys.exit(1)
    
    print(f"\n1. Loading pages data from: {pages_path}")
    pages_df = pd.read_parquet(pages_path)
    print(f"   ✓ Loaded {len(pages_df)} pages")
    
    # Initialize ChromaDB client
    print(f"\n2. Initializing ChromaDB client at: {settings.chroma_persist_dir}")
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    print("   ✓ ChromaDB client initialized")
    
    # Create collection
    print("\n3. Creating 'pages' collection...")
    PAGE_EMBED_FIELD = "page_text_embedding"
    
    pages_collection = recreate_collection(
        chroma_client, 
        "pages", 
        {"source": "pages_for_hackathon"}
    )
    print("   ✓ Collection created")
    
    # Prepare data for indexing
    print("\n4. Preparing data for indexing...")
    ids, docs, embs, metas = [], [], [], []
    seen_ids = set()  # Track IDs to ensure uniqueness
    
    for idx, row in pages_df.iterrows():
        # Generate unique ID from book_id and page_number
        book_id = str(row.get("book_id", ""))
        page_num = row.get("book_page_number")
        if page_num is None or pd.isna(page_num):
            page_num = row.get("page_number", idx)
        
        # Create base ID
        base_page_id = f"{book_id}_{int(page_num)}"
        
        # Ensure uniqueness by appending suffix if duplicate
        page_id = base_page_id
        counter = 1
        while page_id in seen_ids:
            page_id = f"{base_page_id}_{counter}"
            counter += 1
        seen_ids.add(page_id)
        
        # Get embedding
        emb = to_float_list(row[PAGE_EMBED_FIELD])
        
        # Use page_text as document
        doc = str(row.get("page_text", "") or "")
        
        # Prepare metadata
        md = clean_metadata({
            "book_id": row.get("book_id"),
            "book_filename": row.get("book_filename"),
            "book_name": row.get("book_name"),
            "grade": row.get("grade"),
            "book_page_number": row.get("book_page_number"),
            "page_number": row.get("page_number"),
            "section_title": row.get("section_title"),
            "topic_title": row.get("topic_title"),
            "book_topic_id": row.get("book_topic_id"),
            "book_section_id": row.get("book_section_id"),
            "global_discipline_id": row.get("global_discipline_id"),
            "global_discipline_name": row.get("global_discipline_name"),
            "contains_theory": row.get("contains_theory"),
        })
        
        ids.append(page_id)
        docs.append(doc)
        embs.append(emb)
        metas.append(md)
    
    # Validate embedding dims
    dim = len(embs[0])
    assert all(len(e) == dim for e in embs), "Embedding dimension mismatch!"
    print(f"   ✓ Prepared {len(ids)} pages (embedding dim: {dim})")
    
    # Index in batches
    print("\n5. Indexing pages into ChromaDB...")
    batch_size = 100
    total = len(ids)
    
    for i in range(0, total, batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]
        batch_embs = embs[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        
        pages_collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas
        )
        print(f"   Indexed {min(i+batch_size, total)}/{total} pages...", end='\r')
    
    print(f"\n   ✓ Successfully indexed {total} pages")
    
    # Verify collection
    print("\n6. Verifying collection...")
    count = pages_collection.count()
    print(f"   ✓ Collection contains {count} documents")
    
    print("\n" + "="*60)
    print("✓ ChromaDB pages collection setup complete!")
    print("="*60)
    print(f"\nCollection: 'pages'")
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

