# ChromaDB Setup Scripts

This directory contains scripts to initialize and populate ChromaDB collections for the TopicRouter system.

## Scripts

### `setup_chroma_toc.py`
Sets up the **TOC (Table of Contents) topics** collection.

**What it creates:**
- Collection name: `toc_topics`
- Contains: All topics from the textbook TOC
- Includes: Full topic text for immediate retrieval
- Purpose: Fast semantic search for topic routing

**Usage:**
```bash
cd Groke-Lapa
python scripts/setup/setup_chroma_toc.py
```

**Output:**
- Creates/updates `toc_topics` collection in ChromaDB
- Indexes ~237 topics with embeddings
- Each topic includes full `topic_text` content

### `setup_chroma_pages.py`
Sets up the **pages** collection for detailed page-by-page content.

**What it creates:**
- Collection name: `pages`
- Contains: Individual pages from textbooks
- Links: Pages linked to topics via `book_topic_id`
- Purpose: Detailed page content retrieval

**Usage:**
```bash
cd Groke-Lapa
python scripts/setup/setup_chroma_pages.py
```

**Output:**
- Creates/updates `pages` collection in ChromaDB
- Indexes ~1,318 pages with embeddings
- Pages are linked to topics for easy retrieval

## Setup Order

1. **First, set up TOC collection** (required):
   ```bash
   python scripts/setup/setup_chroma_toc.py
   ```

2. **Then, set up pages collection** (optional, but recommended):
   ```bash
   python scripts/setup/setup_chroma_pages.py
   ```

## Prerequisites

- Python environment with dependencies installed
- Data files in `Lapathon2026_Mriia_public_files/text-embedding-qwen/`:
  - `toc_for_hackathon_with_subtopics.parquet`
  - `pages_for_hackathon.parquet`
- ChromaDB will be created in `./chroma_db/` directory

## What Gets Indexed

### TOC Collection (`toc_topics`)
- Topic titles and summaries
- Subtopic names
- Full topic text content
- Section information
- Grade and subject metadata
- Page ranges

### Pages Collection (`pages`)
- Full page text content
- Page numbers
- Topic and section links
- Grade and subject metadata
- Theory/exercise flags

## Troubleshooting

**File not found errors:**
- Check that data files are in the correct location
- Scripts will try multiple paths automatically
- Verify the path in error messages

**Duplicate ID errors:**
- Scripts handle duplicates automatically
- If issues persist, delete the ChromaDB directory and re-run

**Collection already exists:**
- Scripts will delete and recreate collections
- This is safe - all data is re-indexed from parquet files

