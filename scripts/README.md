# Scripts Directory

This directory contains organized scripts for setting up and working with the TopicRouter system.

## Directory Structure

```
scripts/
├── setup/              # ChromaDB initialization scripts
├── utils/               # Utility and diagnostic scripts
├── examples/           # Example scripts and evaluation tools
└── README.md          # This file
```

## Setup Scripts (`setup/`)

These scripts initialize and populate ChromaDB collections.

### `setup_chroma_toc.py`
Creates and indexes the `toc_topics` collection from the TOC parquet file.

**Usage:**
```bash
python scripts/setup/setup_chroma_toc.py
```

**What it does:**
- Loads `toc_for_hackathon_with_subtopics.parquet`
- Creates ChromaDB collection `toc_topics`
- Indexes all topics with embeddings and metadata
- Includes full `topic_text` for immediate retrieval

### `setup_chroma_pages.py`
Creates and indexes the `pages` collection from the pages parquet file.

**Usage:**
```bash
python scripts/setup/setup_chroma_pages.py
```

**What it does:**
- Loads `pages_for_hackathon.parquet`
- Creates ChromaDB collection `pages`
- Indexes all pages with embeddings and metadata
- Links pages to topics via `book_topic_id`

**Prerequisites:**
- Run `setup_chroma_toc.py` first (recommended, but not required)

## Utility Scripts (`utils/`)

Diagnostic and utility scripts for exploring the data.

### `list_all_topics.py` / `list_all_topics_compact.py`
List all topics in the ChromaDB collection, organized by subject and grade.

**Usage:**
```bash
python scripts/utils/list_all_topics_compact.py
```

### `check_disciplines.py` / `check_all_disciplines.py`
Check what disciplines (subjects) are available in the ChromaDB collection.

**Usage:**
```bash
python scripts/utils/check_all_disciplines.py
```

### `show_topic_structure.py`
Display the structure of a sample topic in ChromaDB.

**Usage:**
```bash
python scripts/utils/show_topic_structure.py
```

## Example Scripts (`examples/`)

Example scripts demonstrating usage and evaluation.

### `create_test_set.py`
Generate a test set of queries for TopicRouter evaluation.

**Usage:**
```bash
python scripts/examples/create_test_set.py
```

**Output:** `test_set_20.json` in the project root

### `evaluate_topic_router.py`
Evaluate TopicRouter performance against a test set.

**Usage:**
```bash
python scripts/examples/evaluate_topic_router.py
```

**Input:** `test_set_20.json` (created by `create_test_set.py`)

## Main Working Example

The main working example is in the examples directory:

**`scripts/examples/test_topic_router.py`** - Clean, well-documented example of TopicRouter usage

**Usage:**
```bash
python scripts/examples/test_topic_router.py
```

This is the recommended starting point for understanding how to use TopicRouter.

**Other examples:**
- `scripts/examples/topic_router_example.py` - Simpler example with explicit parameters
- `scripts/examples/create_test_set.py` - Generate test data
- `scripts/examples/evaluate_topic_router.py` - Evaluate performance

## Quick Start

1. **Set up ChromaDB collections:**
   ```bash
   python scripts/setup/setup_chroma_toc.py
   python scripts/setup/setup_chroma_pages.py
   ```

2. **Test the router:**
   ```bash
   python test_topic_router.py
   ```

3. **Explore the data:**
   ```bash
   python scripts/utils/list_all_topics_compact.py
   ```

## Notes

- All scripts automatically adjust their paths to find the `src/` directory
- Make sure you have set up your `.env` file with `LAPATHON_API_KEY`
- Data files should be in `Lapathon2026_Mriia_public_files/` directory

