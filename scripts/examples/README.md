# Example Scripts

This directory contains example scripts demonstrating how to use the TopicRouter.

## Examples

### `test_topic_router.py` ⭐ **Main Working Example**

A clean, well-documented example showing TopicRouter usage with automatic grade/subject inference.

**Features:**
- Automatic grade and subject inference
- Clean step-by-step output
- Error handling and troubleshooting tips
- Demonstrates the full routing workflow

**Usage:**
```bash
python scripts/examples/test_topic_router.py
```

**What it shows:**
- How to initialize TopicRouter
- How to route queries without specifying grade/subject
- How to get results with topic and content
- How the router infers grade and subject from queries

### `topic_router_example.py`

A simpler example showing basic TopicRouter usage with explicit grade and subject.

**Features:**
- Explicit grade and subject specification
- Simple, straightforward example
- Shows the basic routing API

**Usage:**
```bash
python scripts/examples/topic_router_example.py
```

**What it shows:**
- How to specify grade and subject explicitly
- Basic routing with known parameters
- Simple output format

### `create_test_set.py`

Generate a test set of queries for evaluation.

**Usage:**
```bash
python scripts/examples/create_test_set.py
```

**Output:** Creates `test_set_20.json` in the project root

### `evaluate_topic_router.py`

Evaluate TopicRouter performance against a test set.

**Usage:**
```bash
python scripts/examples/evaluate_topic_router.py
```

**Input:** Requires `test_set_20.json` (created by `create_test_set.py`)

## Quick Start

**Recommended:** Start with `test_topic_router.py`:

```bash
python scripts/examples/test_topic_router.py
```

This is the most complete example with:
- Automatic inference
- Detailed output
- Error handling
- Troubleshooting tips

## Differences Between Examples

| Feature | test_topic_router.py | topic_router_example.py |
|---------|---------------------|------------------------|
| Grade/Subject | Inferred automatically | Specified explicitly |
| Output Detail | Detailed, step-by-step | Simple JSON output |
| Error Handling | Comprehensive | Basic |
| Documentation | Extensive | Minimal |
| Use Case | Learning/Testing | Quick reference |

## Prerequisites

Before running examples:

1. **Set up ChromaDB collections:**
   ```bash
   python scripts/setup/setup_chroma_toc.py
   python scripts/setup/setup_chroma_pages.py
   ```

2. **Set environment variables:**
   - Create `.env` file in `Groke-Lapa/` directory
   - Add `LAPATHON_API_KEY=your-api-key-here`

3. **Verify data files:**
   - Check that parquet files are in `Lapathon2026_Mriia_public_files/text-embedding-qwen/`

## Example Queries to Try

Try these queries in `test_topic_router.py`:

- **Algebra:** "Поясни формулу дискриміната"
- **History:** "Що таке козацтво?"
- **Ukrainian Language:** "як будувати зв'язний усний опис місцевості"

