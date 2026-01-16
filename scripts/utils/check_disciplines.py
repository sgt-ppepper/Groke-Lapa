"""Check what discipline IDs are in the ChromaDB collection."""
import sys
from pathlib import Path
# Add src to path (go up two levels from scripts/utils/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import chromadb
from chromadb.config import Settings
from src.config import get_settings

settings = get_settings()
client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection('toc_topics')
results = collection.get()

print(f"Total documents: {len(results['ids'])}")
print("\nDiscipline IDs and counts:")
from collections import Counter
discipline_ids = [m.get('global_discipline_id') for m in results['metadatas'] if m.get('global_discipline_id')]
print(Counter(discipline_ids))

print("\nDiscipline names:")
discipline_names = [m.get('global_discipline_name') for m in results['metadatas'] if m.get('global_discipline_name')]
print(Counter(discipline_names))

print("\nGrades:")
grades = [m.get('grade') for m in results['metadatas'] if m.get('grade')]
print(Counter(grades))

# Check for grade 8 and discipline_id 2
print("\n\nTopics for grade=8 and discipline_id=2:")
filtered = collection.get(
    where={
        "$and": [
            {"grade": 8},
            {"global_discipline_id": 2}
        ]
    }
)
print(f"Found {len(filtered['ids'])} topics")
if filtered['ids']:
    print("Sample topic titles:")
    for i, meta in enumerate(filtered['metadatas'][:5], 1):
        print(f"  {i}. {meta.get('topic_title', 'N/A')}")

