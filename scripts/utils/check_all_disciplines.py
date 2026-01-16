"""Check all disciplines available in the data with detailed breakdown."""
import sys
from pathlib import Path
# Add src to path (go up two levels from scripts/utils/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import chromadb
from chromadb.config import Settings
from src.config import get_settings
from collections import Counter, defaultdict

settings = get_settings()
client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection('toc_topics')
results = collection.get()

print("="*70)
print("ALL DISCIPLINES IN DATA")
print("="*70)
print(f"\nTotal documents: {len(results['ids'])}")

# Group by discipline_id and name
discipline_data = defaultdict(lambda: {'name': None, 'count': 0, 'grades': Counter()})

for meta in results['metadatas']:
    disc_id = meta.get('global_discipline_id')
    disc_name = meta.get('global_discipline_name')
    grade = meta.get('grade')
    
    if disc_id is not None:
        discipline_data[disc_id]['name'] = disc_name
        discipline_data[disc_id]['count'] += 1
        if grade is not None:
            discipline_data[disc_id]['grades'][grade] += 1

print("\n" + "-"*70)
print(f"{'ID':<6} | {'Name':<30} | {'Total':<8} | {'By Grade'}")
print("-"*70)

for disc_id in sorted(discipline_data.keys()):
    data = discipline_data[disc_id]
    name = data['name'] or 'Unknown'
    count = data['count']
    grades_str = ', '.join([f"G{g}: {c}" for g, c in sorted(data['grades'].items())])
    print(f"{disc_id:<6} | {name:<30} | {count:<8} | {grades_str}")

print("\n" + "="*70)
print("CURRENT MAPPING IN CODE:")
print("="*70)
from src.agents.topic_router import SUBJECT_TO_DISCIPLINE_ID
for subject, disc_id in sorted(SUBJECT_TO_DISCIPLINE_ID.items()):
    name = discipline_data.get(disc_id, {}).get('name', 'NOT FOUND')
    print(f"  '{subject}' -> ID {disc_id} ({name})")

print("\n" + "="*70)
