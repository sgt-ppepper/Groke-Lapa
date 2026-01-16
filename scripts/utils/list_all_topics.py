"""List all topics in the ChromaDB collection.

Shows all topics organized by subject and grade.
"""
import sys
from pathlib import Path
# Add src to path (go up two levels from scripts/utils/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import chromadb
from chromadb.config import Settings
from src.config import get_settings
from collections import defaultdict

settings = get_settings()
client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection('toc_topics')
results = collection.get()

print("="*80)
print("ALL TOPICS IN CHROMADB COLLECTION")
print("="*80)
print(f"\nTotal topics: {len(results['ids'])}")

# Organize by subject and grade
topics_by_subject = defaultdict(lambda: defaultdict(list))

for i, meta in enumerate(results['metadatas']):
    subject = meta.get('global_discipline_name', 'Unknown')
    grade = meta.get('grade', 'Unknown')
    topic_title = meta.get('topic_title', 'Unknown')
    topic_id = results['ids'][i]
    
    topics_by_subject[subject][grade].append({
        'id': topic_id,
        'title': topic_title,
        'section': meta.get('section_title', ''),
        'type': meta.get('topic_type', ''),
        'start_page': meta.get('topic_start_page'),
        'end_page': meta.get('topic_end_page')
    })

# Sort subjects
subject_order = ['Алгебра', 'Історія України', 'Українська мова']
for subject in subject_order:
    if subject not in topics_by_subject:
        continue
    
    print(f"\n{'='*80}")
    print(f"SUBJECT: {subject}")
    print(f"{'='*80}")
    
    # Sort by grade
    for grade in sorted(topics_by_subject[subject].keys()):
        topics = topics_by_subject[subject][grade]
        print(f"\n  Grade {grade} ({len(topics)} topics):")
        print(f"  {'-'*76}")
        
        for i, topic in enumerate(topics, 1):
            page_info = ""
            if topic['start_page'] and topic['end_page']:
                page_info = f" (pp. {int(topic['start_page'])}-{int(topic['end_page'])})"
            elif topic['start_page']:
                page_info = f" (p. {int(topic['start_page'])})"
            
            type_info = f" [{topic['type']}]" if topic['type'] else ""
            print(f"  {i:3d}. {topic['title'][:70]}{type_info}{page_info}")
            if topic['section']:
                print(f"       Section: {topic['section'][:70]}")

# Show any other subjects not in the list
for subject in sorted(topics_by_subject.keys()):
    if subject not in subject_order:
        print(f"\n{'='*80}")
        print(f"SUBJECT: {subject}")
        print(f"{'='*80}")
        for grade in sorted(topics_by_subject[subject].keys()):
            topics = topics_by_subject[subject][grade]
            print(f"\n  Grade {grade} ({len(topics)} topics):")
            for i, topic in enumerate(topics, 1):
                print(f"  {i:3d}. {topic['title']}")

print("\n" + "="*80)
print(f"Total: {len(results['ids'])} topics")
print("="*80)

