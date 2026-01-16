"""List all topics in a compact format, optionally save to file."""
import sys
from pathlib import Path
import json
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
print(f"\nTotal topics: {len(results['ids'])}\n")

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

# Prepare output
output_lines = []
all_topics_list = []

# Sort subjects
subject_order = ['Алгебра', 'Історія України', 'Українська мова']
for subject in subject_order:
    if subject not in topics_by_subject:
        continue
    
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"SUBJECT: {subject}")
    output_lines.append(f"{'='*80}")
    
    # Sort by grade
    for grade in sorted(topics_by_subject[subject].keys()):
        topics = topics_by_subject[subject][grade]
        output_lines.append(f"\n  Grade {grade} ({len(topics)} topics):")
        output_lines.append(f"  {'-'*76}")
        
        for i, topic in enumerate(topics, 1):
            page_info = ""
            if topic['start_page'] and topic['end_page']:
                page_info = f" (pp. {int(topic['start_page'])}-{int(topic['end_page'])})"
            elif topic['start_page']:
                page_info = f" (p. {int(topic['start_page'])})"
            
            type_info = f" [{topic['type']}]" if topic['type'] else ""
            line = f"  {i:3d}. {topic['title']}{type_info}{page_info}"
            output_lines.append(line)
            
            # Also add to flat list
            all_topics_list.append({
                'subject': subject,
                'grade': int(grade),
                'topic_title': topic['title'],
                'section': topic['section'],
                'type': topic['type'],
                'start_page': topic['start_page'],
                'end_page': topic['end_page'],
                'topic_id': topic['id']
            })

# Print to console
for line in output_lines:
    print(line)

print("\n" + "="*80)
print(f"Total: {len(results['ids'])} topics")
print("="*80)

# Save to JSON file
output_file = Path(__file__).parent / "all_topics_list.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_topics_list, f, ensure_ascii=False, indent=2)

print(f"\n✓ Full list saved to: {output_file}")

