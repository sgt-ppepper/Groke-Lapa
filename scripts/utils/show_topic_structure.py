"""Show what fields represent 'topics' in the data."""
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
results = collection.get(limit=2)

print("="*70)
print("TOPIC STRUCTURE IN CHROMADB")
print("="*70)

if results['metadatas']:
    print("\n1. METADATA FIELDS (stored with each topic):")
    print("-"*70)
    sample_meta = results['metadatas'][0]
    for key, value in sorted(sample_meta.items()):
        if value is not None:
            print(f"  {key:25s}: {str(value)[:60]}")
    
    print("\n2. DOCUMENT STRUCTURE (what's stored as the document text):")
    print("-"*70)
    print("First 300 characters of sample document:")
    print(results['documents'][0][:300])
    
    print("\n3. KEY FIELDS FOR 'TOPICS':")
    print("-"*70)
    print("  • book_topic_id: Unique identifier for the topic")
    print("  • topic_title: The name/title of the topic (returned as 'topic')")
    print("  • topic_summary: Summary description of the topic")
    print("  • topic_text: Full text content of the topic")
    print("  • section_title: The section/chapter this topic belongs to")
    print("  • subtopics: List of subtopics within this topic")
    
    print("\n4. SAMPLE TOPIC:")
    print("-"*70)
    meta = results['metadatas'][0]
    print(f"  ID: {meta.get('book_topic_id')}")
    print(f"  Title: {meta.get('topic_title')}")
    print(f"  Section: {meta.get('section_title')}")
    print(f"  Discipline: {meta.get('global_discipline_name')} (ID: {meta.get('global_discipline_id')})")
    print(f"  Grade: {meta.get('grade')}")

print("\n" + "="*70)
print("WHAT WE RETURN AS 'topic':")
print("="*70)
print("The 'topic' field in the output is the 'topic_title' from metadata")
print("This represents the specific topic/lesson name from the textbook TOC")

