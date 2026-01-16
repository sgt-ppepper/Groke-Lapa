"""Create a test set of 20 queries for TopicRouter evaluation.

Distribution:
- 22% Algebra (4-5 questions)
- 33% History (6-7 questions)  
- Rest Ukrainian Language (8-9 questions)

Each entry contains:
- query: Based on subtopics (not identical to title)
- expected_grade: Grade level
- expected_subject: Subject name
- expected_topic_title: The topic title
- expected_content: Topic content/summary
"""
import sys
from pathlib import Path
import json
import random
from typing import Dict, List, Any

# Add project root to path (go up two levels from scripts/examples/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings
from src.config import get_settings

# Distribution
TOTAL_QUESTIONS = 20
ALGEBRA_PCT = 0.22  # ~4-5 questions
HISTORY_PCT = 0.33  # ~6-7 questions
# Rest is Ukrainian Language

ALGEBRA_COUNT = round(TOTAL_QUESTIONS * ALGEBRA_PCT)  # 4-5
HISTORY_COUNT = round(TOTAL_QUESTIONS * HISTORY_PCT)  # 6-7
UKR_LANG_COUNT = TOTAL_QUESTIONS - ALGEBRA_COUNT - HISTORY_COUNT  # 8-9

DISCIPLINE_MAPPING = {
    72: {"name": "Алгебра", "count": ALGEBRA_COUNT},
    107: {"name": "Історія України", "count": HISTORY_COUNT},
    131: {"name": "Українська мова", "count": UKR_LANG_COUNT},
}


def extract_subtopics(metadata: Dict, document: str) -> List[str]:
    """Extract subtopics from topic metadata or document."""
    subtopics = []
    
    # Try to get subtopics from document (it's in router_text format)
    for line in document.split("\n"):
        if line.startswith("SUBTOPICS:"):
            subtopics_text = line.replace("SUBTOPICS:", "").strip()
            if subtopics_text:
                # Split by semicolon
                subtopics = [s.strip() for s in subtopics_text.split(";") if s.strip()]
                break
    
    return subtopics


def create_query_from_subtopic(topic_title: str, subtopics: List[str], summary: str) -> str:
    """Create a query based on subtopics, not the title itself."""
    if not subtopics:
        # Fallback: use summary or create a question about the topic
        if summary and len(summary) > 50:
            # Extract key concept and make it a question
            words = summary.split()[:10]  # First 10 words
            key_concept = " ".join(words)
            return f"Поясни {key_concept.lower()}"
        else:
            # Generic question
            return f"Розкажи про {topic_title.split('.')[0] if '.' in topic_title else topic_title[:30]}"
    
    # Pick a random subtopic
    subtopic = random.choice(subtopics)
    
    # Create query variations
    query_templates = [
        f"Поясни {subtopic}",
        f"Що таке {subtopic}?",
        f"Розкажи про {subtopic}",
        f"Опиши {subtopic}",
        f"Як працює {subtopic}?",
    ]
    
    return random.choice(query_templates)


def get_topic_content(document: str) -> str:
    """Extract topic content from document."""
    # Document format: "TOPIC: ...\nSUBTOPICS: ...\nSUMMARY: ...\nSECTION: ...\nTEXT: ..."
    parts = {}
    current_key = None
    current_value = []
    
    for line in document.split("\n"):
        if ":" in line and line.split(":")[0].strip() in ["TOPIC", "SUBTOPICS", "SUMMARY", "SECTION", "TEXT"]:
            if current_key:
                parts[current_key] = "\n".join(current_value).strip()
            current_key = line.split(":")[0].strip()
            current_value = [line.split(":", 1)[1].strip()] if ":" in line else []
        elif current_key:
            current_value.append(line)
    
    if current_key:
        parts[current_key] = "\n".join(current_value).strip()
    
    # Prefer SUMMARY, then TEXT, then TOPIC
    if parts.get("SUMMARY"):
        return parts["SUMMARY"]
    elif parts.get("TEXT"):
        return parts["TEXT"][:500]  # Limit length
    elif parts.get("TOPIC"):
        return parts["TOPIC"]
    else:
        return document[:500]


def create_test_set() -> List[Dict[str, Any]]:
    """Create test set by sampling topics from each discipline."""
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection('toc_topics')
    test_set = []
    
    random.seed(42)  # For reproducibility
    
    for discipline_id, info in DISCIPLINE_MAPPING.items():
        discipline_name = info["name"]
        count = info["count"]
        
        print(f"\nSampling {count} topics from {discipline_name} (ID: {discipline_id})...")
        
        # Get all topics for this discipline
        all_topics = collection.get(
            where={"global_discipline_id": discipline_id}
        )
        
        if len(all_topics['ids']) < count:
            print(f"  Warning: Only {len(all_topics['ids'])} topics available, requested {count}")
            count = len(all_topics['ids'])
        
        # Sample random topics
        indices = random.sample(range(len(all_topics['ids'])), count)
        
        for idx in indices:
            metadata = all_topics['metadatas'][idx]
            document = all_topics['documents'][idx]
            topic_id = all_topics['ids'][idx]
            
            topic_title = metadata.get('topic_title', 'Unknown')
            grade = metadata.get('grade')
            summary = None
            
            # Extract summary from document
            for line in document.split("\n"):
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                    break
            
            # Extract subtopics
            subtopics = extract_subtopics(metadata, document)
            
            # Create query from subtopic
            query = create_query_from_subtopic(topic_title, subtopics, summary or "")
            
            # Get content
            content = get_topic_content(document)
            
            test_entry = {
                "query": query,
                "expected_grade": int(grade) if grade else None,
                "expected_subject": discipline_name,
                "expected_topic_title": topic_title,
                "expected_content": content,
                "expected_discipline_id": discipline_id,
                "topic_id": topic_id,
                "subtopics_used": subtopics[:3] if subtopics else []  # For reference
            }
            
            test_set.append(test_entry)
            print(f"  ✓ {query[:50]}... -> {topic_title[:50]}...")
    
    return test_set


def main():
    """Generate and save test set."""
    print("="*70)
    print("CREATING TEST SET")
    print("="*70)
    print(f"\nTarget distribution:")
    print(f"  Algebra: {ALGEBRA_COUNT} questions (22%)")
    print(f"  History: {HISTORY_COUNT} questions (33%)")
    print(f"  Ukrainian Language: {UKR_LANG_COUNT} questions ({100-ALGEBRA_PCT*100-HISTORY_PCT*100:.0f}%)")
    print(f"  Total: {TOTAL_QUESTIONS} questions")
    
    test_set = create_test_set()
    
    # Shuffle to mix subjects
    random.shuffle(test_set)
    
    # Save to JSON
    output_file = Path(__file__).parent / "test_set_20.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ Test set created: {output_file}")
    print(f"  Total entries: {len(test_set)}")
    print("="*70)
    
    # Show summary
    print("\nDistribution:")
    from collections import Counter
    subjects = Counter([entry['expected_subject'] for entry in test_set])
    for subject, count in subjects.items():
        print(f"  {subject}: {count} questions")
    
    print("\nSample entries:")
    for i, entry in enumerate(test_set[:3], 1):
        print(f"\n{i}. Query: {entry['query']}")
        print(f"   Expected: Grade {entry['expected_grade']}, {entry['expected_subject']}")
        print(f"   Topic: {entry['expected_topic_title'][:60]}...")


if __name__ == "__main__":
    main()

