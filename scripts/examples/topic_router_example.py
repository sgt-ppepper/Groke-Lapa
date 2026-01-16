"""Example usage of TopicRouter agent.

This script demonstrates how to use the TopicRouter to route teacher queries
to relevant topics from the textbook TOC.
"""
import json
import sys
from pathlib import Path

# Add project root to path (go up two levels from scripts/examples/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.topic_router import TopicRouter, get_discipline_id


def main():
    """Example: Route a teacher query to find relevant topics."""
    
    # Initialize router
    print("Initializing TopicRouter...")
    router = TopicRouter()
    
    # Example query
    # query = "Поясни Руїну"
    # grade = 8
    # subject = "Історія України"

    query = "Поясни формулу дискриміната"
    grade = 8
    subject = "Алгебра"
    discipline_id = get_discipline_id(subject)
    
    print(f"\nQuery: {query}")
    print(f"Grade: {grade}")
    print(f"Subject: {subject} (discipline_id: {discipline_id})")
    print("\nRouting query...")
    
    # Route the query
    result = router.route(
        query=query,
        grade=grade,
        discipline_id=discipline_id,
        top_k=5
    )
    
    # Output in the requested format
    output = {
        "topic": result["topic"],
        "retrieved_docs": result["retrieved_docs"]
    }
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("="*60)


if __name__ == "__main__":
    main()

