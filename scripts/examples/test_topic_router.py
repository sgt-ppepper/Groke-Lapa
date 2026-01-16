"""Working example: TopicRouter usage.

This is a clean, working example of how to use the TopicRouter.

Prerequisites:
1. ChromaDB collections must be set up (run scripts/setup/setup_chroma_toc.py)
2. Environment variable LAPATHON_API_KEY must be set
3. Data files must be in the correct location

Usage:
    python scripts/examples/test_topic_router.py

Example queries you can try:
- "Поясни формулу дискриміната" (Algebra)
- "Що таке козацтво?" (History)
- "як будувати зв'язний усний опис місцевості" (Ukrainian Language)
"""
import json
import sys
from pathlib import Path

# Add project root to path (go up two levels from scripts/examples/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.topic_router import TopicRouter


def main():
    """Example: Route a query and get topic with content."""
    
    print("="*70)
    print("TopicRouter - Working Example")
    print("="*70)
    
    try:
        # Step 1: Initialize the router
        print("\n[1/4] Initializing TopicRouter...")
        router = TopicRouter()
        print("   ✓ TopicRouter initialized successfully")
        
        # Step 2: Define your query
        # The router will automatically infer grade and subject if not provided
        query = "Поясни формулу дискриміната"
        query = "Як будувати зв’язний усний опис місцевості"
        query = "Поясни Руїну"
        query = "Як українська козацька держава взаємодіяла з іншими європейськими країнами"
        query = "Як спрощувати вирази з дробами"
        query = "чому порядок слів у реченні може змінювати зміст або інтонацію"
        
        print(f"\n[2/4] Query:")
        print(f"   \"{query}\"")
        print(f"   Grade: (will be inferred)")
        print(f"   Subject: (will be inferred)")
        
        # Step 3: Route the query
        print(f"\n[3/4] Routing query...")
        result = router.route(
            query=query,
            top_k=5  # Number of document chunks to return
        )
        
        # Step 4: Display results
        print(f"\n[4/4] Results:")
        print(f"   ✓ Inferred Grade: {result.get('grade')}")
        print(f"   ✓ Inferred Subject: {result.get('subject')}")
        print(f"   ✓ Matched Topic: {result.get('topic')}")
        print(f"   ✓ Retrieved {len(result.get('retrieved_docs', []))} document(s)")
        
        # Format output as JSON
        output = {
            "topic": result["topic"],
            "retrieved_docs": result["retrieved_docs"],
            "grade": result.get("grade"),
            "subject": result.get("subject"),
            "discipline_id": result.get("discipline_id")
        }
        
        print("\n" + "="*70)
        print("Full Output (JSON):")
        print("="*70)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        print("="*70)
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Make sure ChromaDB collections are set up:")
        print("      python scripts/setup/setup_chroma_toc.py")
        print("   2. Check that LAPATHON_API_KEY is set in .env file")
        print("   3. Verify data files are in the correct location")
        sys.exit(1)


if __name__ == "__main__":
    main()

