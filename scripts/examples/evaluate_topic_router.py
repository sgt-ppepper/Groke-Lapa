"""Evaluate TopicRouter against the test set.

This script runs each query from the test set and compares the results
with expected values.
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path (go up two levels from scripts/examples/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.topic_router import TopicRouter, get_discipline_id


def evaluate_query(router: TopicRouter, test_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single query and return results.
    
    Note: grade and discipline_id are NOT provided - they will be inferred.
    """
    query = test_entry["query"]
    expected_grade = test_entry["expected_grade"]
    expected_subject = test_entry["expected_subject"]
    expected_topic_title = test_entry["expected_topic_title"]
    expected_discipline_id = test_entry["expected_discipline_id"]
    
    try:
        # Route the query WITHOUT providing grade/subject - they will be inferred
        result = router.route(
            query=query,
            top_k=5
        )
        
        # Check if inferred grade/subject match expected
        inferred_grade = result.get("grade")
        inferred_subject = result.get("subject")
        inferred_discipline_id = result.get("discipline_id")
        
        grade_match = inferred_grade == expected_grade
        subject_match = inferred_subject == expected_subject
        discipline_match = inferred_discipline_id == expected_discipline_id
        
        predicted_topic = result["topic"]
        retrieved_docs = result["retrieved_docs"]
        
        # Check if predicted topic matches expected
        topic_match = predicted_topic == expected_topic_title
        
        # Check if expected topic is in retrieved documents
        topic_in_docs = any(
            expected_topic_title in doc or expected_topic_title[:50] in doc
            for doc in retrieved_docs
        )
        
        return {
            "query": query,
            "expected_topic": expected_topic_title,
            "predicted_topic": predicted_topic,
            "topic_match": topic_match,
            "topic_in_docs": topic_in_docs,
            "retrieved_count": len(retrieved_docs),
            "success": topic_match or topic_in_docs,
            "error": None,
            # Inference results
            "expected_grade": expected_grade,
            "inferred_grade": inferred_grade,
            "grade_match": grade_match,
            "expected_subject": expected_subject,
            "inferred_subject": inferred_subject,
            "subject_match": subject_match,
            "expected_discipline_id": expected_discipline_id,
            "inferred_discipline_id": inferred_discipline_id,
            "discipline_match": discipline_match
        }
    except Exception as e:
        return {
            "query": query,
            "expected_topic": expected_topic_title,
            "predicted_topic": None,
            "topic_match": False,
            "topic_in_docs": False,
            "retrieved_count": 0,
            "success": False,
            "error": str(e),
            "expected_grade": expected_grade,
            "inferred_grade": None,
            "grade_match": False,
            "expected_subject": expected_subject,
            "inferred_subject": None,
            "subject_match": False,
            "expected_discipline_id": expected_discipline_id,
            "inferred_discipline_id": None,
            "discipline_match": False
        }


def main():
    """Run evaluation on test set."""
    # Load test set
    test_file = Path(__file__).parent / "test_set_20.json"
    if not test_file.exists():
        print(f"❌ Test set not found: {test_file}")
        print("   Run create_test_set.py first to generate the test set.")
        sys.exit(1)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    
    print("="*70)
    print("EVALUATING TOPICROUTER")
    print("="*70)
    print(f"\nTest set: {len(test_set)} queries")
    print("Initializing TopicRouter...")
    
    router = TopicRouter()
    print("✓ TopicRouter initialized\n")
    
    results = []
    for i, test_entry in enumerate(test_set, 1):
        print(f"[{i}/{len(test_set)}] Query: {test_entry['query'][:50]}...")
        result = evaluate_query(router, test_entry)
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Success: {result['predicted_topic'][:50]}...")
        else:
            print(f"  ✗ Failed: Expected '{test_entry['expected_topic_title'][:50]}...'")
            if result["error"]:
                print(f"    Error: {result['error']}")
    
    # Calculate metrics
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    exact_matches = sum(1 for r in results if r["topic_match"])
    in_docs = sum(1 for r in results if r["topic_in_docs"])
    
    # Inference metrics
    grade_correct = sum(1 for r in results if r.get("grade_match", False))
    subject_correct = sum(1 for r in results if r.get("subject_match", False))
    discipline_correct = sum(1 for r in results if r.get("discipline_match", False))
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal queries: {total}")
    print(f"\nTopic Matching:")
    print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"    - Exact topic match: {exact_matches} ({exact_matches/total*100:.1f}%)")
    print(f"    - Topic found in retrieved docs: {in_docs} ({in_docs/total*100:.1f}%)")
    print(f"  Failed: {total - successful} ({(total-successful)/total*100:.1f}%)")
    
    print(f"\nInference Accuracy:")
    print(f"  Grade inference: {grade_correct}/{total} ({grade_correct/total*100:.1f}%)")
    print(f"  Subject inference: {subject_correct}/{total} ({subject_correct/total*100:.1f}%)")
    print(f"  Discipline ID inference: {discipline_correct}/{total} ({discipline_correct/total*100:.1f}%)")
    
    # By subject
    print("\nBy Subject:")
    from collections import defaultdict
    by_subject = defaultdict(lambda: {"total": 0, "success": 0})
    
    for test_entry, result in zip(test_set, results):
        subject = test_entry["expected_subject"]
        by_subject[subject]["total"] += 1
        if result["success"]:
            by_subject[subject]["success"] += 1
    
    for subject, stats in sorted(by_subject.items()):
        pct = stats["success"] / stats["total"] * 100
        print(f"  {subject}: {stats['success']}/{stats['total']} ({pct:.1f}%)")
    
    # Save detailed results
    output_file = Path(__file__).parent / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total,
                "successful": successful,
                "exact_matches": exact_matches,
                "in_docs": in_docs,
                "success_rate": successful/total*100,
                "grade_inference_accuracy": grade_correct/total*100,
                "subject_inference_accuracy": subject_correct/total*100,
                "discipline_inference_accuracy": discipline_correct/total*100
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()

