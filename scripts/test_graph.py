#!/usr/bin/env python3
"""Test script for invoking the Mriia AI Tutor graph.

This script tests the full tutoring pipeline with sample queries.
Run from the Groke-Lapa directory:
    python scripts/test_graph.py

Or with custom query:
    python scripts/test_graph.py --query "Поясни теорему Піфагора" --subject "Алгебра" --grade 9
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def test_graph(
    query: str,
    subject: str = "Українська мова",
    grade: int = 9,
    student_id: int = None,
    verbose: bool = True
):
    """Test the LangGraph workflow with a query.
    
    Args:
        query: Teacher query to test
        subject: Subject name
        grade: Grade level (8 or 9)
        student_id: Optional student ID for personalization
        verbose: Whether to print detailed output
    """
    print("\n" + "=" * 60)
    print("🎓 MRIIA AI TUTOR - Graph Test")
    print("=" * 60)
    print(f"\n📝 Query: {query}")
    print(f"📚 Subject: {subject}")
    print(f"🎒 Grade: {grade}")
    if student_id:
        print(f"👤 Student ID: {student_id}")
    print("-" * 60)
    
    # Import after path setup
    from src.agents.graph import tutor_graph
    from src.agents.state import create_initial_state
    
    # Create initial state
    initial_state = create_initial_state(
        teacher_query=query,
        grade=grade,
        subject=subject,
        student_id=student_id,
        mode="demo"
    )
    
    print("\n▶️ Running graph...")
    
    try:
        # Invoke the graph
        result = tutor_graph.invoke(initial_state)
        
        print("\n✅ Graph execution completed!")
        print("-" * 60)
        
        # Display results
        print("\n📊 RESULTS:\n")
        
        # Topic routing
        matched_topics = result.get("matched_topics", [])
        print(f"🔍 Matched Topics: {len(matched_topics)}")
        for i, topic in enumerate(matched_topics[:3]):
            if isinstance(topic, dict):
                topic_name = topic.get("topic", "Unknown")
                print(f"   {i+1}. {topic_name[:60]}...")
        
        # Pages retrieved
        matched_pages = result.get("matched_pages", [])
        print(f"\n📄 Retrieved Pages: {len(matched_pages)}")
        
        # Lecture content
        lecture_content = result.get("lecture_content", "")
        print(f"\n📖 Lecture Content: {len(lecture_content)} characters")
        if verbose and lecture_content:
            print("\n--- LECTURE PREVIEW (first 500 chars) ---")
            print(lecture_content[:500])
            if len(lecture_content) > 500:
                print("...[truncated]")
            print("--- END PREVIEW ---")
        
        # Control questions
        control_questions = result.get("control_questions", [])
        print(f"\n❓ Control Questions: {len(control_questions)}")
        if verbose:
            for i, q in enumerate(control_questions[:3], 1):
                print(f"   {i}. {q[:60]}...")
        
        # Practice questions
        practice_questions = result.get("practice_questions", [])
        print(f"\n🧪 Practice Questions: {len(practice_questions)}")
        if verbose and practice_questions:
            for i, q in enumerate(practice_questions[:3], 1):
                q_text = q.get("question", "")[:50]
                correct = q.get("correct_answer", "?")
                print(f"   {i}. {q_text}... [Correct: {correct}]")
            if len(practice_questions) > 3:
                print(f"   ... and {len(practice_questions) - 3} more")
        
        # Sources
        sources = result.get("sources", [])
        print(f"\n📚 Sources: {len(sources)}")
        if verbose:
            for source in sources[:5]:
                print(f"   {source}")
        
        # Personalization
        student_profile = result.get("student_profile")
        if student_profile:
            print("\n👤 Student Profile:")
            metrics = student_profile.get("metrics", {})
            if metrics:
                print(f"   Average Score: {metrics.get('average_score', 0):.1f}/12")
                print(f"   Grades Count: {metrics.get('grades_count', 0)}")
            enrichment = student_profile.get("enrichment", {})
            weak = enrichment.get("weak_topics", [])
            if weak:
                print(f"   Weak Topics: {', '.join(weak[:3])}")
        
        # Errors
        error = result.get("error")
        if error:
            print(f"\n⚠️ Error: {error}")
        
        print("\n" + "=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Mriia AI Tutor graph")
    parser.add_argument(
        "--query", "-q",
        default="Поясни тему складних речень",
        help="Teacher query to test"
    )
    parser.add_argument(
        "--subject", "-s",
        default="Українська мова",
        choices=["Українська мова", "Алгебра", "Історія України"],
        help="Subject name"
    )
    parser.add_argument(
        "--grade", "-g",
        type=int,
        default=9,
        choices=[8, 9],
        help="Grade level"
    )
    parser.add_argument(
        "--student-id", "-u",
        type=int,
        default=None,
        help="Student ID for personalization"
    )
    parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Less verbose output"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run all test queries"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("LAPATHON_API_KEY"):
        print("⚠️ Warning: LAPATHON_API_KEY not set in environment")
        print("   Create a .env file with: LAPATHON_API_KEY=your_key_here")
    
    if args.test_all:
        # Run multiple test queries
        test_queries = [
            ("Поясни тему складних речень", "Українська мова", 9),
            ("Розкажи про Руїну", "Історія України", 9),
            ("Поясни теорему Віета", "Алгебра", 9),
            ("Що таке квадратні рівняння?", "Алгебра", 8),
        ]
        
        results = []
        for query, subject, grade in test_queries:
            print(f"\n\n{'#' * 60}")
            print(f"# TEST: {subject} - {grade} клас")
            print(f"{'#' * 60}")
            result = test_graph(
                query=query,
                subject=subject,
                grade=grade,
                verbose=not args.quiet
            )
            results.append({
                "query": query,
                "subject": subject,
                "grade": grade,
                "success": result is not None and not result.get("error"),
                "lecture_length": len(result.get("lecture_content", "")) if result else 0,
                "questions_count": len(result.get("practice_questions", [])) if result else 0,
            })
        
        # Print summary
        print("\n\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"{status} {r['subject']} ({r['grade']} кл): {r['query'][:30]}...")
            print(f"   Lecture: {r['lecture_length']} chars, Questions: {r['questions_count']}")
        
    else:
        # Run single test
        test_graph(
            query=args.query,
            subject=args.subject,
            grade=args.grade,
            student_id=args.student_id,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
