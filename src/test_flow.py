#!/usr/bin/env python3
"""Test script to run the LangGraph workflow with real input.

Usage:
    python -m src.test_flow
    
Or from project root:
    python src/test_flow.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import create_tutor_graph
from src.agents.state import create_initial_state


def test_demo_flow():
    """Test the demo flow (no student answers)."""
    print("\n" + "="*60)
    print("🧪 TEST: Demo Flow (Learning Mode)")
    print("="*60 + "\n")
    
    # Create graph
    graph = create_tutor_graph()
    
    # Create initial state
    state = create_initial_state(
        teacher_query="Квадратні рівняння",
        grade=9,
        subject="Алгебра",
        student_id=None,  # No personalization
        mode="demo"
    )
    
    print(f"📝 Query: {state['teacher_query']}")
    print(f"📚 Subject: {state['subject']}, Grade: {state['grade']}")
    print(f"👤 Student ID: {state['student_id'] or 'Anonymous'}")
    print("\n" + "-"*60 + "\n")
    
    # Run the graph
    try:
        result = graph.invoke(state)
        print_result(result)
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_personalized_flow():
    """Test the flow with personalization."""
    print("\n" + "="*60)
    print("🧪 TEST: Personalized Flow (With Student Profile)")
    print("="*60 + "\n")
    
    graph = create_tutor_graph()
    
    state = create_initial_state(
        teacher_query="Складні речення",
        grade=9,
        subject="Українська мова",
        student_id=101,  # With personalization
        mode="demo"
    )
    
    print(f"📝 Query: {state['teacher_query']}")
    print(f"📚 Subject: {state['subject']}, Grade: {state['grade']}")
    print(f"👤 Student ID: {state['student_id']}")
    print("\n" + "-"*60 + "\n")
    
    try:
        result = graph.invoke(state)
        print_result(result)
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_answer_check_flow():
    """Test the flow with student answers (check mode)."""
    print("\n" + "="*60)
    print("🧪 TEST: Answer Check Flow (Practice Mode)")
    print("="*60 + "\n")
    
    graph = create_tutor_graph()
    
    state = create_initial_state(
        teacher_query="Квадратні рівняння",
        grade=9,
        subject="Алгебра",
        student_id=101,
        mode="practice"
    )
    
    # Simulate student answers
    state["student_answers"] = ["A", "B", "C", "D", "A", "B", "C", "D"]
    
    print(f"📝 Query: {state['teacher_query']}")
    print(f"📚 Subject: {state['subject']}, Grade: {state['grade']}")
    print(f"👤 Student ID: {state['student_id']}")
    print(f"✍️ Student Answers: {state['student_answers']}")
    print("\n" + "-"*60 + "\n")
    
    try:
        result = graph.invoke(state)
        print_result(result, show_evaluation=True)
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(result, show_evaluation=False):
    """Print the result in a readable format."""
    print("\n" + "="*60)
    print("📊 RESULT")
    print("="*60 + "\n")
    
    # Error check
    if result.get("error"):
        print(f"⚠️ Error: {result['error']}")
    
    # Lecture
    lecture = result.get("lecture_content", "")
    print(f"📖 Lecture Content: {len(lecture)} chars")
    if lecture:
        print(f"   Preview: {lecture[:200]}...")
    
    # Control questions
    control = result.get("control_questions", [])
    print(f"\n❓ Control Questions: {len(control)}")
    for i, q in enumerate(control[:3], 1):
        print(f"   {i}. {q[:60]}...")
    
    # Practice questions
    practice = result.get("practice_questions", [])
    print(f"\n📝 Practice Questions: {len(practice)}")
    for i, q in enumerate(practice[:3], 1):
        text = q.get("question", "")[:50]
        answer = q.get("correct_answer", "?")
        validated = "✓" if q.get("is_validated") else "?"
        print(f"   {i}. {text}... (Answer: {answer}) [{validated}]")
    if len(practice) > 3:
        print(f"   ... and {len(practice) - 3} more")
    
    # Sources
    sources = result.get("sources", [])
    print(f"\n📚 Sources: {len(sources)}")
    for s in sources[:3]:
        print(f"   • {s}")
    
    # Personalization
    profile = result.get("student_profile")
    if profile:
        metrics = profile.get("metrics", {})
        avg = metrics.get("average_score", 0)
        print(f"\n👤 Student Profile: avg score {avg:.1f}/12")
    
    # Evaluation (if answers were provided)
    if show_evaluation:
        eval_results = result.get("evaluation_results", [])
        if eval_results:
            correct = sum(1 for r in eval_results if r.get("is_correct"))
            total = len(eval_results)
            print(f"\n✅ Evaluation: {correct}/{total} correct")
        
        recommendations = result.get("recommendations", "")
        if recommendations:
            print(f"\n📋 Recommendations:")
            print(f"   {recommendations[:200]}...")
        
        next_topics = result.get("next_topics", [])
        if next_topics:
            print(f"\n📌 Next Topics: {', '.join(next_topics[:3])}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Run all tests."""
    print("\n" + "🚀 Starting Mriia AI Tutor Test Flow")
    print("="*60)
    
    # Choose which test to run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["demo", "personalized", "check", "all"], 
                       default="demo", help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "demo" or args.test == "all":
        test_demo_flow()
    
    if args.test == "personalized" or args.test == "all":
        test_personalized_flow()
    
    if args.test == "check" or args.test == "all":
        test_answer_check_flow()
    
    print("\n✅ Tests completed!")


if __name__ == "__main__":
    main()
