from .state import TutorState
from .content_generator import content_generator_node

import json

# Define a mock state matching the TutorState structure
state: TutorState = {
    "teacher_query": "Explain photosynthesis",
    "grade": 9,
    "subject": "Biology",
    "matched_pages": [
        {
            "book_id": "Book1", 
            "page_number": 10, 
            "page_text": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.", 
            "topic_title": "Photosynthesis"
        }
    ],
    "student_profile": {"weak_topics": ["Biology Basics"]}
}

print("Running Content Generator Node...")
try:
    # Run the node
    result = content_generator_node(state)

    # Print results
    print("\n=== Generated Lecture ===")
    print(result.get("lecture_content"))

    print("\n=== Control Questions ===")
    for i, q in enumerate(result.get("control_questions", []), 1):
        print(f"{i}. {q}")

    print("\n=== Sources ===")
    print(result.get("sources"))

except Exception as e:
    print(f"\nError running node: {e}")
