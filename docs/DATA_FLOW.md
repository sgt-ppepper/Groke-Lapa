# Data Flow Documentation - Mriia AI Tutor

This document describes the input and output data for each module and LangGraph node in the tutoring pipeline.

## Pipeline Overview

```
START → Topic Router → Context Retriever → Personalization → 
Content Generator → Practice Generator → Solver/Validator → 
(loop if invalid) → Check Answers (if provided) → Recommendations → END
```

---

## Shared State Schema (TutorState)

All nodes read from and write to a shared `TutorState` TypedDict:

```python
class TutorState(TypedDict, total=False):
    # === Input (from API request) ===
    teacher_query: str              # Teacher's topic request
    student_id: Optional[int]       # Student identifier (for personalization)
    grade: int                      # Grade level (8 or 9)
    subject: Optional[str]          # Subject name
    mode: Literal["demo", "benchmark", "practice"]
    student_answers: Optional[List[str]]  # Student's submitted answers
    
    # === Routing ===
    matched_topics: List[dict]      # Topics from TOC matching query
    matched_pages: List[dict]       # Book pages for context
    
    # === Student Profile ===
    student_profile: Optional[dict] # Scores, absences, weak topics
    
    # === Generated Content ===
    lecture_content: str            # Generated lecture text
    practice_questions: List[dict]  # Generated test questions
    
    # === Validation ===
    solved_answers: List[dict]      # LLM-solved answers for validation
    validation_passed: bool         # Whether questions passed validation
    regeneration_count: int         # Number of regeneration attempts
    
    # === Evaluation ===
    evaluation_results: List[dict]  # Checked student answers
    recommendations: str            # Learning recommendations
    
    # === Metadata ===
    sources: List[str]              # Page/topic references for grounding
    final_response: dict            # Assembled final response
```

---

## Graph Nodes - Input/Output Specification

### 1. Topic Router

**Purpose:** Match teacher query to relevant topics from the Table of Contents.

**LLM Used:** MamayLM (or embedding-based search)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `teacher_query` | `str` | Teacher's topic request (e.g., "Складні речення") |
| **Input** | `grade` | `int` | Grade level (8 or 9) |
| **Input** | `subject` | `Optional[str]` | Subject filter |
| **Output** | `matched_topics` | `List[dict]` | Top 3 matched topics |

**Output Schema - `matched_topics[i]`:**
```python
{
    "book_id": str,           # Textbook identifier
    "topic_title": str,       # Topic name
    "subtopics": List[str],   # List of subtopics
    "start_page": int,        # First page of topic
    "end_page": int,          # Last page of topic
    "similarity_score": float # Match confidence (0-1)
}
```

**Data Source:** `toc_for_hackathon_with_subtopics.parquet`

---

### 2. Context Retriever

**Purpose:** Retrieve relevant book pages using semantic search.

**LLM Used:** QwenEmbeddings (for query embedding)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `teacher_query` | `str` | Query for embedding |
| **Input** | `matched_topics` | `List[dict]` | Topics with page ranges |
| **Input** | `grade` | `int` | For filtering |
| **Output** | `matched_pages` | `List[dict]` | Retrieved page content |
| **Output** | `sources` | `List[str]` | Page references |

**Output Schema - `matched_pages[i]`:**
```python
{
    "book_id": str,           # Textbook identifier
    "page_number": int,       # Page number
    "page_text": str,         # Full page content
    "topic": str,             # Associated topic
    "similarity_score": float # Relevance score
}
```

**Data Source:** `pages_for_hackathon.parquet` via ChromaDB

---

### 3. Personalization Engine

**Purpose:** Load student profile and identify weak topics for content adaptation.

**LLM Used:** None (data lookup only)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `student_id` | `Optional[int]` | Student identifier |
| **Input** | `matched_topics` | `List[dict]` | Current topics |
| **Output** | `student_profile` | `Optional[dict]` | Student data |

**Output Schema - `student_profile`:**
```python
{
    "student_id": int,
    "scores": [                     # From benchmark_scores.parquet
        {
            "subject": str,
            "topic": str,
            "score": float,         # 0-100
            "date": str
        }
    ],
    "absences": [                   # From benchmark_absences.parquet
        {
            "date": str,
            "subject": str,
            "topics_missed": List[str]
        }
    ],
    "weak_topics": List[str],       # Topics with low scores
    "difficulty_adjustment": float  # -1.0 to +1.0
}
```

**Data Sources:** 
- `benchmark_scores.parquet`
- `benchmark_absences.parquet`

---

### 4. Content Generator

**Purpose:** Generate structured lecture content grounded in retrieved pages.

**LLM Used:** Lapa LLM (12B Instruct)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `teacher_query` | `str` | Original query |
| **Input** | `matched_topics` | `List[dict]` | Topic context |
| **Input** | `matched_pages` | `List[dict]` | Page content for grounding |
| **Input** | `student_profile` | `Optional[dict]` | For difficulty adaptation |
| **Input** | `grade` | `int` | Grade level |
| **Output** | `lecture_content` | `str` | Generated lecture (Markdown) |
| **Output** | `sources` | `List[str]` | Updated with citations |

**Output Schema - `lecture_content`:**
```markdown
# Topic Title

## Explanation
[Grade-appropriate explanation grounded in textbook pages]

## Key Concepts
- Concept 1: [definition]
- Concept 2: [definition]

## Examples
[Examples from textbook with page references]

## Control Questions
1. [Question to check understanding]
2. [Question to check understanding]

---
Sources: [Page references for grounding]
```

---

### 5. Practice Generator

**Purpose:** Generate 8-12 practice questions with correct answers.

**LLM Used:** MamayLM (12B Gemma-3)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `matched_topics` | `List[dict]` | Topics to cover |
| **Input** | `matched_pages` | `List[dict]` | Context for questions |
| **Input** | `student_profile` | `Optional[dict]` | For difficulty adjustment |
| **Input** | `grade` | `int` | Grade level |
| **Input** | `subject` | `Optional[str]` | Subject for question type |
| **Output** | `practice_questions` | `List[dict]` | Generated questions |

**Output Schema - `practice_questions[i]`:**
```python
{
    "id": str,                      # Unique question ID
    "question_text": str,           # Question content
    "question_type": Literal[       # Question format
        "single_choice",
        "multiple_choice",
        "short_answer",
        "step_by_step"
    ],
    "options": Optional[List[str]], # For choice questions
    "correct_answer": str,          # Expected answer
    "solution_steps": Optional[str],# For step-by-step (Algebra)
    "difficulty": Literal["easy", "medium", "hard"],
    "topic": str,                   # Associated topic
    "source_page": Optional[int]    # Grounding reference
}
```

---

### 6. Solver/Validator

**Purpose:** Self-validate generated questions by solving them.

**LLM Used:** MamayLM + Python REPL (for Algebra)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `practice_questions` | `List[dict]` | Questions to validate |
| **Input** | `subject` | `Optional[str]` | For validation strategy |
| **Output** | `solved_answers` | `List[dict]` | LLM's solutions |
| **Output** | `validation_passed` | `bool` | All questions valid? |
| **Output** | `regeneration_count` | `int` | Attempt counter |

**Output Schema - `solved_answers[i]`:**
```python
{
    "question_id": str,             # Reference to question
    "llm_answer": str,              # LLM's computed answer
    "matches_expected": bool,       # LLM answer == correct_answer?
    "python_verified": Optional[bool], # For Algebra: REPL check
    "confidence": float             # 0-1 confidence score
}
```

**Validation Logic:**
```python
# For Algebra: Use Python REPL
if subject == "Алгебра":
    result = execute_python(question.solution_steps)
    is_valid = (result == question.correct_answer)

# For other subjects: LLM self-check
else:
    llm_answer = mamay.solve(question)
    is_valid = (llm_answer == question.correct_answer)
```

**Conditional Edge:**
- If `validation_passed == False` and `regeneration_count < 3`: → Practice Generator
- Otherwise: → Next node

---

### 7. Check Answers (Conditional)

**Purpose:** Evaluate student-submitted answers.

**LLM Used:** MamayLM

**Condition:** Only runs if `student_answers` is provided.

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `student_answers` | `List[str]` | Student's responses |
| **Input** | `practice_questions` | `List[dict]` | Original questions |
| **Output** | `evaluation_results` | `List[dict]` | Per-question evaluation |

**Output Schema - `evaluation_results[i]`:**
```python
{
    "question_id": str,
    "student_answer": str,
    "correct_answer": str,
    "is_correct": bool,
    "partial_credit": Optional[float],  # 0-1 for partial answers
    "feedback": str,                    # Explanation of errors
    "knowledge_gap": Optional[str]      # Identified gap topic
}
```

---

### 8. Recommendations

**Purpose:** Generate personalized learning recommendations.

**LLM Used:** Lapa LLM

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `evaluation_results` | `List[dict]` | Answer evaluation |
| **Input** | `student_profile` | `Optional[dict]` | Historical data |
| **Input** | `matched_topics` | `List[dict]` | Current topics |
| **Output** | `recommendations` | `str` | Learning advice |

**Output Schema - `recommendations`:**
```markdown
## Summary
[X/Y questions correct - overall performance]

## Errors Analysis
- [Error 1]: [Explanation and correct approach]
- [Error 2]: [Explanation and correct approach]

## Knowledge Gaps
- [Topic/concept needing review]

## Recommended Next Steps
1. [Specific topic to study]
2. [Suggested exercises]
3. [Difficulty adjustment suggestion]
```

---

### 9. Response Finalizer

**Purpose:** Assemble the final API response.

**LLM Used:** None (data assembly)

| Direction | Field | Type | Description |
|-----------|-------|------|-------------|
| **Input** | `lecture_content` | `str` | Generated lecture |
| **Input** | `practice_questions` | `List[dict]` | Validated questions |
| **Input** | `recommendations` | `Optional[str]` | If answers provided |
| **Input** | `sources` | `List[str]` | All citations |
| **Output** | `final_response` | `dict` | Complete response |

**Output Schema - `final_response`:**
```python
{
    "lecture": str,                 # Lecture content
    "questions": List[dict],        # Practice questions (without answers)
    "recommendations": Optional[str],
    "sources": List[str],           # Grounding references
    "metadata": {
        "topics_covered": List[str],
        "difficulty_level": str,
        "question_count": int,
        "grade": int
    }
}
```

---

## API Endpoints - Request/Response

### POST `/tutor/query`

**Request:**
```python
{
    "query": str,                   # Teacher's request
    "grade": int,                   # 8 or 9
    "subject": Optional[str],       # Subject filter
    "student_id": Optional[int],    # For personalization
    "student_answers": Optional[List[str]]  # If checking answers
}
```

**Response:**
```python
{
    "lecture": str,
    "questions": List[QuestionSchema],  # Answers hidden
    "recommendations": Optional[str],
    "sources": List[str]
}
```

### POST `/tutor/check-answers`

**Request:**
```python
{
    "session_id": str,              # Reference to previous query
    "answers": List[str]            # Student's answers
}
```

**Response:**
```python
{
    "results": List[EvaluationResult],
    "score": float,                 # 0-100
    "recommendations": str
}
```

### POST `/benchmark/solve`

**Request:**
```python
{
    "questions": List[BenchmarkQuestion]
}
```

**Response:**
```python
{
    "answers": List[{
        "question_id": str,
        "answer": str
    }]
}
```

---

## Data Sources Summary

| File | Contents | Used By |
|------|----------|---------|
| `toc_for_hackathon_with_subtopics.parquet` | Topics, subtopics, page ranges | Topic Router |
| `pages_for_hackathon.parquet` | Book page text + embeddings | Context Retriever |
| `lms_questions_dev.parquet` | Benchmark questions | Benchmark Solver |
| `benchmark_scores.parquet` | Student performance history | Personalization |
| `benchmark_absences.parquet` | Student attendance data | Personalization |

---

## LLM Usage Summary

| Node | LLM | Purpose |
|------|-----|---------|
| Topic Router | MamayLM / Embeddings | Semantic matching |
| Context Retriever | QwenEmbeddings | Query embedding |
| Content Generator | Lapa LLM | Lecture generation |
| Practice Generator | MamayLM | Question generation |
| Solver/Validator | MamayLM + Python | Answer validation |
| Check Answers | MamayLM | Answer evaluation |
| Recommendations | Lapa LLM | Learning advice |
