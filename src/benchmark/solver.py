import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import time

from src.agents.topic_router import TopicRouter, get_discipline_id
from src.llm.mamay import MamayLLM
from src.llm.lapa import LapaLLM

class BenchmarkSolver:
    """Solver for CodaBench benchmark tasks."""
    
    def __init__(self, fast_mode: bool = False):
        """Initialize the solver.
        
        Args:
            fast_mode: If True, skips some LLM steps in retrieval to save time.
        """
        self.router = TopicRouter()
        # Use Mamay for reasoning (it's the main model)
        self.llm = MamayLLM() 
        self.fast_mode = fast_mode
        print("[BenchmarkSolver] Initialized")

    def load_data(self, path: str) -> pd.DataFrame:
        """Load questions from parquet file."""
        print(f"[BenchmarkSolver] Loading data from {path}...")
        return pd.read_parquet(path)

    def solve(self, input_path: str, output_path: str):
        """Run the main solution pipeline."""
        df = self.load_data(input_path)
        print(f"[BenchmarkSolver] Found {len(df)} questions")
        
        predictions = []
        start_time = time.time()
        
        # Process questions
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Solving"):
            try:
                # Enforce time limit safety (leave 100s buffer)
                if time.time() - start_time > 2300:
                    print("[BenchmarkSolver] ⚠️ Time limit approaching, stopping early")
                    break
                
                answer = self.solve_one(row)
                predictions.append({"id": row["question_id"], "answer": answer})
                
                # Accuracy tracking
                if "correct_answer_indices" in row and row["correct_answer_indices"] is not None:
                    try:
                        indices = row["correct_answer_indices"]
                        if hasattr(indices, 'tolist'): indices = indices.tolist() # Handle numpy array
                        if indices:
                            correct_idx = indices[0]
                            correct_letter = ["A", "B", "C", "D"][correct_idx]
                            if answer == correct_letter:
                                correct_count += 1
                            total_with_truth += 1
                    except Exception:
                        pass # Ignore errors in ground truth checking
                
            except Exception as e:
                print(f"[BenchmarkSolver] Error on question {row.get('question_id')}: {e}")
                # Fallback to random or 'A' in case of error
                predictions.append({"id": row["question_id"], "answer": "A"})

        # Save results
        print(f"[BenchmarkSolver] Saving {len(predictions)} predictions to {output_path}...")
        
        if total_with_truth > 0:
            acc = correct_count / total_with_truth * 100
            print(f"[BenchmarkSolver] 📊 Accuracy: {acc:.2f}% ({correct_count}/{total_with_truth})")
        
        # Ensure directory exists if path has one
        
        # Ensure directory exists if path has one
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(predictions, f)
            
        print("[BenchmarkSolver] Done!")

    def solve_one(self, row: pd.Series) -> str:
        """Solve a single question."""
        question = row["question_text"]
        options = row["answers"] # List ["A", "B", "C", "D"]
        subject = row["global_discipline_name"]
        grade = int(row["grade"])
        
        # 1. Retrieve context
        # Map subject to discipline_id
        discipline_id = get_discipline_id(subject)
        
        # Route logic
        # We use the question as the query
        # Since we have explicit grade/subject, we pass them
        route_result = self.router.route(
            query=question,
            grade=grade,
            discipline_id=discipline_id,
            top_k=2 if self.fast_mode else 3  # Reduced k for speed
        )
        
        context_docs = route_result.get("retrieved_docs", [])
        topic_name = route_result.get("topic", "")
        
        # 2. Form prompt
        context_text = "\n\n".join(context_docs[:3]) # Limit context length
        
        options_text = ""
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            if i < len(letters):
                options_text += f"{letters[i]}) {opt}\n"
        
        prompt = self._build_prompt(question, options_text, context_text, subject, topic_name)
        
        # 3. Generate answer
        # We need a system prompt that enforces strict output format if possible,
        # or we parse the output.
        system_prompt = "Ти - розумний помічник для вирішення шкільних тестів. Твоя задача - вибрати правильну відповідь (A, B, C або D)."
        
        response = self.llm.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.1, # Low temp for precision
            max_tokens=100    # Short answer
        )
        
        # 4. Parse answer
        return self._parse_answer(response)

    def _build_prompt(self, question: str, options: str, context: str, subject: str, topic: str) -> str:
        """Construct the LLM prompt."""
        return f"""
Предмет: {subject}
Тема: {topic}

Контекст з підручника:
{context}

---
Запитання:
{question}

Варіанти відповідей:
{options}

Вкажи правильну відповідь.
Напиши спочатку букву правильної відповіді (A, B, C або D), а потім коротке пояснення.
Відповідь:"""

    def _parse_answer(self, response: str) -> str:
        """Parse LLM response to extract A, B, C, or D."""
        clean = response.strip().upper()
        
        # Check patterns like "A", "A)", "A.", "ВІДПОВІДЬ: A"
        import re
        
        # Look for explicit letter at start
        match = re.match(r"^([ABCD])[\.\)\s]", clean)
        if match:
            return match.group(1)
            
        if clean in ["A", "B", "C", "D"]:
            return clean
            
        # Search for "Відповідь: X"
        match = re.search(r"ВІДПОВІДЬ:?\s*([ABCD])", clean)
        if match:
            return match.group(1)
            
        # Fallback: look for the first occurrence of a letter pattern
        match = re.search(r"\b([ABCD])\)", clean)
        if match:
            return match.group(1)
            
        return "A" # Ultimate fallback
