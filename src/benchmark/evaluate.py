"""Evaluation utilities for benchmark results."""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Metrics for benchmark evaluation."""
    accuracy: float
    correct: int
    total: int
    by_subject: Dict[str, Dict[str, Any]]
    by_grade: Dict[int, Dict[str, Any]]
    

def load_ground_truth(parquet_path: Path) -> Dict[str, int]:
    """Load ground truth answers from parquet file.
    
    Args:
        parquet_path: Path to questions parquet
        
    Returns:
        Dict mapping question_id -> correct_answer_index
    """
    df = pd.read_parquet(parquet_path)
    ground_truth = {}
    
    for _, row in df.iterrows():
        qid = row["question_id"]
        correct_indices = row.get("correct_answer_indices", [])
        if correct_indices:
            ground_truth[qid] = correct_indices[0]
    
    return ground_truth


def load_predictions(json_path: Path) -> Dict[str, int]:
    """Load predictions from JSON file.
    
    Args:
        json_path: Path to predictions JSON
        
    Returns:
        Dict mapping question_id -> predicted_answer_index
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    predictions = {}
    
    # Handle different formats
    if isinstance(data, dict) and "results" in data:
        # Full results format
        for r in data["results"]:
            predictions[r["question_id"]] = r["predicted_answer_index"]
    elif isinstance(data, dict):
        # CodaBench format (question_id -> answer_letter)
        for qid, answer in data.items():
            if isinstance(answer, str):
                predictions[qid] = ord(answer.upper()) - ord('A')
            else:
                predictions[qid] = answer
    
    return predictions


def evaluate_predictions(
    predictions: Dict[str, int],
    ground_truth: Dict[str, int],
    questions_df: Optional[pd.DataFrame] = None
) -> EvaluationMetrics:
    """Evaluate predictions against ground truth.
    
    Args:
        predictions: Dict mapping question_id -> predicted_answer_index
        ground_truth: Dict mapping question_id -> correct_answer_index
        questions_df: Optional DataFrame with question metadata for breakdown
        
    Returns:
        EvaluationMetrics with accuracy and breakdowns
    """
    correct = 0
    total = 0
    by_subject = {}
    by_grade = {}
    
    for qid, pred in predictions.items():
        if qid not in ground_truth:
            continue
        
        is_correct = pred == ground_truth[qid]
        correct += int(is_correct)
        total += 1
        
        # Get metadata if available
        if questions_df is not None:
            row = questions_df[questions_df["question_id"] == qid]
            if len(row) > 0:
                row = row.iloc[0]
                
                # By subject
                subj = row.get("global_discipline_name", "Unknown")
                if subj not in by_subject:
                    by_subject[subj] = {"correct": 0, "total": 0}
                by_subject[subj]["total"] += 1
                by_subject[subj]["correct"] += int(is_correct)
                
                # By grade
                grade = int(row.get("grade", 0))
                if grade not in by_grade:
                    by_grade[grade] = {"correct": 0, "total": 0}
                by_grade[grade]["total"] += 1
                by_grade[grade]["correct"] += int(is_correct)
    
    # Calculate accuracies
    for d in [by_subject, by_grade]:
        for k in d:
            d[k]["accuracy"] = d[k]["correct"] / d[k]["total"] if d[k]["total"] else 0
    
    return EvaluationMetrics(
        accuracy=correct / total if total else 0,
        correct=correct,
        total=total,
        by_subject=by_subject,
        by_grade={int(k): v for k, v in by_grade.items()}
    )


def print_evaluation_report(metrics: EvaluationMetrics) -> None:
    """Print a formatted evaluation report.
    
    Args:
        metrics: EvaluationMetrics to display
    """
    print("=" * 50)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 50)
    print(f"\nOverall Accuracy: {metrics.accuracy:.2%}")
    print(f"Correct: {metrics.correct} / {metrics.total}")
    
    if metrics.by_subject:
        print("\n--- By Subject ---")
        for subj, m in sorted(metrics.by_subject.items()):
            print(f"  {subj}: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")
    
    if metrics.by_grade:
        print("\n--- By Grade ---")
        for grade, m in sorted(metrics.by_grade.items()):
            print(f"  Grade {grade}: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")
    
    print("=" * 50)


def main():
    """CLI entry point for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate benchmark predictions")
    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        required=True,
        help="Path to predictions JSON"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=Path,
        help="Path to ground truth parquet (default: from config)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for evaluation report JSON"
    )
    
    args = parser.parse_args()
    
    from ..config import get_settings
    settings = get_settings()
    
    gt_path = args.ground_truth or settings.questions_parquet_path
    
    # Load data
    print(f"Loading ground truth from {gt_path}")
    ground_truth = load_ground_truth(gt_path)
    questions_df = pd.read_parquet(gt_path)
    
    print(f"Loading predictions from {args.predictions}")
    predictions = load_predictions(args.predictions)
    
    print(f"Evaluating {len(predictions)} predictions against {len(ground_truth)} ground truth")
    
    # Evaluate
    metrics = evaluate_predictions(predictions, ground_truth, questions_df)
    print_evaluation_report(metrics)
    
    # Save if output specified
    if args.output:
        report = {
            "accuracy": metrics.accuracy,
            "correct": metrics.correct,
            "total": metrics.total,
            "by_subject": metrics.by_subject,
            "by_grade": metrics.by_grade
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
