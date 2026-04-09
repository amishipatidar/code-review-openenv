"""
Graders for the Code Review environment.
Each grader returns a score STRICTLY between 0.0 and 1.0 (exclusive).
Scores are clamped to [0.01, 0.99] to satisfy validator requirements.
"""
from env.models import TaskConfig


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0.0, 1.0) — never 0.0 or 1.0 exactly."""
    return round(max(0.01, min(0.99, score)), 4)


def grade_bug_identification(task: TaskConfig, review: dict) -> tuple:
    """Score bug identification: correct line ±2 lines gets partial credit."""
    identified_line = review.get("bug_line")
    description = (review.get("bug_description") or "").lower()

    if identified_line is None:
        return _clamp(0.01), "No bug line identified."

    line_score = 0.01
    if identified_line == task.bug_line:
        line_score = 0.99
    elif abs(identified_line - task.bug_line) <= 2:
        line_score = 0.5

    # Check description quality
    bug_keywords = task.bug_description.lower().split()
    important = [w for w in bug_keywords if len(w) > 4]
    desc_score = 0.01
    if important:
        matches = sum(1 for kw in important if kw in description)
        desc_score = _clamp(matches / max(len(important) * 0.4, 1))

    score = _clamp(0.6 * line_score + 0.4 * desc_score)
    return score, f"Line score: {line_score:.2f}, Desc score: {desc_score:.2f}"


def grade_fix_suggestion(task: TaskConfig, review: dict) -> tuple:
    """Score fix suggestion based on presence of key fix terms."""
    fix = (review.get("fix") or "").lower()
    if not fix:
        return _clamp(0.01), "No fix suggested."

    matches = sum(1 for kw in task.fix_keywords if kw.lower() in fix)
    score = _clamp(matches / max(len(task.fix_keywords), 1))
    return score, f"Fix keyword matches: {matches}/{len(task.fix_keywords)}"


def grade_quality_rating(review: dict) -> tuple:
    """Score quality rating — valid number gets near-full credit."""
    rating = review.get("quality_score")
    if rating is None:
        return _clamp(0.01), "No quality rating provided."
    if 0.0 <= float(rating) <= 10.0:
        return _clamp(0.98), f"Valid quality rating: {rating}"
    return _clamp(0.01), "Quality rating out of range (must be 0–10)."


def compute_total_reward(task: TaskConfig, review: dict, step: int, max_steps: int) -> tuple:
    """
    Compute reward. Weights: bug_id=40%, fix=40%, quality=20%.
    All scores strictly within (0.0, 1.0).
    """
    bug_score,  bug_feedback  = grade_bug_identification(task, review)
    fix_score,  fix_feedback  = grade_fix_suggestion(task, review)
    qual_score, qual_feedback = grade_quality_rating(review)

    base = 0.4 * bug_score + 0.4 * fix_score + 0.2 * qual_score

    # Small efficiency bonus for finishing early when doing well
    step_fraction = step / max_steps
    efficiency = max(0.0, 0.05 * (1 - step_fraction)) if base > 0.5 else 0.0

    total = _clamp(base + efficiency)

    breakdown = {
        "bug_identification": bug_score,
        "bug_feedback": bug_feedback,
        "fix_suggestion": fix_score,
        "fix_feedback": fix_feedback,
        "quality_rating": qual_score,
        "qual_feedback": qual_feedback,
        "efficiency_bonus": round(efficiency, 4),
        "total": total,
    }
    return total, breakdown


def final_grade(task: TaskConfig, review: dict, steps_taken: int, max_steps: int) -> float:
    """Return final score strictly within (0.0, 1.0)."""
    total, _ = compute_total_reward(task, review, steps_taken, max_steps)
    return _clamp(total)
