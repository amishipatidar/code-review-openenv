"""
Graders for the Code Review environment.
Each grader returns a score between 0.0 and 1.0.
"""
from env.models import TaskConfig


def grade_bug_identification(task: TaskConfig, review: dict) -> tuple[float, str]:
    """Score bug identification: correct line ±2 lines gets partial credit."""
    identified_line = review.get("bug_line")
    description = (review.get("bug_description") or "").lower()

    if identified_line is None:
        return 0.0, "No bug line identified."

    line_score = 0.0
    if identified_line == task.bug_line:
        line_score = 1.0
    elif abs(identified_line - task.bug_line) <= 2:
        line_score = 0.5

    # Check description quality — look for key terms
    desc_score = 0.0
    bug_keywords = task.bug_description.lower().split()
    important = [w for w in bug_keywords if len(w) > 4]
    if important:
        matches = sum(1 for kw in important if kw in description)
        desc_score = min(matches / max(len(important) * 0.4, 1), 1.0)

    score = 0.6 * line_score + 0.4 * desc_score
    feedback = f"Line score: {line_score:.1f}, Description score: {desc_score:.1f}"
    return round(score, 2), feedback


def grade_fix_suggestion(task: TaskConfig, review: dict) -> tuple[float, str]:
    """Score fix suggestion based on presence of key fix terms."""
    fix = (review.get("fix") or "").lower()
    if not fix:
        return 0.0, "No fix suggested."

    matches = sum(1 for kw in task.fix_keywords if kw.lower() in fix)
    score = min(matches / max(len(task.fix_keywords), 1), 1.0)
    return round(score, 2), f"Fix keyword matches: {matches}/{len(task.fix_keywords)}"


def grade_quality_rating(review: dict) -> tuple[float, str]:
    """Score quality rating — just check it's a valid number."""
    rating = review.get("quality_score")
    if rating is None:
        return 0.0, "No quality rating provided."
    if 0.0 <= float(rating) <= 10.0:
        return 1.0, f"Valid quality rating: {rating}"
    return 0.0, "Quality rating out of range (must be 0–10)."


def compute_total_reward(task: TaskConfig, review: dict, step: int, max_steps: int) -> tuple[float, dict]:
    """
    Compute incremental reward based on what has been done so far.
    Weights: bug_id=0.4, fix=0.4, quality=0.2
    Penalises wasted steps slightly.
    """
    bug_score, bug_feedback = grade_bug_identification(task, review)
    fix_score, fix_feedback = grade_fix_suggestion(task, review)
    qual_score, qual_feedback = grade_quality_rating(review)

    base = 0.4 * bug_score + 0.4 * fix_score + 0.2 * qual_score

    # Efficiency bonus: finish early = small bonus
    step_fraction = step / max_steps
    efficiency = max(0.0, 0.1 * (1 - step_fraction)) if base > 0.5 else 0.0

    total = round(min(base + efficiency, 1.0), 2)

    breakdown = {
        "bug_identification": bug_score,
        "bug_feedback": bug_feedback,
        "fix_suggestion": fix_score,
        "fix_feedback": fix_feedback,
        "quality_rating": qual_score,
        "qual_feedback": qual_feedback,
        "efficiency_bonus": round(efficiency, 2),
        "total": total,
    }
    return total, breakdown


def final_grade(task: TaskConfig, review: dict, steps_taken: int, max_steps: int) -> float:
    """Return final 0.0–1.0 grade for the episode."""
    total, _ = compute_total_reward(task, review, steps_taken, max_steps)
    return total
