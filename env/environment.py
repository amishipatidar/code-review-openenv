"""
Code Review Environment — OpenEnv-compliant implementation.

Implements: reset() → Observation
            step(Action) → (Observation, reward, done, info)
            state() → dict
            close() → float
"""
import copy
from typing import Optional
from env.models import Observation, Action, Reward, StepResult, TaskConfig
from env.tasks import get_task
from env.graders import compute_total_reward, final_grade

# Penalty constants
PENALTY_REPEATED_ACTION  = -0.05
PENALTY_INVALID_ACTION   = -0.05
PENALTY_WASTED_STEP      = -0.02


class CodeReviewEnv:
    def __init__(self, task_id: str):
        self.task: TaskConfig = get_task(task_id)
        self._step: int = 0
        self._done: bool = False
        self._review: dict = {}
        self._last_reward: float = 0.0
        self._reward_history: list = []
        self._action_counts: dict = {}
        self._invalid_count: int = 0

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self._step = 0
        self._done = False
        self._review = {}
        self._last_reward = 0.0
        self._reward_history = []
        self._action_counts = {}
        self._invalid_count = 0
        return self._make_observation(feedback="Task started. Begin your code review.")

    def step(self, action: Action) -> StepResult:
        """Apply action, compute reward with penalties, return StepResult."""
        if self._done:
            obs = self._make_observation(feedback="Episode already finished.")
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        self._step += 1
        penalty = 0.0
        penalty_reason = None
        atype = action.action_type
        valid_actions = {"identify_bug", "suggest_fix", "rate_quality", "submit"}

        if atype not in valid_actions:
            penalty = PENALTY_INVALID_ACTION
            penalty_reason = f"Invalid action '{atype}'. Penalty: {PENALTY_INVALID_ACTION}."
            self._invalid_count += 1
            if self._invalid_count >= 3:
                self._done = True
                feedback = "Too many invalid actions. Episode terminated early."
                obs = self._make_observation(feedback=feedback)
                self._reward_history.append(penalty)
                reward_obj = Reward(value=penalty, cumulative=max(self._last_reward + penalty, 0.0),
                                    penalty=penalty, penalty_reason=penalty_reason, breakdown={})
                return StepResult(observation=obs, reward=penalty, done=True,
                                  info={"reward_obj": reward_obj.model_dump()})
        else:
            self._action_counts[atype] = self._action_counts.get(atype, 0) + 1
            if self._action_counts[atype] > 1 and atype != "submit":
                penalty = PENALTY_REPEATED_ACTION
                penalty_reason = f"Repeated action '{atype}'. Penalty: {PENALTY_REPEATED_ACTION}."

            review_complete = all(k in self._review for k in ["bug_line", "fix", "quality_score"])
            if review_complete and atype != "submit":
                penalty = min(penalty, PENALTY_WASTED_STEP)
                penalty_reason = (penalty_reason or "") + f" Wasted step. Penalty: {PENALTY_WASTED_STEP}."

        feedback = self._apply_action(action)
        if penalty_reason:
            feedback = f"⚠️ {penalty_reason} | {feedback}"

        base_reward, breakdown = compute_total_reward(
            self.task, self._review, self._step, self.task.max_steps
        )
        incremental = round(base_reward - self._last_reward + penalty, 2)
        self._last_reward = max(base_reward, 0.0)
        self._reward_history.append(incremental)

        if atype == "submit" or self._step >= self.task.max_steps:
            self._done = True

        obs = self._make_observation(feedback=feedback)
        reward_obj = Reward(
            value=incremental,
            cumulative=round(self._last_reward, 2),
            penalty=round(penalty, 2),
            penalty_reason=penalty_reason,
            breakdown=breakdown,
        )
        return StepResult(
            observation=obs, reward=incremental, done=self._done,
            info={"reward_obj": reward_obj.model_dump(), "cumulative_reward": self._last_reward},
        )

    def state(self) -> dict:
        """Return full current state snapshot."""
        return {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "step": self._step,
            "max_steps": self.task.max_steps,
            "done": self._done,
            "review": copy.deepcopy(self._review),
            "cumulative_reward": self._last_reward,
            "reward_history": list(self._reward_history),
            "action_counts": dict(self._action_counts),
            "invalid_count": self._invalid_count,
        }

    def close(self) -> float:
        """Finalise and return final 0.0–1.0 score."""
        self._done = True
        return final_grade(self.task, self._review, self._step, self.task.max_steps)

    def _apply_action(self, action: Action) -> str:
        atype = action.action_type
        if atype == "identify_bug":
            self._review["bug_line"] = action.line_number
            self._review["bug_description"] = action.description or ""
            return (f"Bug recorded at line {action.line_number}. "
                    "Next: suggest_fix.")
        elif atype == "suggest_fix":
            self._review["fix"] = action.fixed_code or action.description or ""
            return "Fix recorded. Next: rate_quality (0–10)."
        elif atype == "rate_quality":
            score = action.quality_score
            if score is None or not (0.0 <= float(score) <= 10.0):
                return "Invalid score. Must be 0.0–10.0."
            self._review["quality_score"] = score
            return f"Quality {score}/10 recorded. Next: submit."
        elif atype == "submit":
            return "Review submitted. Episode complete."
        else:
            return f"Unknown action '{atype}'."

    def _make_observation(self, feedback: Optional[str] = None) -> Observation:
        return Observation(
            task_id=self.task.task_id,
            task_description=self.task.description,
            snippet=self.task.snippet,
            step=self._step,
            max_steps=self.task.max_steps,
            current_review=copy.deepcopy(self._review),
            feedback=feedback,
        )
