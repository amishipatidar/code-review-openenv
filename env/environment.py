"""
Code Review Environment — OpenEnv-compliant implementation.
All reward values are clamped strictly within (0.0, 1.0).
"""
import copy
from typing import Optional
from env.models import Observation, Action, Reward, StepResult, TaskConfig
from env.tasks import get_task
from env.graders import compute_total_reward, final_grade

PENALTY_REPEATED_ACTION = -0.05
PENALTY_INVALID_ACTION  = -0.05
PENALTY_WASTED_STEP     = -0.02


def _clamp_reward(value: float) -> float:
    """Clamp any reward to strictly (0.01, 0.99) — never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, value)), 4)


class CodeReviewEnv:
    def __init__(self, task_id: str):
        self.task: TaskConfig = get_task(task_id)
        self._step: int = 0
        self._done: bool = False
        self._review: dict = {}
        self._last_reward: float = 0.01
        self._reward_history: list = []
        self._action_counts: dict = {}
        self._invalid_count: int = 0

    def reset(self) -> Observation:
        self._step = 0
        self._done = False
        self._review = {}
        self._last_reward = 0.01
        self._reward_history = []
        self._action_counts = {}
        self._invalid_count = 0
        return self._make_observation(feedback="Task started. Begin your code review.")

    def step(self, action: Action) -> StepResult:
        """Apply action and return StepResult. All rewards strictly within (0.0, 1.0)."""
        if self._done:
            obs = self._make_observation(feedback="Episode already finished.")
            safe_reward = _clamp_reward(self._last_reward)
            return StepResult(observation=obs, reward=safe_reward, done=True, info={})

        self._step += 1
        penalty = 0.0
        penalty_reason = None
        atype = action.action_type
        valid_actions = {"identify_bug", "suggest_fix", "rate_quality", "submit"}

        if atype not in valid_actions:
            penalty = PENALTY_INVALID_ACTION
            penalty_reason = f"Invalid action '{atype}'. Penalty applied."
            self._invalid_count += 1
            if self._invalid_count >= 3:
                self._done = True
                obs = self._make_observation(feedback="Too many invalid actions. Episode terminated.")
                safe_reward = _clamp_reward(self._last_reward + penalty)
                self._reward_history.append(safe_reward)
                reward_obj = Reward(
                    value=safe_reward, cumulative=safe_reward,
                    penalty=_clamp_reward(abs(penalty)),
                    penalty_reason=penalty_reason, breakdown={}
                )
                return StepResult(observation=obs, reward=safe_reward, done=True,
                                  info={"reward_obj": reward_obj.model_dump()})
        else:
            self._action_counts[atype] = self._action_counts.get(atype, 0) + 1
            if self._action_counts[atype] > 1 and atype != "submit":
                penalty = PENALTY_REPEATED_ACTION
                penalty_reason = f"Repeated action '{atype}'."
            review_complete = all(k in self._review for k in ["bug_line", "fix", "quality_score"])
            if review_complete and atype != "submit":
                penalty = min(penalty, PENALTY_WASTED_STEP)
                penalty_reason = (penalty_reason or "") + " Wasted step."

        feedback = self._apply_action(action)
        if penalty_reason:
            feedback = f"⚠️ {penalty_reason} | {feedback}"

        base_reward, breakdown = compute_total_reward(
            self.task, self._review, self._step, self.task.max_steps
        )

        # Always clamp — never allow 0.0 or 1.0
        base_reward = _clamp_reward(base_reward)
        incremental = _clamp_reward(base_reward - self._last_reward + penalty)
        self._last_reward = base_reward
        self._reward_history.append(incremental)

        if atype == "submit" or self._step >= self.task.max_steps:
            self._done = True

        obs = self._make_observation(feedback=feedback)
        reward_obj = Reward(
            value=incremental,
            cumulative=_clamp_reward(self._last_reward),
            penalty=_clamp_reward(abs(penalty)) if penalty < 0 else 0.0,
            penalty_reason=penalty_reason,
            breakdown=breakdown,
        )
        return StepResult(
            observation=obs,
            reward=incremental,
            done=self._done,
            info={"reward_obj": reward_obj.model_dump(), "cumulative_reward": self._last_reward},
        )

    def state(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "step": self._step,
            "max_steps": self.task.max_steps,
            "done": self._done,
            "review": copy.deepcopy(self._review),
            "cumulative_reward": _clamp_reward(self._last_reward),
            "reward_history": [_clamp_reward(r) for r in self._reward_history],
            "action_counts": dict(self._action_counts),
            "invalid_count": self._invalid_count,
        }

    def close(self) -> float:
        self._done = True
        return _clamp_reward(
            final_grade(self.task, self._review, self._step, self.task.max_steps)
        )

    def _apply_action(self, action: Action) -> str:
        atype = action.action_type
        if atype == "identify_bug":
            self._review["bug_line"] = action.line_number
            self._review["bug_description"] = action.description or ""
            return f"Bug recorded at line {action.line_number}. Next: suggest_fix."
        elif atype == "suggest_fix":
            self._review["fix"] = action.fixed_code or action.description or ""
            return "Fix recorded. Next: rate_quality (0-10)."
        elif atype == "rate_quality":
            score = action.quality_score
            if score is None or not (0.0 <= float(score) <= 10.0):
                return "Invalid score. Must be 0.0-10.0."
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
