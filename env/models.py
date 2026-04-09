from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class CodeSnippet(BaseModel):
    language: str
    code: str
    filename: str


class Observation(BaseModel):
    task_id: str
    task_description: str
    snippet: CodeSnippet
    step: int
    max_steps: int
    current_review: Dict[str, Any] = {}
    feedback: Optional[str] = None


class Action(BaseModel):
    action_type: str  # "identify_bug" | "suggest_fix" | "rate_quality" | "submit"
    line_number: Optional[int] = None
    description: Optional[str] = None
    fixed_code: Optional[str] = None
    quality_score: Optional[float] = None  # 0.0 - 10.0


# ✅ ADD THIS CLASS (this was missing and causing crash)
class Reward(BaseModel):
    value: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class TaskConfig(BaseModel):
    task_id: str
    difficulty: str
    description: str
    snippet: CodeSnippet
    bug_line: int
    bug_description: str
    fix_keywords: List[str]
    max_steps: int = 5
