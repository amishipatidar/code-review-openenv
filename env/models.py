from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class CodeSnippet(BaseModel):
    language: str
    code: str
    filename: str


class Observation(BaseModel):
    """Typed observation returned by reset() and step()."""
    task_id: str
    task_description: str
    snippet: CodeSnippet
    step: int
    max_steps: int
    current_review: Dict[str, Any] = {}
    feedback: Optional[str] = None


class Action(BaseModel):
    """Typed action submitted by the agent."""
    action_type: str = Field(
        ...,
        description="One of: identify_bug | suggest_fix | rate_quality | submit"
    )
    line_number: Optional[int] = Field(None, description="Bug line number (for identify_bug)")
    description: Optional[str] = Field(None, description="Bug description or fix notes")
    fixed_code: Optional[str] = Field(None, description="Corrected code (for suggest_fix)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Code quality rating 0–10")


class Reward(BaseModel):
    """Typed reward signal returned after each step."""
    value: float = Field(..., ge=-1.0, le=1.0, description="Incremental reward for this step")
    cumulative: float = Field(..., ge=0.0, le=1.0, description="Total reward so far")
    penalty: float = Field(0.0, description="Penalty applied this step (negative contribution)")
    penalty_reason: Optional[str] = Field(None, description="Why a penalty was applied")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Score component breakdown")


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
