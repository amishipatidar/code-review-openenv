"""
FastAPI server — exposes the CodeReviewEnv as an HTTP API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.environment import CodeReviewEnv
from env.models import Action
from env.tasks import list_tasks

app = FastAPI(title="Code Review OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (single-session for hackathon purposes)
_sessions: dict[str, CodeReviewEnv] = {}


class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    line_number: Optional[int] = None
    description: Optional[str] = None
    fixed_code: Optional[str] = None
    quality_score: Optional[float] = None


@app.get("/")
def root():
    return {"message": "Code Review OpenEnv is running 🚀", "tasks": list_tasks()}


@app.get("/tasks")
def tasks():
    return list_tasks()


@app.post("/reset")
def reset(req: ResetRequest):
    env = CodeReviewEnv(task_id=req.task_id)
    obs = env.reset()
    session_id = req.task_id  # simple 1:1 mapping for hackathon
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    action = Action(
        action_type=req.action_type,
        line_number=req.line_number,
        description=req.description,
        fixed_code=req.fixed_code,
        quality_score=req.quality_score,
    )
    result = env.step(action)
    return result.model_dump()


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()


@app.post("/close/{session_id}")
def close(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    final_score = env.close()
    final_score = round(max(0.01, min(0.99, final_score)), 4)
    del _sessions[session_id]
    return {"final_score": final_score}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
