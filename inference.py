"""
inference.py — Code Review OpenEnv agent.
Reads tasks from the environment server, uses an LLM to review code,
and emits [START] / [STEP] / [END] lines to stdout.
"""
import os
import sys
import json
import requests
from openai import OpenAI

# ── Environment variables (all with defaults as required by spec) ─────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "https://amishipatidar-code-review-env.hf.space")

# ── Validate required env vars ────────────────────────────────────────────────
if HF_TOKEN is None:
    print("[END] success=false steps=0 rewards=", flush=True)
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client (wrapped in try/except as required) ────────────────────────
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception as e:
    print(f"[END] success=false steps=0 rewards=", flush=True)
    raise RuntimeError(f"Failed to initialise OpenAI client: {e}") from e

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be shown a code snippet and asked to:
1. Identify the bug (line number + description)
2. Suggest a fix
3. Rate the overall code quality (0–10)

Always respond in valid JSON matching the action requested. No markdown, no extra text.
"""


def call_llm(messages: list) -> str:
    """Call the LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"error": str(e)})


def parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    try:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip().rstrip("```").strip()
        return json.loads(text)
    except Exception:
        return {}


def build_code_context(obs: dict) -> str:
    """Format the code snippet with line numbers."""
    try:
        snippet = obs["snippet"]
        lines = snippet["code"].split("\n")
        numbered = "\n".join(f"{i+1:3}: {line}" for i, line in enumerate(lines))
        return f"File: {snippet['filename']} ({snippet['language']})\n\n{numbered}"
    except Exception:
        return str(obs.get("snippet", ""))


def env_post(path: str, payload: dict) -> dict:
    """POST to the environment server with error handling."""
    try:
        resp = requests.post(f"{ENV_URL}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def run_episode(task_id: str) -> dict:
    """Run one full episode. Always emits [START]...[STEP]...[END]."""
    task_name  = task_id
    benchmark  = "code-review"
    step_n     = 0
    rewards    = []
    done       = False
    success    = False
    last_error = None

    # ── Reset ────────────────────────────────────────────────────────────────
    try:
        data       = env_post("/reset", {"task_id": task_id})
        session_id = data.get("session_id", task_id)
        obs        = data.get("observation", {})
    except Exception as e:
        last_error = str(e)
        print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 rewards= ", flush=True)
        return {"task_id": task_id, "success": False, "steps": 0, "rewards": []}

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)

    code_ctx     = build_code_context(obs)
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Step 1: identify_bug ─────────────────────────────────────────────────
    try:
        user_msg = (
            f"Task: {obs.get('task_description', '')}\n\n"
            f"{code_ctx}\n\n"
            "Identify the bug. Respond ONLY with JSON:\n"
            '{"line_number": <int>, "description": "<string>"}'
        )
        conversation.append({"role": "user", "content": user_msg})
        raw = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw})
        parsed = parse_json(raw)
        payload1 = {
            "session_id": session_id,
            "action_type": "identify_bug",
            "line_number": int(parsed.get("line_number", 0)),
            "description": parsed.get("description", ""),
        }
        last_error = None
    except Exception as e:
        payload1   = {"session_id": session_id, "action_type": "identify_bug"}
        last_error = str(e)

    result1 = env_post("/step", payload1)
    step_n += 1
    reward1  = result1.get("reward", 0.0)
    done     = result1.get("done", False)
    rewards.append(reward1)
    action_str1 = f"identify_bug(line={payload1.get('line_number','?')})"
    error_str1  = last_error or result1.get("error") or "null"
    print(f"[STEP] step={step_n} action={action_str1} reward={reward1:.2f} done={str(done).lower()} error={error_str1}", flush=True)

    if done:
        success = sum(rewards) >= 0.5
        print(f"[END] success={str(success).lower()} steps={step_n} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
        return {"task_id": task_id, "success": success, "steps": step_n, "rewards": rewards}

    # ── Step 2: suggest_fix ──────────────────────────────────────────────────
    try:
        obs2 = result1.get("observation", {})
        conversation.append({
            "role": "user",
            "content": (
                f"Feedback: {obs2.get('feedback','')}\n\n"
                "Suggest a fix. Respond ONLY with JSON:\n"
                '{"fixed_code": "<corrected line or block>"}'
            ),
        })
        raw2   = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw2})
        parsed2 = parse_json(raw2)
        payload2 = {
            "session_id": session_id,
            "action_type": "suggest_fix",
            "fixed_code": parsed2.get("fixed_code", ""),
        }
        last_error = None
    except Exception as e:
        payload2   = {"session_id": session_id, "action_type": "suggest_fix", "fixed_code": ""}
        last_error = str(e)

    result2 = env_post("/step", payload2)
    step_n += 1
    reward2  = result2.get("reward", 0.0)
    done     = result2.get("done", False)
    rewards.append(reward2)
    error_str2 = last_error or result2.get("error") or "null"
    print(f"[STEP] step={step_n} action=suggest_fix(code=...) reward={reward2:.2f} done={str(done).lower()} error={error_str2}", flush=True)

    if done:
        success = sum(rewards) >= 0.5
        print(f"[END] success={str(success).lower()} steps={step_n} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
        return {"task_id": task_id, "success": success, "steps": step_n, "rewards": rewards}

    # ── Step 3: rate_quality ─────────────────────────────────────────────────
    try:
        obs3 = result2.get("observation", {})
        conversation.append({
            "role": "user",
            "content": (
                f"Feedback: {obs3.get('feedback','')}\n\n"
                "Rate the code quality 0–10. Respond ONLY with JSON:\n"
                '{"quality_score": <float>}'
            ),
        })
        raw3   = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw3})
        parsed3    = parse_json(raw3)
        score_val  = float(parsed3.get("quality_score", 5.0))
        score_val  = max(0.0, min(10.0, score_val))
        payload3   = {
            "session_id": session_id,
            "action_type": "rate_quality",
            "quality_score": score_val,
        }
        last_error = None
    except Exception as e:
        payload3   = {"session_id": session_id, "action_type": "rate_quality", "quality_score": 5.0}
        last_error = str(e)

    result3 = env_post("/step", payload3)
    step_n += 1
    reward3  = result3.get("reward", 0.0)
    done     = result3.get("done", False)
    rewards.append(reward3)
    score_used = payload3.get("quality_score", 5.0)
    error_str3 = last_error or result3.get("error") or "null"
    print(f"[STEP] step={step_n} action=rate_quality(score={score_used}) reward={reward3:.2f} done={str(done).lower()} error={error_str3}", flush=True)

    # ── Step 4: submit ───────────────────────────────────────────────────────
    result4 = env_post("/step", {"session_id": session_id, "action_type": "submit"})
    step_n += 1
    reward4  = result4.get("reward", 0.0)
    done     = result4.get("done", True)
    rewards.append(reward4)
    print(f"[STEP] step={step_n} action=submit() reward={reward4:.2f} done={str(done).lower()} error=null", flush=True)

    # ── Close ────────────────────────────────────────────────────────────────
    try:
        close_resp   = requests.post(f"{ENV_URL}/close/{session_id}", timeout=30)
        final_score  = close_resp.json().get("final_score", sum(rewards))
    except Exception:
        final_score = sum(rewards)

    success = final_score >= 0.5
    print(f"[END] success={str(success).lower()} steps={step_n} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
    return {"task_id": task_id, "success": success, "steps": step_n,
            "rewards": rewards, "final_score": final_score}


if __name__ == "__main__":
    task_ids = ["easy_off_by_one", "medium_logic_error", "hard_security_flaw"]
    results  = []

    for tid in task_ids:
        try:
            result = run_episode(tid)
        except Exception as e:
            # Always emit [END] even on unexpected crash
            print(f"[END] success=false steps=0 rewards=0.00", flush=True)
            result = {"task_id": tid, "success": False, "steps": 0,
                      "rewards": [], "error": str(e)}
        results.append(result)

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        status = "✅" if r.get("success") else "❌"
        score  = r.get("final_score", "N/A")
        print(f"{status} {r['task_id']}: final_score={score} steps={r['steps']}", flush=True)
