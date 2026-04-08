import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be shown a code snippet and asked to:
1. Identify the bug (line number + description)
2. Suggest a fix
3. Rate the overall code quality (0–10)
Always respond in valid JSON matching the action requested.
"""

def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()
    return json.loads(text)

def build_code_context(obs: dict) -> str:
    snippet = obs["snippet"]
    lines = snippet["code"].split("\n")
    numbered = "\n".join(f"{i+1:3}: {line}" for i, line in enumerate(lines))
    return f"File: {snippet['filename']} ({snippet['language']})\n\n{numbered}"

def safe_error(err):
    if err is None:
        return "null"
    return str(err).replace("\n", " ").replace("\r", " ")

def run_episode(task_id: str):

    step_n = 0
    rewards = []
    success = False

    try:
        # RESET
        resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        obs = data["observation"]

        print(f"[START] task={obs['task_id']} env=code-review model={MODEL_NAME}", flush=True)

        code_ctx = build_code_context(obs)
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        # STEP 1
        user_msg = f"""Task: {obs['task_description']}

{code_ctx}

Identify the bug. Respond ONLY with JSON:
{{"line_number": <int>, "description": "<string>"}}"""

        conversation.append({"role": "user", "content": user_msg})
        raw = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw})

        try:
            parsed = parse_json(raw)
            payload = {
                "session_id": session_id,
                "action_type": "identify_bug",
                "line_number": int(parsed.get("line_number", 0)),
                "description": parsed.get("description", "")
            }
            err = None
        except Exception as e:
            payload = {"session_id": session_id, "action_type": "identify_bug"}
            err = e

        res = requests.post(f"{ENV_URL}/step", json=payload).json()
        step_n += 1
        rewards.append(res["reward"])

        print(f"[STEP] step={step_n} action=identify_bug(...) reward={res['reward']:.2f} done={str(res['done']).lower()} error={safe_error(err)}", flush=True)

        if res["done"]:
            success = res["reward"] > 0.5
            return step_n, rewards, success

        # STEP 2
        obs2 = res["observation"]
        conversation.append({
            "role": "user",
            "content": f"""Environment feedback: {obs2['feedback']}

Suggest a fix. Respond ONLY with JSON:
{{"fixed_code": "<code>"}}"""
        })

        raw2 = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw2})

        try:
            parsed2 = parse_json(raw2)
            payload2 = {
                "session_id": session_id,
                "action_type": "suggest_fix",
                "fixed_code": parsed2.get("fixed_code", "")
            }
            err2 = None
        except Exception as e:
            payload2 = {"session_id": session_id, "action_type": "suggest_fix", "fixed_code": ""}
            err2 = e

        res2 = requests.post(f"{ENV_URL}/step", json=payload2).json()
        step_n += 1
        rewards.append(res2["reward"])

        print(f"[STEP] step={step_n} action=suggest_fix(...) reward={res2['reward']:.2f} done={str(res2['done']).lower()} error={safe_error(err2)}", flush=True)

        if res2["done"]:
            success = sum(rewards) > 0.5
            return step_n, rewards, success

        # STEP 3
        obs3 = res2["observation"]
        conversation.append({
            "role": "user",
            "content": f"""Environment feedback: {obs3['feedback']}

Rate quality 0-10. Respond ONLY with JSON:
{{"quality_score": <float>}}"""
        })

        raw3 = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw3})

        try:
            parsed3 = parse_json(raw3)
            payload3 = {
                "session_id": session_id,
                "action_type": "rate_quality",
                "quality_score": float(parsed3.get("quality_score", 5))
            }
            err3 = None
        except Exception as e:
            payload3 = {"session_id": session_id, "action_type": "rate_quality", "quality_score": 5}
            err3 = e

        res3 = requests.post(f"{ENV_URL}/step", json=payload3).json()
        step_n += 1
        rewards.append(res3["reward"])

        print(f"[STEP] step={step_n} action=rate_quality(...) reward={res3['reward']:.2f} done={str(res3['done']).lower()} error={safe_error(err3)}", flush=True)

        # STEP 4
        res4 = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, "action_type": "submit"}).json()
        step_n += 1
        rewards.append(res4["reward"])

        print(f"[STEP] step={step_n} action=submit() reward={res4['reward']:.2f} done={str(res4['done']).lower()} error=null", flush=True)

        # CLOSE
        close = requests.post(f"{ENV_URL}/close/{session_id}").json()
        success = close.get("final_score", 0.0) >= 0.5

    except Exception as e:
        print(f"[STEP] step={step_n} action=null reward=0.00 done=true error={safe_error(e)}", flush=True)

    finally:
        print(f"[END] success={str(success).lower()} steps={step_n} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


if __name__ == "__main__":
    for tid in ["easy_off_by_one", "medium_logic_error", "hard_security_flaw"]:
        run_episode(tid)