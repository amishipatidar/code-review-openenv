import os
import json
import requests

# ✅ FIX: Use Hugging Face Space instead of localhost
ENV_URL = os.getenv(
    "ENV_URL",
    "https://amishipatidar-code-review-env-v2.hf.space"
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")


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

        print(f"[START] task={task_id} env=code-review model={MODEL_NAME}", flush=True)

        # STEP 1
        res = requests.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "action_type": "identify_bug",
            "line_number": 1,
            "description": "possible bug"
        }).json()

        step_n += 1
        rewards.append(res.get("reward", 0.0))

        print(f"[STEP] step={step_n} action=identify_bug(...) reward={rewards[-1]:.2f} done={str(res.get('done')).lower()} error=null", flush=True)

        if res.get("done"):
            success = sum(rewards) > 0.5
            return

        # STEP 2
        res2 = requests.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "action_type": "suggest_fix",
            "fixed_code": "fixed code"
        }).json()

        step_n += 1
        rewards.append(res2.get("reward", 0.0))

        print(f"[STEP] step={step_n} action=suggest_fix(...) reward={rewards[-1]:.2f} done={str(res2.get('done')).lower()} error=null", flush=True)

        if res2.get("done"):
            success = sum(rewards) > 0.5
            return

        # STEP 3
        res3 = requests.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "action_type": "rate_quality",
            "quality_score": 5
        }).json()

        step_n += 1
        rewards.append(res3.get("reward", 0.0))

        print(f"[STEP] step={step_n} action=rate_quality(...) reward={rewards[-1]:.2f} done={str(res3.get('done')).lower()} error=null", flush=True)

        # STEP 4
        res4 = requests.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "action_type": "submit"
        }).json()

        step_n += 1
        rewards.append(res4.get("reward", 0.0))

        print(f"[STEP] step={step_n} action=submit() reward={rewards[-1]:.2f} done={str(res4.get('done')).lower()} error=null", flush=True)

        success = sum(rewards) > 0.5

    except Exception as e:
        print(f"[STEP] step={step_n} action=null reward=0.00 done=true error={safe_error(e)}", flush=True)

    finally:
        print(f"[END] success={str(success).lower()} steps={step_n} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


if __name__ == "__main__":
    for tid in ["easy_off_by_one", "medium_logic_error", "hard_security_flaw"]:
        run_episode(tid)