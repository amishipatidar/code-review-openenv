from env.models import TaskConfig, CodeSnippet

TASKS = {
    "easy_off_by_one": TaskConfig(
        task_id="easy_off_by_one",
        difficulty="easy",
        description=(
            "Review the following Python function that is supposed to return "
            "the sum of all elements in a list. Identify the bug, suggest a fix, "
            "and rate the overall code quality."
        ),
        snippet=CodeSnippet(
            language="python",
            filename="sum_list.py",
            code="""\
def sum_list(items):
    \"\"\"Return the sum of all elements in the list.\"\"\"
    total = 0
    for i in range(len(items) - 1):   # line 4
        total += items[i]
    return total

# Expected: sum_list([1, 2, 3, 4]) == 10
# Actual:   sum_list([1, 2, 3, 4]) == 6
""",
        ),
        bug_line=4,
        bug_description="off-by-one error: range should be range(len(items)) not range(len(items) - 1)",
        fix_keywords=["range(len(items))", "range(len(items))"],
        max_steps=5,
    ),

    "medium_logic_error": TaskConfig(
        task_id="medium_logic_error",
        difficulty="medium",
        description=(
            "Review the following Python function that checks whether a number "
            "is prime. Identify the logical bug, suggest a fix, and rate the code quality."
        ),
        snippet=CodeSnippet(
            language="python",
            filename="is_prime.py",
            code="""\
def is_prime(n):
    \"\"\"Return True if n is a prime number.\"\"\"
    if n < 2:
        return False
    for i in range(2, n):             # line 5  (inefficient but correct-ish)
        if n % i == 0:
            return False
    return True

def first_n_primes(count):
    primes = []
    candidate = 2
    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 2                # line 15 — BUG: skips even candidates > 2
    return primes

# Expected: first_n_primes(5) == [2, 3, 5, 7, 11]
# Actual:   first_n_primes(5) == [3, 5, 7, 11, 13]  (misses 2)
""",
        ),
        bug_line=15,
        bug_description="candidate increments by 2 starting from 2, so it jumps to 4 and misses 2 itself; should start candidate at 2 and increment by 1, or special-case 2",
        fix_keywords=["candidate += 1", "candidate + 1", "special case", "start at 2"],
        max_steps=6,
    ),

    "hard_security_flaw": TaskConfig(
        task_id="hard_security_flaw",
        difficulty="hard",
        description=(
            "Review the following Python web handler that fetches a user record "
            "from a database. Identify the security vulnerability, explain the risk, "
            "suggest a safe fix, and rate the code quality."
        ),
        snippet=CodeSnippet(
            language="python",
            filename="user_handler.py",
            code="""\
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)
DB_PATH = "users.db"

@app.route("/user")
def get_user():
    username = request.args.get("username", "")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # line 11 — direct string interpolation into SQL query
    query = f"SELECT id, username, email FROM users WHERE username = '{username}'"
    cursor.execute(query)                          # line 13
    row = cursor.fetchone()
    conn.close()
    if row:
        return jsonify({"id": row[0], "username": row[1], "email": row[2]})
    return jsonify({"error": "User not found"}), 404

# A request to /user?username=' OR '1'='1 returns ALL users.
""",
        ),
        bug_line=12,
        bug_description="SQL injection vulnerability: user input is interpolated directly into the SQL query string instead of using parameterized queries",
        fix_keywords=["parameterized", "?", "placeholder", "cursor.execute(query, (username,))", "prepared statement"],
        max_steps=7,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks():
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
        }
        for t in TASKS.values()
    ]
