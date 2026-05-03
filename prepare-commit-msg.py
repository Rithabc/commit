#!/usr/bin/env python
"""Git prepare-commit-msg hook that generates commit messages using Ollama."""
import sys
import subprocess
import json
import re
import urllib.request
import urllib.error

# Configurable options
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "commitgen"
MAX_DIFF_LINES = 200  # Truncate large diffs to avoid confusing the model


def get_git_diff():
    """Get staged diff and stat, truncated to MAX_DIFF_LINES."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, check=True,
            encoding='utf-8', errors='replace'
        )
        stat = result.stdout.strip()
    except Exception:
        stat = ""

    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, check=True,
            encoding='utf-8', errors='replace'
        )
        diff = result.stdout
    except subprocess.CalledProcessError:
        return "", ""
    except Exception:
        return "", ""

    lines = diff.split("\n")
    if len(lines) > MAX_DIFF_LINES:
        diff = "\n".join(lines[:MAX_DIFF_LINES]) + f"\n... ({len(lines) - MAX_DIFF_LINES} more lines truncated)"

    return diff, stat


def get_recent_commits():
    """Get last 5 non-merge commit subjects for style context."""
    try:
        result = subprocess.run(
            ["git", "log", "-n", "5", "--no-merges", "--pretty=format:%s"],
            capture_output=True, check=True,
            encoding='utf-8', errors='replace'
        )
        return result.stdout.strip()
    except Exception:
        return ""


def clean_response(raw):
    """Extract a clean conventional commit message from model output."""
    if not raw:
        return None

    raw = raw.strip()
    raw = raw.strip('`').strip('"').strip("'").strip()

    # Take only the first line
    first_line = raw.split('\n')[0].strip()

    # Try to find a conventional commit pattern anywhere in the output
    match = re.search(
        r'(feat|fix|refactor|docs|style|test|chore|perf|ci|build|revert)'
        r'(\([a-zA-Z0-9_/. -]+\))?!?:\s*.+',
        first_line,
        re.IGNORECASE
    )
    if match:
        msg = match.group(0).strip()
        if len(msg) > 72:
            msg = msg[:69] + "..."
        return msg

    # Fallback: use the first line if it looks clean
    if len(first_line) < 80 and not any(
        phrase in first_line.lower()
        for phrase in ["here's", "summary", "suggestion", "provided", "commit message",
                       "the following", "you've", "i would", "let me", "based on"]
    ):
        if len(first_line) > 72:
            first_line = first_line[:69] + "..."
        return first_line

    return None


def generate_commit_message(diff, stat, recent_commits):
    """Send diff to Ollama and get a commit message back."""

    # NOTE: This prompt must stay in sync with the format_example() function
    # in format_dataset.py so inference matches the training distribution.
    prompt = (
        f"Files changed:\n{stat}\n\n"
        f"Diff:\n{diff}\n\n"
        f"Recent commits for style reference:\n{recent_commits}\n"
        f"Commit message:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 60,
            "stop": ["\n\n", "Explanation:", "Note:", "This commit", "Here", "The ", "I "]
        }
    }

    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=120)
        result = json.loads(response.read().decode('utf-8'))
        raw = result.get('response', '').strip()
        return clean_response(raw)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        try:
            print(f"Ollama API Error: {json.loads(error_body).get('error', error_body)}")
        except json.JSONDecodeError:
            print(f"Ollama HTTP Error {e.code}: {error_body}")
        return None
    except urllib.error.URLError:
        print(f"Cannot connect to Ollama at {OLLAMA_API_URL}. Is it running?")
        return None
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: prepare-commit-msg <commit_message_file>")
        sys.exit(1)

    commit_msg_file = sys.argv[1]

    # Skip if user provided a message via -m, or it's a merge/squash
    if len(sys.argv) > 2 and sys.argv[2] in ['message', 'merge', 'squash', 'commit']:
        sys.exit(0)

    diff, stat = get_git_diff()
    if not diff or not diff.strip():
        sys.exit(0)

    print("Generating commit message with Ollama...")
    recent_commits = get_recent_commits()  # ← now actually used in the prompt
    commit_message = generate_commit_message(diff, stat, recent_commits)

    if commit_message:
        try:
            with open(commit_msg_file, 'r', encoding='utf-8', errors='replace') as f:
                existing_content = f.read()
        except Exception:
            existing_content = ""

        with open(commit_msg_file, 'w', encoding='utf-8') as f:
            f.write(commit_message + "\n\n" + existing_content)
        print(f"Generated: {commit_message}")
    else:
        print("Failed to generate commit message. Proceeding with default.")


if __name__ == "__main__":
    main()
