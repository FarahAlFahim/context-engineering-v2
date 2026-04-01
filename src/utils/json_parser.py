"""Robust JSON parsing from LLM output text."""

import json
from typing import Optional


def parse_json_best_effort(text, preferred_keys: Optional[list] = None):
    """Parse a JSON object from *text* robustly.

    Uses a balanced-brace scanner to extract every top-level {...} candidate
    and tries json.loads on each. When *preferred_keys* is given, any candidate
    containing one of those keys is returned immediately. Otherwise the largest
    valid JSON candidate is returned.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown fences
    stripped = text
    for fence in ("```json\n", "```JSON\n", "```\n"):
        if fence in stripped:
            stripped = stripped.replace(fence, "").replace("\n```", "")
    if stripped != text:
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # Balanced-brace scanner
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate_str = text[i:j + 1]
                            try:
                                obj = json.loads(candidate_str)
                                if isinstance(obj, dict):
                                    if preferred_keys:
                                        for pk in preferred_keys:
                                            if pk in obj:
                                                return obj
                                    candidates.append(obj)
                            except Exception:
                                pass
                            break
                j += 1
            i = j + 1 if j < len(text) else i + 1
        else:
            i += 1

    if not candidates:
        return None
    return max(candidates, key=lambda c: len(c))
