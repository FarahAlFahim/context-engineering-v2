"""Node/path filtering utilities for code graphs."""

import re
from typing import List


def path_contains_excluded_dir(path: str, exclude_dirs: list) -> bool:
    """Check if a file path contains any excluded directory component."""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in exclude_dirs:
            return True
    return False


def is_test_node(node_id: str, path: str = "") -> bool:
    """Check if a node is a test node based on its ID or path."""
    check = (node_id + " " + path).lower()
    # Common test patterns
    if any(p in check for p in ["test_", "_test.", "tests/", "test/", "/test_", "conftest"]):
        return True
    # Check for test class/function prefixes
    parts = node_id.split("::")
    for part in parts:
        name = part.split(".")[-1] if "." in part else part
        if name.lower().startswith("test") or name.lower().endswith("test"):
            return True
    return False


def test_name_to_prod_candidates(test_name: str) -> List[str]:
    """Derive production candidate names from a test name.

    E.g., 'test_my_function' -> ['my_function']
         'TestMyClass' -> ['MyClass']
         'test_MyClass_method' -> ['MyClass', 'method']
    """
    candidates = []

    # Strip common test prefixes
    name = test_name
    for prefix in ("test_", "Test", "test"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    if name:
        candidates.append(name)

    # Split on underscores for compound names
    parts = name.split("_")
    if len(parts) > 1:
        candidates.extend(parts)

    # CamelCase split
    camel_parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)", name)
    if len(camel_parts) > 1:
        for cp in camel_parts:
            if len(cp) > 2:
                candidates.append(cp)

    return list(set(c for c in candidates if len(c) > 1))
