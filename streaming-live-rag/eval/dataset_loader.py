"""
eval/dataset_loader.py — YAML reader for labeled_set.yaml.

pyyaml is a declared dependency (pyproject.toml), so it is required. The
native fallback is kept for bare environments and now strips trailing '#'
comments correctly (the old fallback kept "null # wait" as a truthy string).
"""

import os

try:
    import yaml
    _HAVE_YAML = True
except ImportError:  # pragma: no cover - pyyaml is a declared dependency
    _HAVE_YAML = False


def _strip_comment(line: str) -> str:
    """Remove a trailing '#' comment, respecting single/double quotes."""
    out = []
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in (" ", "\t")):
            break  # unquoted, whitespace-preceded '#' starts a comment
        else:
            out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _native_parse(path: str) -> dict:
    """Minimal YAML-subset parser for the dataset (scalars + lists + nested maps)."""
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [_strip_comment(ln.rstrip("\n")) for ln in f]
    lines = [ln for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]

    def _scalar(v: str):
        v = v.strip()
        if (len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"')):
            return v[1:-1]
        if v == "null" or v == "~":
            return None
        if v == "true":
            return True
        if v == "false":
            return False
        if v.startswith("[") and v.endswith("]"):
            # flow-style list: ["a", "b"] — split on commas outside quotes
            items, cur, quote = [], [], None
            for ch in v[1:-1]:
                if quote:
                    cur.append(ch)
                    if ch == quote:
                        quote = None
                elif ch in ("'", '"'):
                    quote = ch
                    cur.append(ch)
                elif ch == ",":
                    items.append(_scalar("".join(cur)))
                    cur = []
                else:
                    cur.append(ch)
            if "".join(cur).strip():
                items.append(_scalar("".join(cur)))
            return items
        if v.isdigit():
            return int(v)
        try:
            return float(v)
        except ValueError:
            return v

    root: dict = {}
    # stack of (indent, container); containers are dicts or lists
    stack: list = [(-1, root)]

    def _peek_list_item(indent):
        # placeholder
        return None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            item_body = stripped[2:].strip()
            if not isinstance(parent, list):
                raise ValueError(f"native parser: list item outside a list at line {i}: {line}")
            if ":" in item_body and not item_body.startswith(('"', "'")):
                # "- key: value" inline map start
                d: dict = {}
                k, v = item_body.split(":", 1)
                if v.strip():
                    d[k.strip()] = _scalar(v)
                parent.append(d)
                stack.append((indent, d))
            else:
                parent.append(_scalar(item_body))
        elif ":" in stripped:
            k, v = stripped.split(":", 1)
            k, v = k.strip(), v.strip()
            if v:
                if isinstance(parent, dict):
                    parent[k] = _scalar(v)
                else:
                    raise ValueError(f"native parser: map entry inside list at line {i}: {line}")
            else:
                # nested block: look ahead to decide list vs map
                j = i + 1
                child_is_list = False
                while j < n:
                    nxt = lines[j]
                    if not nxt.strip():
                        j += 1
                        continue
                    child_is_list = nxt.strip().startswith("- ")
                    break
                child = [] if child_is_list else {}
                if isinstance(parent, dict):
                    parent[k] = child
                else:
                    raise ValueError(f"native parser: nested map inside list at line {i}: {line}")
                stack.append((indent, child))
        i += 1
    return root


def load_labeled_set(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "labeled_set.yaml")
    if _HAVE_YAML:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    return _native_parse(path)
