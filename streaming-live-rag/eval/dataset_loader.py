"""
eval/dataset_loader.py — Robust YAML reader for labeled_set.yaml.
Supports pyyaml if present, and provides a lightweight native parser fallback.
"""

import os

def load_labeled_set(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "labeled_set.yaml")
        
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass

    queries = []
    cur = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- id:"):
                if cur:
                    queries.append(cur)
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                cur = {"id": val}
            elif ":" in stripped and cur is not None:
                parts = stripped.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip().strip('"').strip("'")
                if v == "null":
                    v = None
                elif v.isdigit():
                    v = int(v)
                cur[k] = v
        if cur:
            queries.append(cur)
            
    return {"queries": queries}
