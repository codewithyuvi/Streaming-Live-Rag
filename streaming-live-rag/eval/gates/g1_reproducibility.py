"""
Gate 1 — Reproducibility & Packaging (G1).

Honest checks only: verifies the packaging artifacts EXIST and are
well-formed. It does NOT claim to build the Docker image or boot a clean
machine — that requires docker and is out of scope for this script.
"""

import os
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval.gates._common import banner, footer, STATUS_PASS, STATUS_FAIL, gate_main


REQUIRED_DEPS = ["fastapi", "qdrant-client", "fastembed", "pyyaml", "httpx"]


def evaluate_g1():
    banner("GATE 1 — Reproducibility & Packaging (G1)")
    checks = []

    # 1. Dockerfile exists (presence only — no build claim).
    df = os.path.join(ROOT_DIR, "Dockerfile")
    checks.append(("Dockerfile present (build not attempted here)",
                   os.path.isfile(df)))

    # 2. docker-compose.yml exists (presence only).
    dc = os.path.join(ROOT_DIR, "docker-compose.yml")
    checks.append(("docker-compose.yml present", os.path.isfile(dc)))

    # 3. pyproject.toml is valid TOML and declares the required deps.
    pp = os.path.join(ROOT_DIR, "pyproject.toml")
    try:
        with open(pp, "rb") as f:
            toml = tomllib.load(f)
        proj_deps = toml.get("project", {}).get("dependencies", []) or []
        poetry_deps = toml.get("tool", {}).get("poetry", {}).get("dependencies", {}) or {}
        declared = " ".join(list(proj_deps) + list(poetry_deps)).lower()
        missing = [d for d in REQUIRED_DEPS if d not in declared]
        ok = not missing
        detail = f"missing: {missing}" if missing else "all required deps declared"
    except FileNotFoundError:
        ok, detail = False, "pyproject.toml not found"
    except Exception as e:  # invalid TOML
        ok, detail = False, f"invalid TOML: {e}"
    checks.append((f"pyproject.toml is valid TOML ({detail})", ok))

    # 4. requirements.txt exists.
    req = os.path.join(ROOT_DIR, "requirements.txt")
    checks.append(("requirements.txt present", os.path.isfile(req)))

    # 5. Modules import cleanly without model/network dependencies.
    try:
        import llm_config  # noqa: F401
        from session.store import Session  # noqa: F401
        from retrieval.grounding import validate  # noqa: F401
        from retrieval.merge import merge_and_dedup  # noqa: F401
        from controller.heuristics import is_stable_enough  # noqa: F401
        from streaming.live_stream import play_utterance  # noqa: F401
        from telemetry.schema import TelemetryEvent  # noqa: F401
        ok = True
    except Exception as e:
        ok = False
        print(f"   import failed: {e}")
    checks.append(("Core modules import without model/network deps", ok))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = 100.0 * passed / total
    print()
    for name, ok in checks:
        print(f"  {'OK ' if ok else 'MISS'} {name}")
    print(f"\nG1 score: {score:.1f}% ({passed}/{total}) — target 100%")

    ok_all = passed == total
    footer(ok_all, "GATE 1")
    return (STATUS_PASS if ok_all else STATUS_FAIL), f"{score:.1f}% ({passed}/{total})"


if __name__ == "__main__":
    sys.exit(gate_main("G1", evaluate_g1))
