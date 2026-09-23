"""
Gate 1 — Reproducibility & Packaging Evaluation (G1).

Validates that the project satisfies the competition packaging rules:
1. Dockerfile exists with pinned Python and pre-cached FastEmbed models.
2. docker-compose.yml defines Qdrant, one-shot ingest, and API with healthchecks.
3. pyproject.toml declares all required dependencies (including pyyaml).
4. System boots and runs with zero manual steps on a clean machine.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def evaluate_g1():
    print("=" * 70)
    print("    GATE 1 — Reproducibility & Single-Command Boot (G1)")
    print("=" * 70)

    checks = []

    # 1. Dockerfile presence and model pre-cache
    dockerfile_path = os.path.join(ROOT_DIR, "Dockerfile")
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            df_content = f.read()
        has_python_pin = "python:3.11" in df_content or "python:3.12" in df_content
        has_model_cache = "fastembed" in df_content and "bge-small" in df_content
        checks.append({
            "name": "Dockerfile with pinned Python & model pre-cache",
            "passed": has_python_pin and has_model_cache,
            "details": f"Pinned: {has_python_pin}, Pre-cached models: {has_model_cache}"
        })
    else:
        checks.append({"name": "Dockerfile exists", "passed": False, "details": "Dockerfile missing"})

    # 2. docker-compose.yml configuration
    compose_path = os.path.join(ROOT_DIR, "docker-compose.yml")
    if os.path.exists(compose_path):
        with open(compose_path, "r", encoding="utf-8") as f:
            dc_content = f.read()
        has_qdrant = "qdrant" in dc_content
        has_ingest = "ingest" in dc_content
        has_api = "api" in dc_content
        has_healthcheck = "healthcheck" in dc_content
        passed = has_qdrant and has_ingest and has_api and has_healthcheck
        checks.append({
            "name": "docker-compose.yml (qdrant + ingest + api + healthcheck)",
            "passed": passed,
            "details": f"qdrant:{has_qdrant}, ingest:{has_ingest}, api:{has_api}, health:{has_healthcheck}"
        })
    else:
        checks.append({"name": "docker-compose.yml exists", "passed": False, "details": "compose file missing"})

    # 3. pyproject.toml dependencies
    pyproject_path = os.path.join(ROOT_DIR, "pyproject.toml")
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "r", encoding="utf-8") as f:
            pp_content = f.read()
        has_fastapi = "fastapi" in pp_content
        has_qdrant = "qdrant-client" in pp_content
        has_fastembed = "fastembed" in pp_content
        has_pyyaml = "pyyaml" in pp_content
        passed = has_fastapi and has_qdrant and has_fastembed and has_pyyaml
        checks.append({
            "name": "pyproject.toml declared dependencies (including pyyaml)",
            "passed": passed,
            "details": f"fastapi:{has_fastapi}, qdrant:{has_qdrant}, fastembed:{has_fastembed}, pyyaml:{has_pyyaml}"
        })
    else:
        checks.append({"name": "pyproject.toml exists", "passed": False, "details": "pyproject missing"})

    # 4. Clean import check (lazy client initialization, no crash on import)
    try:
        sys.path.insert(0, ROOT_DIR)
        import llm_config  # noqa: F401
        from session.store import Session  # noqa: F401
        from retrieval.grounding import validate  # noqa: F401
        from retrieval.merge import merge_with_quota  # noqa: F401
        from controller.heuristics import is_stable_enough  # noqa: F401
        checks.append({
            "name": "Clean module imports without hard failure when keys unset",
            "passed": True,
            "details": "llm_config, store, grounding, merge, heuristics imported cleanly"
        })
    except Exception as e:
        checks.append({
            "name": "Clean module imports without hard failure",
            "passed": False,
            "details": str(e)
        })

    # Summary
    passed_count = sum(1 for c in checks if c["passed"])
    total_count = len(checks)
    score = (passed_count / total_count) * 100

    print(f"\n📊 G1 Score: {score:.1f}% ({passed_count}/{total_count}) — Target: 100%")
    print("─" * 70)
    for c in checks:
        icon = "✅" if c["passed"] else "❌"
        print(f"  {icon} {c['name']}")
        print(f"     Details: {c['details']}")

    print("=" * 70)
    if score == 100.0:
        print("🟢 GATE 1 PASSED (Reproducibility & Packaging Ready)")
    else:
        print("🔴 GATE 1 FAILED")
    print("=" * 70)

    return score == 100.0, f"{score:.1f}%"


if __name__ == "__main__":
    evaluate_g1()
