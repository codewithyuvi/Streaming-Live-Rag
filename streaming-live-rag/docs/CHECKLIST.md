# Final Submission Checklist (25 Sep 2026)

This checklist tracks engineering deliverables and official submission requirements for Samsung PRISM GenAI Hackathon (Theme 4 - Streaming Live RAG).

---

## 1. Engineering & Codebase Deliverables (All Completed)

- [x] **Working Prototype Code:** Full streaming live RAG pipeline implemented (`api/main.py`, two-stage controller, multi-intent decomposition, hybrid Qdrant search, session refinement, claim-level grounding).
- [x] **Reproducible Packaging (Gate 1):** Single-command setup via `docker compose up --build -d` and local virtual environment support (`requirements.txt`).
- [x] **Pre-cached Embeddings & Deterministic Offline Build:** Dockerfile caches FastEmbed models (`bge-small`, `bm25`, `ms-marco-MiniLM`) at build time to prevent network timeouts during evaluation.
- [x] **Evaluation Gate Suite (Gates G1–G6):** Master evaluation runner (`eval/run_eval.py`, `run_eval.bat`, `run_eval.sh`) with 100% pass across all 6 gates documented in `eval/results/scorecard.json`.
- [x] **Interactive Demonstration Dashboard:** Web UI (`static/index.html`, `scripts/start_ui.bat`) featuring simulated chunk-by-chunk speech streaming, 6 one-click evaluation presets, controller visualizer, and live latency breakdown.
- [x] **Full Documentation Suite:**
  - `README.md` — 3-command quick start, architecture overview, and scorecard summary.
  - `docs/RUNBOOK.md` — Complete operational runbook covering all phases (0 through 6).
  - `docs/BENCHMARK_REPORT.md` — Verified gate scores, latency profiles, Ablations #1 & #2, and documented edge cases.
  - `docs/ARCHITECTURE_BRIEF.md` — End-to-end component flow, telemetry schema, and boundary contracts.
  - `docs/adr/` — Complete Architectural Decision Records (ADR-1 through ADR-5).
  - `agents.md` — Synchronized chronological collaboration log.

---

## 2. Team Administrative & Submission Tasks

- [ ] **Confirm Team Registration:** Ensure college/team registration details match PRISM records.
- [ ] **Record Demo Video (≤ 5 minutes):**
  - Walk through the Interactive Demo Dashboard (`scripts\start_ui.bat`).
  - Demonstrate Scenario 1 (Early Retrieval at $t_1$).
  - Demonstrate Scenario 2 (Multi-Intent parallel retrieval with Quota Merge).
  - Demonstrate Scenario 3 (Chit-Chat suppression with zero DB queries).
  - Demonstrate Scenario 4 (Multi-Turn Late Detail Refinement $1 \to 1 \to 2$).
  - Demonstrate Scenario 5 (Presentation-Only reformat without DB queries).
  - Host on YouTube (Unlisted) or Google Drive (link shared to "Anyone with link can view"). Test in a private/incognito window.
- [ ] **Finalize Presentation Deck:**
  - File name **must be exactly**: `CollegeName_TeamName_Submission_ppt` (PPT or PDF).
  - Content: Theme ID, project title, problem statement, solution & architecture diagram, tech stack, innovation highlights, results scorecard, limitations.
- [ ] **Tag Final Release Commit:**
  - Tag name **must be exactly**: `PRISM_GENAI_HACKATHON_Y2026`.
  - Command:
    ```bash
    git tag PRISM_GENAI_HACKATHON_Y2026
    git push origin PRISM_GENAI_HACKATHON_Y2026
    ```
- [ ] **Submit Official Google Form:**
  - Exactly one submission per team before **25 Sep 2026, 11:59 PM IST**.
  - Query support email if required: `prism@samsung.com`.

---

## 3. Non-Negotiables Verification Matrix

| Rule | Implementation Guarantee | Status |
| :--- | :--- | :--- |
| **Corpus isolation** | All factual claims derived strictly from Qdrant local dev corpus; no external web searches or ungrounded generation. | Verified |
| **No hardcoding / no precomputation** | Decomposer, controller, and synthesizer use dynamic LLM prompts and deterministic heuristics; zero canned answers. | Verified |
| **Rigorous factual grounding** | Deterministic 44-line validator checks `[Doc_ID §Section]` tags against retrieved context; 0 fabricated IDs. | Verified |
| **Session-bound state** | Ephemeral in-memory store keyed strictly by `session_id`; zero cross-session data leakage or persistent user profiles. | Verified |
| **Architectural parsimony** | Dual-provider setup (Groq for sub-400ms controller, Gemini for synthesis, FastEmbed ONNX local embeddings). | Verified |
