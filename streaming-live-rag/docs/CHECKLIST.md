# Final Submission Checklist (25 Sep)

Copied and organized directly from the brief's own submission mechanics — treat every unchecked box as a blocker:

- [ ] Team registration completed (deadline was 16 Sep 11:59 PM — confirm, don't assume).
- [ ] Final-submission Google Form link confirmed received.
- [ ] Working prototype code in a public or shared GitHub repo.
- [ ] README with reproducible setup instructions (`docker compose up` or an equally clean CLI runner), Docker files, and any other requirements spelled out.
- [ ] Release tag on the final commit named **exactly** `PRISM_GENAI_HACKATHON_Y2026` — the tagged commit is what gets judged.
- [ ] Everything referenced anywhere in your submission (PPT, demo video, documentation) is physically present **inside that tagged commit**.
- [ ] Demo video, ≤ 5 minutes, hosted on YouTube or Drive, link tested in a logged-out browser.
- [ ] Presentation file (PPT or PDF) named **exactly** `CollegeName_TeamName_Submission_ppt`.
- [ ] PPT/PDF content includes: Theme ID, project title, team details; problem statement in your own words; solution and architecture diagram; tools and tech stack used; innovation highlights, results, and limitations.
- [ ] Exactly one submission per team, via the official Google Form.
- [ ] Team/PPT naming consistently uses `CollegeName_TeamName` everywhere.
- [ ] Submitted before 25 Sep 2026, 11:59 PM — with real buffer, not at the deadline.
- [ ] Query channel noted in case anything goes wrong at the last minute: `prism@samsung.com`.

## Non-Negotiables Cheat Sheet

Hard engineering rules (violating these fails the theme regardless of how polished the demo looks):

| Rule | What it means in practice |
| :--- | :--- |
| **Corpus isolation** | Every factual claim traces to the supplied corpus. No web calls, no third-party KBs, no answers from the model's parametric memory. |
| **No hardcoding / no precomputation** | The held-out replay is private. Do not embed example prompts, queries, or canned responses in code — including the three worked examples in the guide itself. Build general logic. |
| **Rigorous factual grounding**| Every assertion carries a `[Doc_ID §Section]` tag; explicit uncertainty when the corpus doesn't cover a sub-intent. |
| **Session-bound state** | Memory lives and dies with one conversation session. No persistent profiles, no cross-session leakage. |
| **Architectural parsimony** | Every component must justify its latency/compute cost — this is your license (and the jury's expectation) to stay lean. |
