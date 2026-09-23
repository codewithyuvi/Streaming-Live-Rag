"""
Phase 5 — Claim-Level Deterministic Grounding Validator (ADR-5 / C5 / C6).
Tested against edge cases T1-T10 in Appendix B.1 of the audit report.
"""

import re
from dataclasses import dataclass, field

BRACKET = re.compile(r"[\[(]([^\[\]()]*?Doc_\d+[^\[\]()]*?)[\])]")  # [..] or (..) containing a Doc_ mention
ONE_TAG = re.compile(r"Doc_(\d+)\s*(?:§\s*([\w.\-]+))?", re.I)         # Doc_01 §1 | Doc_01 §A-1 | Doc_01§1 | Doc_01
ABSTAIN = re.compile(
    r"\b(not (?:available|found|mentioned|specified|covered|included)"
    r"|no (?:relevant )?information (?:is |was )?(?:available|found|provided)"
    r"|cannot (?:be )?(?:determined|verified|answer(?: that)? reliably)"
    r"|could not (?:be )?(?:determined|verified|answer)"
    r"|unable to (?:verify|determine|answer)"
    r"|insufficient (?:information|evidence))\b",
    re.I
)


def canon(doc, sec):
    """Canonical spelling for both sides of the comparison."""
    return f"Doc_{int(doc):02d} §{sec}" if sec else f"Doc_{int(doc):02d}"


def tags_in(text: str) -> list[str]:
    """Extracts all citation tags from text, handling brackets, commas, semicolons, parentheses, and unbracketed mentions."""
    out = []
    for inner in BRACKET.findall(text):
        for part in re.split(r"[;,]", inner):
            if (m := ONE_TAG.search(part)):
                out.append(canon(m.group(1), m.group(2)))
    # Also extract any unbracketed standalone Doc_XX mentions
    for m in ONE_TAG.finditer(text):
        c = canon(m.group(1), m.group(2))
        if c not in out:
            out.append(c)
    return out


@dataclass
class Report:
    cited: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)  # section-less / malformed IDs are fabricated too
    factual: int = 0
    supported: int = 0
    abstained: bool = False

    @property
    def support(self) -> float:
        """Claim-level support, which is what G4 measures."""
        return 1.0 if self.factual == 0 else self.supported / self.factual

    @property
    def ok(self) -> bool:
        return not self.fabricated and self.support >= 0.85

    # Backwards-compatible properties
    @property
    def is_grounded(self) -> bool:
        return self.ok

    @property
    def score(self) -> float:
        return self.support

    @property
    def valid_tags(self) -> list[str]:
        return [t for t in self.cited if t not in self.fabricated]

    @property
    def fabricated_tags(self) -> list[str]:
        return self.fabricated

    @property
    def uncertainty_expressed(self) -> bool:
        return self.abstained

    @property
    def uncited_claims_detected(self) -> bool:
        return self.factual > self.supported

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "is_grounded": self.is_grounded,
            "cited_tags": self.cited,
            "valid_tags": self.valid_tags,
            "fabricated_tags": self.fabricated,
            "uncited_claims_detected": self.uncited_claims_detected,
            "uncertainty_expressed": self.uncertainty_expressed,
            "score": round(self.score, 3),
            "factual": self.factual,
            "supported": self.supported,
            "abstained": self.abstained,
        }


def validate(answer: str, retrieved_tags: list[str]) -> Report:
    """
    Validates claim-level grounding of an answer against retrieved tags.
    """
    if not answer:
        return Report()

    avail = {t for x in retrieved_tags for t in tags_in(f"[{x}]")}
    r = Report(cited=list(dict.fromkeys(tags_in(answer))))
    r.fabricated = [t for t in r.cited if t not in avail]

    # Canonical abstention messages produced by the pipeline are grounded
    # by construction — they assert no facts and cite nothing.
    stripped = answer.strip()
    if stripped in (
        "This information is not available in the provided documents.",
        "I cannot answer that reliably based on the provided documents.",
    ) and not r.cited and not r.fabricated:
        r.abstained = True
        return r

    # Normalize: if punctuation precedes bracket citation (e.g. "claim. [Doc_01 §1]"),
    # move punctuation to end of citation tag so it attaches to its claim sentence
    normalized = re.sub(r"([.!?])\s*([\[(][^\[\]()]*Doc_\d+[^\[\]()]*[\])])", r" \2\1", answer.strip())

    for s in re.split(r"(?<=[.!?])\s+(?![\[(]\s*Doc_)", normalized):
        if len(s.split()) < 4:
            continue  # fragments / greetings are not claims
        if ABSTAIN.search(s):
            r.abstained = True
            continue  # an abstention sentence is not a claim
        r.factual += 1
        r.supported += bool(set(tags_in(s)) & avail)  # sentence carries >=1 retrievable tag

    return r


def validate_grounding(answer_text: str, available_tags: list[str]) -> Report:
    """Wrapper alias maintaining backward-compatibility with earlier callers."""
    return validate(answer_text, available_tags)
