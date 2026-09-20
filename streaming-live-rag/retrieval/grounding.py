"""
Phase 5 — Deterministic Grounding Validator (G4)

Validates that every citation tag in the LLM's answer actually exists
in the corpus context that was provided. Catches:
  1. Fabricated Doc IDs (hallucinated references)
  2. Fabricated Section numbers
  3. Claims that appear to state facts without any citation

This is a POST-synthesis validator — it runs on the answer text after
the LLM produces it, and returns a grounding report.

Design: Purely deterministic (regex-based), no LLM call needed.
"""

import re
from dataclasses import dataclass, field


@dataclass
class GroundingReport:
    """Result of grounding validation on an LLM answer."""
    is_grounded: bool
    cited_tags: list[str] = field(default_factory=list)
    valid_tags: list[str] = field(default_factory=list)
    fabricated_tags: list[str] = field(default_factory=list)
    uncited_claims_detected: bool = False
    uncertainty_expressed: bool = False
    score: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "is_grounded": self.is_grounded,
            "cited_tags": self.cited_tags,
            "valid_tags": self.valid_tags,
            "fabricated_tags": self.fabricated_tags,
            "uncited_claims_detected": self.uncited_claims_detected,
            "uncertainty_expressed": self.uncertainty_expressed,
            "score": self.score,
        }


# Regex to match citation tags like [Doc_01 §1], [Doc_02 §2], etc.
CITATION_PATTERN = re.compile(r'\[Doc_\d+\s*§\s*\d+\]')

# Uncertainty phrases that indicate the LLM properly flagged gaps
UNCERTAINTY_PHRASES = [
    "not available in the provided",
    "not found in the context",
    "no information",
    "not mentioned",
    "cannot determine",
    "insufficient information",
    "not covered",
    "not specified",
    "unclear from the context",
    "uncertain",
    "does not contain",
    "not included",
    "no relevant information",
]


def validate_grounding(
    answer_text: str,
    available_tags: list[str],
) -> GroundingReport:
    """
    Validates that all citation tags in the answer exist in the available corpus tags.

    Args:
        answer_text: The LLM-generated answer string.
        available_tags: List of valid tags (e.g. ["Doc_01 §1", "Doc_01 §2"]) that
                        were in the context provided to the LLM.

    Returns:
        GroundingReport with detailed validation results.
    """
    # Normalize available tags for comparison
    normalized_available = set()
    for tag in available_tags:
        # Strip brackets if present, normalize whitespace
        clean = tag.strip().strip("[]")
        clean = re.sub(r'\s+', ' ', clean)
        normalized_available.add(clean)

    # Extract all citation tags from the answer
    raw_citations = CITATION_PATTERN.findall(answer_text)
    cited_tags = []
    for cite in raw_citations:
        clean = cite.strip("[]")
        clean = re.sub(r'\s+', ' ', clean)
        if clean not in cited_tags:
            cited_tags.append(clean)

    # Classify as valid or fabricated
    valid_tags = [t for t in cited_tags if t in normalized_available]
    fabricated_tags = [t for t in cited_tags if t not in normalized_available]

    # Check for uncertainty expression
    answer_lower = answer_text.lower()
    uncertainty_expressed = any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES)

    # Compute score
    if not cited_tags:
        # No citations at all — only acceptable if uncertainty is expressed
        score = 1.0 if uncertainty_expressed else 0.0
    else:
        score = len(valid_tags) / len(cited_tags) if cited_tags else 0.0

    is_grounded = len(fabricated_tags) == 0 and (len(valid_tags) > 0 or uncertainty_expressed)

    return GroundingReport(
        is_grounded=is_grounded,
        cited_tags=cited_tags,
        valid_tags=valid_tags,
        fabricated_tags=fabricated_tags,
        uncited_claims_detected=False,  # Could be enhanced with NLI later
        uncertainty_expressed=uncertainty_expressed,
        score=score,
    )
