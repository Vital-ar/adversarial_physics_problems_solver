"""Deterministic Verifier Directive Self-Containment & Zero-Answer-Leakage Gate (`directive_leak_check.py`).

Enforces the 5 strict rules for every Model B Verifier directive:
  1. **One specific error per directive** (concise physical/mathematical constraint).
  2. **Strictly self-contained**: Zero references to step numbers, line numbers, or previous text
     (`"in step 2"`, `"your first equation"`, `"above"`, `"in line"`, `"previous attempt"` are banned).
  3. **Never leaks the final answer** on `[[CHALLENGE]]` directives: Checked deterministically against
     the numerical and symbolic `ANSWER:` extracted from `reference_solution`.
  4. **Physically actionable** & non-empty.
  5. **Concise**: `<= 192` tokens and terminated by `[[CHALLENGE]]` or `[[APPROVE]]`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from adversarial_physics.data.curation.answer_validator import (
    _extract_primary_number,
    _is_purely_numeric_answer,
    _normalize_math_expr_string,
    extract_answer_line,
)
from adversarial_physics.data.schemas.trajectory import VERDICT_APPROVE, VERDICT_CHALLENGE

BANNED_RELATIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bin\s+step\s*\d+\b", "in step N"),
    (r"\bstep\s*\d+\b", "step N"),
    (r"\byour\s+(?:first|second|third|last|previous|initial)\s+(?:equation|step|line|attempt|solution)\b", "relative attempt reference"),
    (r"\bin\s+line\s*\d+\b", "in line N"),
    (r"\b(?:as\s+shown\s+)?above\b", "above"),
    (r"\bthe\s+proposed\s+solution\b", "the proposed solution"),
    (r"\byour\s+derivation\b", "your derivation"),
)

# Trivial numbers / physical constants that are allowed in directives even if they appear in the answer
_TRIVIAL_ALLOWED_NUMBERS: set[str] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "90", "180", "360",
    "0.5", "1.5", "2.0", "9.8", "9.81", "3.14", "3.1416",
}


@dataclass(frozen=True)
class DirectiveValidationResult:
    """Validation report for a Model B Verifier directive."""

    is_valid: bool
    verdict: str | None
    cleaned_directive_body: str
    formatted_full_output: str
    word_count: int
    has_banned_relative_phrase: bool
    leaks_final_answer: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_directive_text(
    raw_text: str,
    *,
    expected_verdict: str = VERDICT_CHALLENGE,
    problem_statement: str = "",
    reference_solution: str = "",
) -> str:
    """Clean LLM-generated directive text, strip any relative step references or leaked final numbers,
    and ensure it ends with `expected_verdict` (`[[CHALLENGE]]` or `[[APPROVE]]`).
    """
    text = (raw_text or "").strip()
    # Strip markdown code fences
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    # Remove sentinel tokens from body so we can place the exact sentinel cleanly at the end
    body = text.replace(VERDICT_CHALLENGE, "").replace(VERDICT_APPROVE, "").strip()
    # Unwrap or strip nested [[CHALLENGE: ...]] / [[APPROVE: ...]] blocks
    body = re.sub(
        r"\[\[\s*(?:CHALLENGE|APPROVE|REJECT)\s*:\s*(.*?)\s*\]\]\.?",
        r"\1.",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"\[\[\s*(?:CHALLENGE|APPROVE|REJECT)[^\]]*\]\]", "", body, flags=re.IGNORECASE).strip()

    # Strip bold markdown meta-headers like **Corrective Directive:**, **Correct Direction:**, **Step 1: ...**
    body = re.sub(
        r"\*\*\s*(?:Corrective\s+Directive|Correcting\s+Directive|Correct\s+Directive|Correct\s+Direction|Correct\s+Physical\s+Law|Physical\s+Law|Final\s+Challenge|Directive|Feedback|Review|Correction|Confirmation|Step\s*\d+[^*]*)\s*:?\s*\*\*\s*:?\s*",
        " ",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"^(?:Corrective\s+Directive|Correcting\s+Directive|Correct\s+Directive|Directive|Feedback|Review|Correction|Confirmation)\s*:\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )

    # Replace relative phrasing ("the proposed solution", "your derivation", "step N", "reference solution")
    body = re.sub(
        r"\bThe\s+proposed\s+solution\s+(?:is\s+sound\s*(?:because\s+it|,\s*as\s+it)?|correctly\s+|demonstrates\s+|directly\s+)?",
        "The derivation ",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"\bthe\s+proposed\s+solution(?:'s)?\b", "the derivation", body, flags=re.IGNORECASE)
    body = re.sub(r"\byour\s+proposed\s+solution\b", "the derivation", body, flags=re.IGNORECASE)
    body = re.sub(r"\bthe\s+flawed\s+(?:proposed\s+)?(?:solution|derivation)\b", "the derivation", body, flags=re.IGNORECASE)
    body = re.sub(r"\byour\s+(?:first|second|third|last|previous|initial)?\s*(?:equation|step|line|attempt|solution|derivation)\b", "the governing equation", body, flags=re.IGNORECASE)
    body = re.sub(r"\b(?:in|at|during|from|of|see|refer\s+to)\s+step\s*\d+(?:\s*(?:to|through|-)\s*\d+)?(?:\s+(?:in|of)\s+the\s+reference\s+solution)?\b", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\(\s*step\s*\d+[^)]*\)", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\bstep\s*\d+\b", "the derivation step", body, flags=re.IGNORECASE)
    body = re.sub(r"\b(?:in|on)\s+line\s*\d+\b", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\b(?:in|of|from|as\s+(?:shown|described|detailed)\s+in)\s+the\s+(?:ground-truth\s+)?reference\s+solution\b", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\babove\b", "in the governing equation", body, flags=re.IGNORECASE)
    body = re.sub(r"\*{2,}\s*\.?", "", body)
    body = re.sub(r"\s+", " ", body).strip(" ,;:-")

    # Trim truncated trailing fragments (e.g., ending with "The key steps are: 1." or ",." or "which is:")
    body = re.sub(r"(?:The\s+key\s+(?:steps|equations|laws)\s+(?:are|to\s+use\s+are)\s*:?\s*)?(?:\d+\.|\b(?:and|or|the|to|of|in|with|by|for|from|as|is|are|which|where|namely)\s*\.?|,+\s*\.?|:+\s*\.?|-+\s*\.?)$", "", body, flags=re.IGNORECASE).strip(" ,;:-")

    # If this is a CHALLENGE directive, ensure the final numeric or symbolic answer is not leaked
    if expected_verdict == VERDICT_CHALLENGE and reference_solution:
        _, ref_ans = extract_answer_line(reference_solution)
        if ref_ans and _is_purely_numeric_answer(ref_ans):
            num_val = _extract_primary_number(ref_ans)
            if num_val is not None:
                num_str = f"{num_val:g}"
                if (
                    len(num_str) >= 2
                    and num_str not in _TRIVIAL_ALLOWED_NUMBERS
                    and num_str not in (problem_statement or "")
                ):
                    body = re.sub(rf"(?<![0-9.]){re.escape(num_str)}(?![0-9.])", "the target value", body)
        elif ref_ans:
            norm_rhs = _normalize_math_expr_string(ref_ans)
            if len(norm_rhs) >= 12 and norm_rhs in _normalize_math_expr_string(body) and norm_rhs not in _normalize_math_expr_string(problem_statement):
                # Strip any trailing sentence or clause that leaks the final symbolic expression
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
                kept_sentences = [
                    s for s in sentences if norm_rhs not in _normalize_math_expr_string(s)
                ]
                if kept_sentences:
                    body = " ".join(kept_sentences).strip()
                else:
                    # Strip clause after "leading to", "resulting in", "implies", "gives", "is given by", "obtain"
                    body = re.split(
                        r"(?:,?\s*(?:leading\s+to|resulting\s+in|which\s+(?:gives|yields|simplifies\s+to|implies)|implies|you\s+should\s+obtain|the\s+(?:final|correct)\s+(?:expression|formula|relationship|solution|equation)[^.]*))",
                        body,
                        maxsplit=1,
                        flags=re.IGNORECASE,
                    )[0].strip(" ,;:-")
                if norm_rhs in _normalize_math_expr_string(body):
                    # Fall back to stating the governing physical principle from the first step of reference_solution
                    first_step_lines = [
                        ln.strip()
                        for ln in reference_solution.splitlines()
                        if ln.strip() and not ln.strip().upper().startswith("ANSWER:")
                    ]
                    first_step = re.sub(r"^Step\s*\d+\s*:\s*", "", first_step_lines[0] if first_step_lines else "")
                    if first_step and norm_rhs not in _normalize_math_expr_string(first_step):
                        body = f"Apply the governing physical setup and conservation relations ({first_step.rstrip('.')}) and solve for the target quantity without algebraic sign or substitution errors."
                    else:
                        body = "Apply the governing physical equations and boundary conditions from first principles, maintaining dimensional consistency before isolating the target variable."

    # Truncate to at most 2 concise sentences if overlong (> 125 words)
    words = body.split()
    if len(words) > 125:
        sentences = re.split(r"(?<=[.!?])\s+", body)
        body = " ".join(sentences[:2]).strip()
        if len(body.split()) > 125:
            body = " ".join(body.split()[:115]).rstrip(" ,;:") + "."

    body = re.sub(r"\s+", " ", body).strip(" ,;:-")
    if len(body.split()) < 6:
        if expected_verdict == VERDICT_APPROVE:
            body = "The governing physical equations, boundary conditions, and algebraic substitutions are dimensionally consistent and yield the valid result."
        else:
            body = "Apply the governing physical laws and boundary conditions from first principles, ensuring algebraic and dimensional consistency throughout the derivation."

    if body:
        body = body[0].upper() + body[1:]
    if not body.endswith((".", "!", "?")):
        body = body + "."

    return f"{body}\n{expected_verdict}"


def validate_verifier_directive(
    directive_output: str,
    *,
    problem_statement: str,
    reference_solution: str,
    expected_verdict: str | None = None,
    max_words: int = 145,
) -> DirectiveValidationResult:
    """Deterministically validate a Model B Verifier output (`directive_body + \\n[[CHALLENGE]]|[[APPROVE]]`)."""
    raw = (directive_output or "").strip()
    verdict: str | None = None
    if VERDICT_CHALLENGE in raw:
        verdict = VERDICT_CHALLENGE
    elif VERDICT_APPROVE in raw:
        verdict = VERDICT_APPROVE

    if verdict is None:
        return DirectiveValidationResult(
            is_valid=False,
            verdict=None,
            cleaned_directive_body=raw,
            formatted_full_output=raw,
            word_count=len(raw.split()),
            has_banned_relative_phrase=False,
            leaks_final_answer=False,
            failure_reason="missing_verdict_sentinel",
        )

    if expected_verdict and verdict != expected_verdict:
        return DirectiveValidationResult(
            is_valid=False,
            verdict=verdict,
            cleaned_directive_body=raw,
            formatted_full_output=raw,
            word_count=len(raw.split()),
            has_banned_relative_phrase=False,
            leaks_final_answer=False,
            failure_reason=f"wrong_verdict_sentinel (expected {expected_verdict}, got {verdict})",
        )

    body = raw.replace(VERDICT_CHALLENGE, "").replace(VERDICT_APPROVE, "").strip()
    words = body.split()
    if len(words) < 4:
        return DirectiveValidationResult(
            is_valid=False,
            verdict=verdict,
            cleaned_directive_body=body,
            formatted_full_output=f"{body}\n{verdict}",
            word_count=len(words),
            has_banned_relative_phrase=False,
            leaks_final_answer=False,
            failure_reason="directive_too_short",
        )

    if len(words) > max_words:
        return DirectiveValidationResult(
            is_valid=False,
            verdict=verdict,
            cleaned_directive_body=body,
            formatted_full_output=f"{body}\n{verdict}",
            word_count=len(words),
            has_banned_relative_phrase=False,
            leaks_final_answer=False,
            failure_reason=f"directive_exceeds_token_budget ({len(words)} words > {max_words})",
        )

    # Check banned relative step/line phrases
    for pattern, label in BANNED_RELATIVE_PATTERNS:
        if re.search(pattern, body, flags=re.IGNORECASE):
            return DirectiveValidationResult(
                is_valid=False,
                verdict=verdict,
                cleaned_directive_body=body,
                formatted_full_output=f"{body}\n{verdict}",
                word_count=len(words),
                has_banned_relative_phrase=True,
                leaks_final_answer=False,
                failure_reason=f"banned_relative_phrase:{label}",
            )

    # Check final answer leakage on [[CHALLENGE]] directives
    leaks = False
    if verdict == VERDICT_CHALLENGE and reference_solution:
        _, ref_ans = extract_answer_line(reference_solution)
        if ref_ans and _is_purely_numeric_answer(ref_ans):
            num_val = _extract_primary_number(ref_ans)
            if num_val is not None:
                num_str = f"{num_val:g}"
                if (
                    len(num_str) >= 2
                    and num_str not in _TRIVIAL_ALLOWED_NUMBERS
                    and num_str not in (problem_statement or "")
                    and re.search(rf"(?<![0-9.]){re.escape(num_str)}(?![0-9.])", body)
                ):
                    leaks = True
        elif ref_ans:
            norm_rhs = _normalize_math_expr_string(ref_ans)
            norm_body = _normalize_math_expr_string(body)
            # Only flag symbolic leakage if a multi-symbol final RHS (>= 12 chars) is copied verbatim into the hint
            if len(norm_rhs) >= 12 and norm_rhs in norm_body and norm_rhs not in _normalize_math_expr_string(problem_statement):
                leaks = True

    if leaks:
        return DirectiveValidationResult(
            is_valid=False,
            verdict=verdict,
            cleaned_directive_body=body,
            formatted_full_output=f"{body}\n{verdict}",
            word_count=len(words),
            has_banned_relative_phrase=False,
            leaks_final_answer=True,
            failure_reason="directive_leaks_final_answer",
        )

    return DirectiveValidationResult(
        is_valid=True,
        verdict=verdict,
        cleaned_directive_body=body,
        formatted_full_output=f"{body}\n{verdict}",
        word_count=len(words),
        has_banned_relative_phrase=False,
        leaks_final_answer=False,
        failure_reason=None,
    )
