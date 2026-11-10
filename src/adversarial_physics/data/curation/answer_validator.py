"""Deterministic `ANSWER:` numerical (`±1.5%`), SymPy symbolic, and SI unit validator (`answer_validator.py`).

Used in Stage 2 (`CuratorKIT` curation) and Stage 6 (`AuditKIT` evaluation) to:
  1. Verify that **Alternative Solution 2** (`OPTICS`, `RELATIVITY`, `THERMODYNAMICS`) arrives at the
     exact same final `ANSWER:` (within `±1.5%` numerical tolerance or SymPy symbolic equivalence)
     and SI unit dimensions as `reference_solution`.
  2. Verify that **Flawed Candidate Solutions** (`aya-expanse-8b` controlled errors and `tiny-aya-global`
     harvested failures) demonstrably **fail** the `ANSWER:` equivalence check (filtering out any trajectory
     that accidentally arrived at the right final answer).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

# Standard SI unit tokens commonly appearing at the end of physics `ANSWER:` lines
_SI_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bm\s*/\s*s\^?2\b|\bm\s*s\^\{?-2\}?\b", "m/s^2"),
    (r"\bm\s*/\s*s\b|\bm\s*s\^\{?-1\}?\b", "m/s"),
    (r"\brad\s*/\s*s\b|\bs\^\{?-1\}?\b", "rad/s"),
    (r"\bkg\s*m\s*/\s*s\b", "kg*m/s"),
    (r"\b(?:joules?|J)\b", "J"),
    (r"\b(?:newtons?|N)\b", "N"),
    (r"\b(?:watts?|W)\b", "W"),
    (r"\b(?:pascals?|Pa)\b", "Pa"),
    (r"\b(?:volts?|V)\b", "V"),
    (r"\b(?:amperes?|amps?|A)\b", "A"),
    (r"\b(?:ohms?|\\Omega|Ω)\b", "Ohm"),
    (r"\b(?:teslas?|T)\b", "T"),
    (r"\b(?:hertz|Hz)\b", "Hz"),
    (r"\b(?:kelvins?|K)\b", "K"),
    (r"\b(?:coulombs?|C)\b", "C"),
    (r"\b(?:farads?|F)\b", "F"),
    (r"\b(?:henr(?:y|ies)|H)\b", "H"),
    (r"\b(?:electron-?volts?|eV|MeV|GeV)\b", "eV"),
    (r"\b(?:degrees?|deg|°)\b", "deg"),
    (r"\b(?:meters?|metres?|m)\b", "m"),
    (r"\b(?:seconds?|sec|s)\b", "s"),
    (r"\b(?:kilograms?|kg)\b", "kg"),
)

_NUM_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*(?:\\times|\*|×)\s*10\^\{?([+-]?\d+)\}?)?)"
)


@dataclass(frozen=True)
class AnswerValidationResult:
    """Result of comparing a candidate derivation's `ANSWER:` against `reference_solution`."""

    has_answer_sentinel: bool
    extracted_candidate_answer: str
    extracted_reference_answer: str
    is_equivalent: bool
    match_method: str  # "exact_normalized" | "numerical_tolerance" | "sympy_symbolic" | "mismatch" | "missing_sentinel"
    relative_error: float | None
    unit_consistent: bool
    candidate_unit: str | None
    reference_unit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_answer_line(solution_text: str) -> tuple[bool, str]:
    """Return `(has_sentinel, answer_value)` from the last `ANSWER:` line in `solution_text`."""
    if not solution_text:
        return False, ""
    matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:\*\*?)?ANSWER(?:\*\*?)?\s*:\s*(.+?)(?=\n\s*(?:\*\*?)?ANSWER|\Z)",
            solution_text.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return False, ""
    raw = matches[-1].group(1).strip()
    # Take first non-empty line or math block inside the trailing ANSWER segment
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and ln.strip() != "$$"]
    if not lines:
        return False, ""
    ans = " ".join(lines).strip()
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    ans = re.sub(r"^\\\(\s*|\s*\\\)$", "", ans).strip()
    ans = re.sub(r"^\\\[\s*|\s*\\\]$", "", ans).strip()
    return bool(ans), ans


def _normalize_math_expr_string(expr: str) -> str:
    """Normalize LaTeX/whitespace/LHS assignment for deterministic algebraic comparison."""
    s = (expr or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(\s*|\s*\\\)$", "", s).strip()
    s = re.sub(r"^\\\[\s*|\s*\\\]$", "", s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\bigl", "").replace("\\bigr", "").replace("\\Bigl", "").replace("\\Bigr", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("−", "-")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    # Strip leading LHS assignment like `v_0 =` or `T =` if there is an `=` sign
    if "=" in s:
        parts = [p.strip() for p in s.split("=") if p.strip()]
        if parts:
            s = parts[-1]
    s = s.strip(" .,;")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _extract_primary_number(ans_str: str) -> float | None:
    """Extract the primary numerical value from a numeric `ANSWER:` string if it is primarily numeric."""
    rhs = ans_str.split("=")[-1] if "=" in ans_str else ans_str
    # Strip SI unit exponent suffixes (e.g. m/s^2, s^{-2}, cm^3) so unit powers are not mistaken for the answer value
    rhs_no_unit_exp = re.sub(
        r"\b(?:m\s*/\s*s\^\{?[+-]?\d+\}?|s\^\{?[+-]?\d+\}?|m\^\{?[+-]?\d+\}?|cm\^\{?[+-]?\d+\}?)",
        "",
        rhs,
        flags=re.IGNORECASE,
    )
    matches = list(_NUM_RE.finditer(rhs_no_unit_exp.replace(",", "")))
    if not matches:
        matches = list(_NUM_RE.finditer(rhs.replace(",", "")))
    if not matches:
        return None
    # Take the leading numerical quantity on the RHS of the ANSWER expression
    m = matches[0]
    base_str = m.group(1)
    exp_part = m.group(2)
    try:
        if exp_part is not None:
            coeff_str = re.split(r"\\times|\*|×", base_str)[0].strip()
            val = float(coeff_str) * (10.0 ** float(exp_part))
        else:
            val = float(base_str)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def _is_purely_numeric_answer(ans_str: str) -> bool:
    """Return True if `ans_str` represents a numeric quantity (plus optional units) rather than a symbolic formula."""
    stripped = ans_str.strip()
    if "=" in stripped:
        stripped = stripped.split("=")[-1].strip()
    # Remove known unit words and LaTeX formatting
    no_units = re.sub(
        r"\\(?:times|cdot|approx|text|mathrm)\b|\b(?:m/s\^?2|m/s|rad/s|kg|m|s|N|J|W|Pa|V|A|T|Hz|K|C|F|H|eV|MeV|GeV|deg|times|greater|smaller)\b",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    # If LaTeX structural commands (\frac, \sqrt, \int, Greek letters) remain, it is symbolic
    if re.search(r"\\(?:frac|sqrt|alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|omega|pi|rho|sigma|tau|phi|varphi)", no_units):
        return False
    letters = re.findall(r"[a-zA-Z]", no_units.replace("e", "").replace("E", ""))
    return len(letters) <= 2 and bool(_NUM_RE.search(stripped))


def extract_si_unit(ans_str: str) -> str | None:
    """Extract normalized SI unit token from an `ANSWER:` string if present."""
    if not ans_str:
        return None
    rhs = ans_str.split("=")[-1] if "=" in ans_str else ans_str
    for pattern, norm_unit in _SI_UNIT_PATTERNS:
        if re.search(pattern, rhs):
            return norm_unit
    return None


def _latex_to_sympy_parseable(expr_str: str) -> str:
    """Convert simple LaTeX math expressions (`\\frac{a}{b}`, `\\sqrt{x}`, `x^{2}`) into SymPy-parseable Python syntax."""
    s = (expr_str or "").strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Convert \frac{A}{B} iteratively for simple non-nested braces
    for _ in range(4):
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = s.replace("^", "**")
    s = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|omega|pi|rho|sigma|tau|phi|varphi)", r" \1 ", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s.strip(" .,;")


def _check_sympy_equivalence(cand_ans: str, ref_ans: str) -> bool:
    """Attempt SymPy symbolic equivalence `simplify(expr_cand - expr_ref) == 0`."""
    try:
        import sympy  # type: ignore[import-not-found]
        from sympy.parsing.sympy_parser import (  # type: ignore[import-not-found]
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transforms = standard_transformations + (implicit_multiplication_application,)
        c_str = _latex_to_sympy_parseable(cand_ans)
        r_str = _latex_to_sympy_parseable(ref_ans)
        if not c_str or not r_str or len(c_str) > 180 or len(r_str) > 180:
            return False
        expr_c = parse_expr(c_str, transformations=transforms, evaluate=True)
        expr_r = parse_expr(r_str, transformations=transforms, evaluate=True)
        diff = sympy.simplify(expr_c - expr_r)
        if diff == 0:
            return True
        if expr_r != 0 and sympy.simplify(expr_c / expr_r) == 1:
            return True
    except Exception:
        return False
    return False


def validate_candidate_answer(
    candidate_solution: str,
    reference_solution: str,
    *,
    rel_tol: float = 0.015,
) -> AnswerValidationResult:
    """Compare a candidate solution's `ANSWER:` against `reference_solution` (`±1.5%` + SymPy + SI units)."""
    has_cand, cand_ans = extract_answer_line(candidate_solution)
    has_ref, ref_ans = extract_answer_line(reference_solution)

    cand_unit = extract_si_unit(cand_ans)
    ref_unit = extract_si_unit(ref_ans)
    unit_ok = True
    if ref_unit and cand_unit and ref_unit != cand_unit:
        unit_ok = False

    if not has_cand:
        return AnswerValidationResult(
            has_answer_sentinel=False,
            extracted_candidate_answer="",
            extracted_reference_answer=ref_ans,
            is_equivalent=False,
            match_method="missing_sentinel",
            relative_error=None,
            unit_consistent=False,
            candidate_unit=cand_unit,
            reference_unit=ref_unit,
        )

    norm_cand = _normalize_math_expr_string(cand_ans)
    norm_ref = _normalize_math_expr_string(ref_ans)

    if norm_cand and norm_ref and (norm_cand == norm_ref or norm_ref in norm_cand or norm_cand in norm_ref):
        if len(norm_cand) >= 2 and len(norm_ref) >= 2:
            return AnswerValidationResult(
                has_answer_sentinel=True,
                extracted_candidate_answer=cand_ans,
                extracted_reference_answer=ref_ans,
                is_equivalent=unit_ok,
                match_method="exact_normalized" if unit_ok else "mismatch",
                relative_error=0.0,
                unit_consistent=unit_ok,
                candidate_unit=cand_unit,
                reference_unit=ref_unit,
            )

    # Check numerical equivalence (`±1.5%` relative tolerance) when answers are numeric
    if _is_purely_numeric_answer(cand_ans) and _is_purely_numeric_answer(ref_ans):
        c_num = _extract_primary_number(cand_ans)
        r_num = _extract_primary_number(ref_ans)
        if c_num is not None and r_num is not None:
            denom = max(abs(r_num), 1e-12)
            rel_err = abs(c_num - r_num) / denom
            is_num_match = (rel_err <= rel_tol or abs(c_num - r_num) <= 1e-6) and unit_ok
            return AnswerValidationResult(
                has_answer_sentinel=True,
                extracted_candidate_answer=cand_ans,
                extracted_reference_answer=ref_ans,
                is_equivalent=is_num_match,
                match_method="numerical_tolerance" if is_num_match else "mismatch",
                relative_error=round(rel_err, 6),
                unit_consistent=unit_ok,
                candidate_unit=cand_unit,
                reference_unit=ref_unit,
            )

    # Check SymPy symbolic equivalence
    if _check_sympy_equivalence(cand_ans, ref_ans):
        return AnswerValidationResult(
            has_answer_sentinel=True,
            extracted_candidate_answer=cand_ans,
            extracted_reference_answer=ref_ans,
            is_equivalent=unit_ok,
            match_method="sympy_symbolic" if unit_ok else "mismatch",
            relative_error=0.0,
            unit_consistent=unit_ok,
            candidate_unit=cand_unit,
            reference_unit=ref_unit,
        )

    return AnswerValidationResult(
        has_answer_sentinel=True,
        extracted_candidate_answer=cand_ans,
        extracted_reference_answer=ref_ans,
        is_equivalent=False,
        match_method="mismatch",
        relative_error=None,
        unit_consistent=unit_ok,
        candidate_unit=cand_unit,
        reference_unit=ref_unit,
    )
