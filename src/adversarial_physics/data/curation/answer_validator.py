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


def _extract_all_numbers(text: str) -> list[tuple[float, str, int, int]]:
    """Extract ``(float_value, matched_substring, start, end)`` for numeric tokens in ``text``."""
    out: list[tuple[float, str, int, int]] = []
    if not text:
        return out
    cleaned = text.replace(",", "")
    for m in _NUM_RE.finditer(cleaned):
        base_str = m.group(1)
        exp_part = m.group(2)
        try:
            if exp_part is not None:
                coeff_str = re.split(r"\\times|\*|×", base_str)[0].strip()
                val = float(coeff_str) * (10.0 ** float(exp_part))
            else:
                val = float(base_str)
            if not (math.isnan(val) or math.isinf(val)):
                out.append((val, base_str, m.start(), m.end()))
        except Exception:
            continue
    return out


def _is_purely_numeric_answer(ans_str: str) -> bool:
    """Return True if `ans_str` represents a numeric quantity (plus optional trailing units), not symbolic algebra."""
    stripped = (ans_str or "").strip()
    if "=" in stripped:
        stripped = stripped.split("=")[-1].strip()
    stripped = re.sub(r"^\$+|\$+$", "", stripped).strip()
    # If LaTeX structural commands or transcendental functions / Greek symbols remain, it is symbolic
    if re.search(
        r"\\(?:frac|sqrt|arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|exp|log|ln|alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|omega|pi|rho|sigma|tau|phi|varphi)\b",
        stripped,
    ):
        return False

    cleaned = re.sub(r"\\(?:times|cdot|approx)\b", " ", stripped)
    cleaned = re.sub(r"\\text\s*\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", cleaned).strip()

    # Strip trailing SI unit or comparative suffixes only when NOT preceded by '/' (avoid stripping `v^2 / s` or `m` in `m v^2 / r`)
    trailing_unit_re = re.compile(
        r"(?:[\s,*]+|(?<=\d))("
        r"m\s*/\s*s\^?\{?[+-]?2\}?|m\s*/\s*s|rad\s*/\s*s|kg\s*\*?\s*m\s*/\s*s|N\s*\*?\s*m|"
        r"meters?|metres?|seconds?|sec|kilograms?|newtons?|joules?|watts?|pascals?|volts?|amperes?|amps?|ohms?|teslas?|hertz|kelvins?|coulombs?|farads?|henr(?:y|ies)|degrees?|"
        r"eV|MeV|GeV|Hz|Pa|deg|rad|kg|m|s|N|J|W|V|A|T|K|C|F|H|\\Omega|Ω|°|"
        r"times(?:\s+(?:greater|smaller|larger))?|smaller|greater|larger"
        r")\s*$",
        flags=re.IGNORECASE,
    )
    for _ in range(2):
        m = trailing_unit_re.search(cleaned)
        if not m or cleaned[: m.start()].rstrip().endswith("/"):
            break
        cleaned = cleaned[: m.start()].strip()

    # Ignore float scientific notation (`1.5e-4`) and explicit `x 10^k` markers before checking leftover variable letters
    no_sci = re.sub(r"(?<=\d)[eE][+-]?\d+", "", cleaned)
    no_sci = re.sub(r"(?<=\d)\s*[x×]\s*10\^\{?[+-]?\d+\}?", "", no_sci)
    letters = re.findall(r"[a-zA-Z]", no_sci)
    return len(letters) == 0 and bool(_NUM_RE.search(cleaned))


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
    # Collapse simple LaTeX subscripts like \rho_{1} -> \rho_1 before brace-to-paren conversion
    s = re.sub(r"_\{([A-Za-z0-9]+)\}", r"_\1", s)
    # Convert \frac{A}{B} iteratively for simple non-nested braces (leading space keeps `\pi\sqrt` separated)
    for _ in range(4):
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r" sqrt(\1)", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = s.replace("^", "**")
    s = re.sub(
        r"\\(alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|omega|pi|rho|sigma|tau|phi|varphi)(?![a-zA-Z])",
        r" \1 ",
        s,
    )
    # Preserve trig/hyperbolic/log/exp functions BEFORE stripping remaining LaTeX commands
    s = re.sub(r"\\ln(?![a-zA-Z])", " log ", s)
    s = re.sub(r"\\arcsin(?![a-zA-Z])", " asin ", s)
    s = re.sub(r"\\arccos(?![a-zA-Z])", " acos ", s)
    s = re.sub(r"\\arctan(?![a-zA-Z])", " atan ", s)
    s = re.sub(r"\\(sinh|cosh|tanh|sin|cos|tan|exp|log)(?![a-zA-Z])", r" \1 ", s)
    s = re.sub(r"\s+_", "_", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Bind bare math function arguments (`sin theta` -> `sin(theta)`) and tighten `sin (` -> `sin(`
    # so implicit space-to-`*` multiplication never produces `sin*(theta)`.
    _fn_names = r"(?:arcsin|arccos|arctan|asin|acos|atan|sinh|cosh|tanh|sin|cos|tan|exp|log|sqrt)"
    s = re.sub(rf"\b({_fn_names})\s*\(", r"\1(", s)
    s = re.sub(rf"\b({_fn_names})\s+([A-Za-z_][A-Za-z0-9_]*)\b", r"\1(\2)", s)
    # Handle `\sin^2(\theta)` -> `(sin(\theta))**2` after `^` -> `**`
    s = re.sub(rf"\b({_fn_names})\*\*\(?([0-9]+)\)?\s*\(([^()]+)\)", r"(\1(\3)**\2)", s)
    # Insert explicit multiplication for whitespace-separated physics symbols (e.g. "2 pi sqrt(...)" or "rho_1 H")
    s = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z(])", "*", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;")


def _parse_sympy_expression(expr_str: str, *, max_len: int = 180) -> tuple[Any, Any]:
    """Parse a normalized math/LaTeX string into ``(sympy_module, expr | None)``."""
    try:
        import sympy  # type: ignore[import-not-found]
        from sympy.parsing.sympy_parser import (  # type: ignore[import-not-found]
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        parsed_str = _latex_to_sympy_parseable(expr_str)
        if not parsed_str or len(parsed_str) > max_len:
            return sympy, None
        transforms = standard_transformations + (implicit_multiplication_application,)
        return sympy, parse_expr(parsed_str, transformations=transforms, evaluate=True)
    except Exception:
        return None, None


def _check_sympy_equivalence(cand_ans: str, ref_ans: str) -> bool:
    """Attempt SymPy symbolic equivalence `simplify(expr_cand - expr_ref) == 0`."""
    try:
        sympy, expr_c = _parse_sympy_expression(cand_ans, max_len=180)
        if sympy is None or expr_c is None:
            return False
        _, expr_r = _parse_sympy_expression(ref_ans, max_len=180)
        if expr_r is None:
            return False
        diff = sympy.simplify(expr_c - expr_r)
        if diff == 0:
            return True
        if expr_r != 0 and sympy.simplify(expr_c / expr_r) == 1:
            return True
    except Exception:
        return False
    return False


def evaluate_symbolic_answer_with_params(
    ans_str: str,
    check_params: dict[str, float],
) -> float | None:
    """Parse ``ans_str`` into SymPy via ``_latex_to_sympy_parseable`` and substitute ``check_params``."""
    try:
        # Strip trailing SI unit words only if they are isolated at the very end
        cleaned = re.sub(
            r"\s+(?:m/s\^?2|m/s|rad/s|N\*?m|N\s*[·*]\s*m|kg\*?m/s|kg|N|J|W|Pa|Hz|V|A|C|K|s|m|deg|rad|eV|MeV|GeV)\s*$",
            "",
            (ans_str or "").strip(),
        )
        sympy, expr = _parse_sympy_expression(cleaned, max_len=220)
        if sympy is None or expr is None:
            return None
        # Map string keys (including pi / g defaults if absent)
        subs_map: dict[Any, float] = {sympy.Symbol(k): float(v) for k, v in check_params.items()}
        for sym in list(expr.free_symbols):
            name = sym.name
            if name in check_params:
                subs_map[sym] = float(check_params[name])
            elif name == "pi":
                subs_map[sym] = math.pi
            elif name == "e":
                subs_map[sym] = math.e
            elif name == "g" and "g" not in check_params:
                subs_map[sym] = 9.81
        val = expr.evalf(subs=subs_map)
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except Exception:
        return None


def _check_aligntune_math_equivalence(cand_ans: str, ref_ans: str) -> bool:
    """Delegate secondary LaTeX/MATH symbolic equivalence to Lexsi AlignTune's `grade_math_answer`."""
    try:
        from aligntune.utils.math_grading import grade_math_answer

        cand_rhs = cand_ans.split("=")[-1].strip() if "=" in cand_ans else cand_ans.strip()
        ref_rhs = ref_ans.split("=")[-1].strip() if "=" in ref_ans else ref_ans.strip()
        cand_clean = re.sub(r"\\text\s*\{[^}]*\}", "", cand_rhs).strip()
        ref_clean = re.sub(r"\\text\s*\{[^}]*\}", "", ref_rhs).strip()
        return bool(cand_clean and ref_clean and grade_math_answer(cand_clean, ref_clean))
    except Exception:
        return False


def _detect_directive_leak(
    body: str,
    problem_statement: str,
    reference_solution: str,
    trivial_numbers: set[str] | frozenset[str],
) -> tuple[str | None, float | None, str | None, str | None]:
    """Inspect a Verifier directive body for numeric (incl. scientific notation) or symbolic answer leaks.

    Returns ``(leak_kind, leaked_num_val, leaked_num_token, norm_rhs)`` where ``leak_kind`` is
    ``"numeric"``, ``"symbolic"``, or ``None``.
    """
    _, ref_ans = extract_answer_line(reference_solution or "")
    if not ref_ans:
        return None, None, None, None

    if _is_purely_numeric_answer(ref_ans):
        num_val = _extract_primary_number(ref_ans)
        if num_val is None:
            return None, None, None, None
        trivial_floats = {float(x) for x in trivial_numbers}
        if any(math.isclose(num_val, tf, rel_tol=1e-6, abs_tol=1e-9) for tf in trivial_floats):
            return None, None, None, None
        stmt_nums = [v for v, _, _, _ in _extract_all_numbers(problem_statement or "")]
        if any(math.isclose(num_val, sv, rel_tol=1e-4, abs_tol=1e-9) for sv in stmt_nums):
            return None, None, None, None
        for b_val, b_tok, _, _ in _extract_all_numbers(body or ""):
            if math.isclose(num_val, b_val, rel_tol=1e-4, abs_tol=1e-9):
                return "numeric", num_val, b_tok, None
        return None, None, None, None

    norm_rhs = _normalize_math_expr_string(ref_ans)
    norm_body = _normalize_math_expr_string(body or "")
    norm_stmt = _normalize_math_expr_string(problem_statement or "")
    norm_rhs_flat = norm_rhs.replace("*", "")
    norm_body_flat = norm_body.replace("*", "")
    norm_stmt_flat = norm_stmt.replace("*", "")
    has_math_struct = bool(
        re.search(
            r"[*/^+\-]|\\frac|\\sqrt|\\(?:arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|exp|log|ln|pi)\b|\d+[a-zA-Z\\]",
            norm_rhs,
        )
    )
    is_meaningful_expr = len(norm_rhs) >= 12 or (has_math_struct and len(norm_rhs) >= 4)
    if (
        is_meaningful_expr
        and (norm_rhs in norm_body or norm_rhs_flat in norm_body_flat)
        and norm_rhs not in norm_stmt
        and norm_rhs_flat not in norm_stmt_flat
    ):
        return "symbolic", None, None, norm_rhs
    return None, None, None, None


def redact_numeric_answer_leaks(body: str, target_val: float, replacement: str = "the target value") -> str:
    """Replace any numeric token in ``body`` matching ``target_val`` across decimal/comma/scientific notations."""
    num_pat = re.compile(
        r"([+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*(?:\\times|\*|×)\s*10\^\{?([+-]?\d+)\}?)?)"
    )

    def _repl(m: re.Match[str]) -> str:
        raw = m.group(1)
        exp_part = m.group(2)
        try:
            cleaned = raw.replace(",", "")
            if exp_part is not None:
                coeff_str = re.split(r"\\times|\*|×", cleaned)[0].strip()
                val = float(coeff_str) * (10.0 ** float(exp_part))
            else:
                val = float(cleaned)
            if math.isclose(target_val, val, rel_tol=1e-4, abs_tol=1e-9):
                return replacement
        except Exception:
            pass
        return raw

    return num_pat.sub(_repl, body)


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

    if norm_cand and norm_ref and norm_cand == norm_ref:
        if len(norm_cand) >= 1 and len(norm_ref) >= 1:
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

    # Check SymPy symbolic equivalence (project parser + AlignTune MATH grader fallback)
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

    if _check_aligntune_math_equivalence(cand_ans, ref_ans):
        return AnswerValidationResult(
            has_answer_sentinel=True,
            extracted_candidate_answer=cand_ans,
            extracted_reference_answer=ref_ans,
            is_equivalent=unit_ok,
            match_method="aligntune_sympy" if unit_ok else "mismatch",
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
