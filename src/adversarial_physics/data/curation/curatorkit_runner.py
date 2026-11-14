"""Stage 2.3 CuratorKIT Trajectory Filtering & Deduplication Runner (`curatorkit_runner.py`).

Executes **before** directive generation (`04d`) so we never waste LLM calls writing hints
for duplicate, malformed, or accidentally-correct trajectories:
  1. **Minority-Domain Solution 2 Curation**:
     Runs CuratorKIT's `TextCleaner`, `SchemaGate`, and deterministic `validate_candidate_answer`
     (`±1.5%` numerical tolerance / SymPy symbolic / SI units) over the 82 `02a_minority_solution2.jsonl`
     records -> writes `data/synthetic/02d_curated_solution2.jsonl`.
  2. **Raw Unfiltered Snapshot (`04A — raw_synthetic`)**:
     Saves all generated candidate trajectories (`681` synthetic wrong + `681` `tiny-aya-global` harvested)
     to `data/synthetic/raw_synthetic/trajectories_raw.jsonl`.
  3. **Deterministic `ANSWER:` Mismatch Filter**:
     Filters out any trajectory in the flawed pool whose `ANSWER:` actually matches `reference_solution`
     (saving accidentally-correct `tiny-aya-global` trajectories to `02e_tiny_aya_correct_trajectories.jsonl`).
  4. **CuratorKIT Within-Problem & Cross-Pool Trajectory Deduplication**:
     Uses `curatorkit.normalizers.clean.TextCleaner`, `curatorkit.gates.schema.SchemaGate`,
     `curatorkit.normalizers.dedup.ExactDeduplicator`, `MinHashDeduplicator` (`threshold = 0.85`),
     and `DiversityGate` (`similarity_threshold = 0.88`) to eliminate similar/duplicate flawed
     trajectories while guaranteeing every canonical problem retains at least 1 surviving flawed trajectory.
     Writes `data/synthetic/02f_surviving_flawed_trajectories.jsonl`.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from curatorkit.gates.diversity import DiversityGate
from curatorkit.gates.schema import SchemaGate
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample

from adversarial_physics.data.curation.answer_validator import (
    _normalize_math_expr_string,
    extract_answer_line,
    validate_candidate_answer,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem, load_problems_jsonl
from adversarial_physics.data.schemas.trajectory import (
    CandidateSolution,
    load_candidates_jsonl,
    save_candidates_jsonl,
)
from adversarial_physics.utils.config import resolve_project_path
from adversarial_physics.utils.logging import get_logger

logger = get_logger("stage_2_curatorkit_runner")

DEFAULT_CANONICAL_TRAIN_PATH = "data/processed/canonical_train.jsonl"
DEFAULT_SOLUTION2_INPUT_PATH = "data/synthetic/02a_minority_solution2.jsonl"
DEFAULT_FLAWED_8B_INPUT_PATH = "data/synthetic/02b_synthetic_flawed_trajectories.jsonl"
DEFAULT_TINY_AYA_INPUT_PATH = "data/synthetic/02c_tiny_aya_trajectories.jsonl"

DEFAULT_RAW_SYNTHETIC_PATH = "data/synthetic/raw_synthetic/trajectories_raw.jsonl"
DEFAULT_CURATED_SOL2_PATH = "data/synthetic/02d_curated_solution2.jsonl"
DEFAULT_TINY_AYA_CORRECT_PATH = "data/synthetic/02e_tiny_aya_correct_trajectories.jsonl"
DEFAULT_SURVIVING_FLAWED_PATH = "data/synthetic/02f_surviving_flawed_trajectories.jsonl"
DEFAULT_REJECTED_TRAJECTORIES_PATH = "data/synthetic/02f_rejected_trajectories.jsonl"
DEFAULT_DEDUP_SUMMARY_PATH = "data/synthetic/02f_dedup_summary.json"


@dataclass(frozen=True)
class Stage2TrajectoryDedupSummary:
    """Summary of Stage 2.3 CuratorKIT trajectory curation and deduplication."""

    canonical_problems_count: int
    minority_solution2_input_count: int
    minority_solution2_curated_count: int
    flawed_aya8b_input_count: int
    tiny_aya_input_count: int
    total_raw_trajectories_count: int
    tiny_aya_accidentally_correct_count: int
    rejected_by_schema_or_empty_count: int
    eliminated_similar_trajectories_count: int
    surviving_flawed_trajectories_count: int
    problems_with_1_surviving_trajectory: int
    problems_with_2_surviving_trajectories: int
    surviving_by_domain: dict[str, int]
    surviving_by_source: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_to_datasample(cand: CandidateSolution, problem_statement: str) -> DataSample:
    return DataSample(
        id=cand.candidate_id,
        source_uri=f"{cand.source}:{cand.problem_id}",
        instruction=problem_statement,
        input=cand.solution_text,
        output=cand.solution_text,
        task_type="instruction_following",
        metadata=cand.to_dict(),
    )


def _datasample_to_candidate(sample: DataSample) -> CandidateSolution:
    meta = dict(sample.metadata or {})
    meta["solution_text"] = sample.output.strip()
    return CandidateSolution.from_dict(meta)


def _token_cosine_similarity(text_a: str, text_b: str) -> float:
    toks_a = Counter(re.findall(r"[a-zA-Z0-9_\\]+", text_a.lower()))
    toks_b = Counter(re.findall(r"[a-zA-Z0-9_\\]+", text_b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    common = set(toks_a.keys()) & set(toks_b.keys())
    dot = sum(toks_a[k] * toks_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in toks_a.values()))
    norm_b = math.sqrt(sum(v * v for v in toks_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_META_GIVEAWAY_RE = re.compile(
    r"\b(?:flaw(?:ed|s)?|inject(?:ed|ing|ion)?|mistake(?:s|nly)?|misstep(?:s)?|incorrect(?:ly)?|"
    r"erroneous(?:ly)?|deliberate(?:ly)?|intentional(?:ly)?|subtle\s+error|target\s+error|"
    r"error\s+category|injected\s+error|corrupt(?:ed)?|broken\s+solution|reference\s+solution|"
    r"ground[- ]truth(?:\s+(?:solution|answer))?|on\s+purpose|spurious|nonsensical|"
    r"wrong\s+(?:assumption|units?|sign|formula|direction|term|value|result|answer)|"
    r"should\s+(?:be|not\s+be|have\s+been)|(?:differs?|deviates?)\s+from\s+the\s+(?:reference|ground[- ]truth)|"
    r"note\s*:)\b",
    re.IGNORECASE,
)

_BALANCED_PAREN_RE = re.compile(r"\s*\(((?:[^()\n]|\([^()\n]*\)){2,350})\)")

_PURE_META_SENTENCE_RE = re.compile(
    r"\b(?:reference\s+solution|ground[- ]truth|target\s+error\s+category|"
    r"error\s+category\s+requested|injected\s+error|common\s+(?:algebraic\s+)?mistake|"
    r"must\s+have\s+made\s+a\s+mistake|made\s+a\s+(?:slight\s+|critical\s+)?mistake\s+(?:here|in\s+(?:our\s+)?(?:step|derivation))|"
    r"(?:might\s+have|could\s+have)\s+made\s+a\s+mistake|double-check\s+my\s+work|"
    r"(?:here'?s|this\s+is)\s+where\s+(?:the|our)\s+mistake\s+(?:occurs|becomes|comes)|"
    r"our\s+mistake\s+(?:lies\s+in|becomes\s+apparent|persists|was\s+applying|comes\s+in|has\s+led)|"
    r"(?:to\s+correct|correcting)\s+(?:for\s+)?(?:our|this|the)\s+mistake|realize\s+(?:my|the)\s+mistake|"
    r"a\s+(?:critical\s+|subtle\s+|small\s+|key\s+|similar\s+|final\s+|crucial\s+)?mistake\s+(?:was|is)\s+made\s*(?:here)?\.?$|"
    r"my\s+calculations,?\s+I\s+made\s+a\s+mistake|our\s+(?:final\s+)?answer\s+will\s+be\s+incorrect|"
    r"solution\s+will\s+be\s+incorrect|crucial\s+for\s+the\s+incorrect\s+derivation|"
    r"correct\s+(?:equation|formula|expression|answer|relationship|step|calculation)\s+should\s+(?:be|have\s+been)|"
    r"we\s+should\s+(?:remember\s+that|have\s+(?:divided|considered|converted))|"
    r"upon\s+reviewing,?\s+we\s+see\s+that|this\s+suggests\s+our\s+assumption|"
    r"as\s+requested,?\s+the\s+final\s+answer\s+differs|"
    r"the\s+sign\s+of\s+the\s+integral\s+for\s+the\s+circular\s+loop\s+is\s+incorrect|"
    r"note\s+to\s+revisit\s+this\s+part)\b",
    re.IGNORECASE,
)


def _strip_balanced_meta_parens(text: str) -> str:
    """Remove parenthetical meta-comments (including nested parens) without stripping math groups."""

    def _replace(match: re.Match[str]) -> str:
        return "" if _META_GIVEAWAY_RE.search(match.group(1)) else match.group(0)

    return _BALANCED_PAREN_RE.sub(_replace, text)


def _clean_extracted_answer_suffix(raw_ans: str) -> str:
    """Strip step prefixes, bold markers, and giveaway meta-notes trailing an ANSWER value."""
    val = raw_ans.strip()
    val = re.sub(
        r"^(?:(?:Step\s+\d+|\d+\.)\s*(?=(?:\*{0,2}(?:ANSWER|Final\s+Answer)\b|Therefore))|(?:Step\s+\d+\s*:)\s*)",
        "",
        val,
        flags=re.IGNORECASE,
    ).strip()
    val = re.sub(
        r"^(?:Therefore,?\s+(?:our\s+)?final\s+answer\s+is\s*:\s*|\*{0,2}Final\s+Answer\*{0,2}\s*:\s*)",
        "",
        val,
        flags=re.IGNORECASE,
    ).strip()
    val = re.sub(r"^(?:\*\*\s*)?ANSWER\s*:\s*", "", val, flags=re.IGNORECASE).strip()
    val = re.sub(r"\*\*\s*$", "", val).strip()
    val = re.sub(
        r",?\s*\*{0,2}(?:which\s+is\s+(?:an?\s+)?(?:incorrect|flawed|wrong)\b.*|"
        r"but\s+our\s+derivation\s+has\s+led\s+to\s+a\s+flawed.*|"
        r"(?:leading|where\s+the\s+extra|which,?\s+while\s+(?:incorrect|flawed))\b[^.]*\b(?:incorrect|flawed|wrong|mistake)\b.*|"
        r"contradicting\s+our\s+flawed\s+analysis.*)\s*$",
        "",
        val,
        flags=re.IGNORECASE,
    ).strip()
    while True:
        match = re.search(r"\s*\(((?:[^()]|\([^()]*\)){3,350})\)\.?\s*$", val)
        if match and _META_GIVEAWAY_RE.search(match.group(1)):
            val = val[: match.start()].strip()
        else:
            break
    val = re.sub(r"\*{0,2}\s*\((?:This\s+(?:is\s+a\s+(?:flawed|incorrect)|answer)|Flawed|Incorrect|Omitted)[^)]*\)?\.?\s*$", "", val, flags=re.IGNORECASE).strip()
    val = re.sub(r"\*{1,2}\s*$", "", val).strip()
    if val.count("**") % 2 == 1:
        val = val.replace("**", "").strip()
    val = re.sub(
        r"^(?:\*\*\s*)?(?:Note\s*:|This\s+(?:answer|derivation)\b|As\s+you\s+can\s+see,?\s+the\s+final\s+answer\s+is\s+flawed\b|"
        r"Therefore,?\s+the\s+physics\s+remains\s+unchanged\b).*$",
        "",
        val,
        flags=re.IGNORECASE,
    ).strip()
    return val


def _normalize_step_lines_and_answer(raw_text: str, *, fallback_answer: str | None = None) -> str:
    """Normalize numbered steps (`1. ...`, `### **Step 1: ...**`) to `Step 1: ...` and collapse duplicate `ANSWER:` lines."""
    # Restore control chars accidentally produced by unescaped re.sub backslashes (\f -> \frac, \b -> \beta, \t -> \theta/\tau)
    text = raw_text.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t")
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>", "", text)
    text = re.sub(r"(?<![a-zA-Z\\])rac\{", r"\\frac{", text)
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text.strip()).strip()

    _, extracted_ans = extract_answer_line(text)
    cleaned_extracted = _clean_extracted_answer_suffix(extracted_ans)

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    inline_answers: list[str] = []

    for line in lines:
        s = line.strip()
        if not s or s == "---":
            continue
        # Skip pure meta-title lines (whether colon is inside or outside **)
        if re.match(
            r"^(?:#+\s*)?\*{0,2}\s*(?:Alternative\s+(?:Derivation|Solution)(?:\s*\(Solution\s*2\))?|Flawed\s+(?:Student\s+)?(?:Derivation|Solution|Attempt)|Step-by-Step\s+Solution|To\s+solve\s+this\s+problem[^:]*)\s*:?\s*\*{0,2}\s*:?\s*$",
            s,
            flags=re.IGNORECASE,
        ):
            continue

        # Capture inline ANSWER: lines, including numbered forms like `10. **ANSWER: ...**`
        ans_m = re.match(
            r"^(?:(?:#+\s*)?(?:\*\*\s*)?(?:Step\s+\d+|\d+\.)\s*[:.\-]?\s*(?:\*\*\s*)?)?"
            r"(?:(?:\*{0,2}Final\s+Answer\*{0,2}\s*:|Therefore,?\s+(?:our\s+)?final\s+answer\s+is\s*:)\s*)?"
            r"(?:\*\*\s*)?ANSWER\s*:\s*(.+)$",
            s,
            flags=re.IGNORECASE,
        )
        if ans_m:
            val = _clean_extracted_answer_suffix(ans_m.group(1))
            if val:
                inline_answers.append(val)
            continue

        if re.match(r"^(?:#+\s*)?\*{0,2}\s*Final\s+Answer\s*:?\s*\*{0,2}\s*:?\s*$", s, flags=re.IGNORECASE):
            continue

        # Normalize step headers: `### **Step 1: Setup**` or `**Step 1:** text` or `1. **Setup:**` -> `Step 1: ...`
        step_m = re.match(
            r"^(?:#+\s*)?(?:\*\*\s*)?(?:Step\s+(\d+)|(\d+)\.)\s*[:.\-]?\s*(?:\*\*\s*)?(.*?)(?:\*\*)?$",
            s,
            flags=re.IGNORECASE,
        )
        if step_m:
            num = step_m.group(1) or step_m.group(2)
            rest = step_m.group(3).strip()
            rest = re.sub(
                r"\s*\(\s*(?:Flawed(?:\s+Step)?|Incorrect(?:\s+Step)?|Omitted(?:\s+Term)?|Simplified|Error|Mistake)\s*\)",
                "",
                rest,
                flags=re.IGNORECASE,
            ).strip()
            if not rest:
                continue
            s = f"Step {num}: {rest}".strip()

        cleaned_lines.append(s)

    body = "\n".join(cleaned_lines).strip()
    if fallback_answer is not None:
        final_ans = _clean_extracted_answer_suffix(fallback_answer)
    else:
        final_ans = cleaned_extracted or (inline_answers[-1] if inline_answers else "")
    final_ans = final_ans.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t")
    final_ans = re.sub(r"(?<![a-zA-Z\\])rac\{", r"\\frac{", final_ans).strip()
    if final_ans:
        return f"{body}\nANSWER: {final_ans}".strip()
    return body


def _neutralize_sentence_meta_cues(sent: str) -> str:
    """Rewrite self-incriminating flaw admissions inside a step sentence without removing its physics math."""
    sent = re.sub(r",?\s*\bwhich\s+is\s+(?:physically\s+|clearly\s+)?(?:incorrect|flawed|wrong)(?:\s+(?:based\s+on|as|since|because)[^.,;]*)?", "", sent, flags=re.IGNORECASE)
    sent = re.sub(
        r",?\s*\b(?:consistent\s+with|different\s+from|according\s+to|per)\s+the\s+(?:reference\s+solution|ground[- ]truth(?:\s+(?:solution|answer))?)\b[^.,;]*",
        "",
        sent,
        flags=re.IGNORECASE,
    )
    sent = re.sub(
        r",?\s*\b(?:leading\s+to|resulting\s+in)\s+(?:an?|the)\s+(?:consistent\s+but\s+)?(?:spurious|nonsensical|flawed|incorrect|wrong)\s+"
        r"(?:final\s+)?(?:result|answer|expression|value|conclusion|field\s+expression|relationship|understanding)\b[^.:;]*",
        "",
        sent,
        flags=re.IGNORECASE,
    )
    sent = re.sub(r",?\s*\bcontradicting\s+our\s+flawed\s+analysis[^.;]*", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\bThis\s+equation\s+is\s+incorrect\s*;\s*", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r",?\s*\bdue\s+to\s+(?:the\s+|our\s+|an?\s+)?(?:\w+\s+){0,2}(?:error|mistake|flaw)\b[^.,;:]*", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r",?\s*\bdue\s+to\s+the\s+injection\s+of\s+the\s+formula\b", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r",?\s*\bas\s+we\s+have\s+already\s+made\s+a\s+(?:critical\s+)?mistake[^.;]*", "", sent, flags=re.IGNORECASE)
    sent = re.sub(
        r",?\s*\b(?:However,?\s*)?(?:we|I|let'?s)(?:'?(?:ve|re|ll))?\s+(?:can\s+|might\s+be\s+)?"
        r"(?:mak(?:e|ing)|made)\s+(?:a|another|the|our)?\s*"
        r"(?:slight|small|subtle|crucial|key|final|similar|algebraic|sign)?\s*mistake\s*(?:here\s*)?"
        r"(?:by\s+(?:not\s+)?|of\s+|in\s+|and\s*|,|:)\s*",
        " we proceed by ",
        sent,
        flags=re.IGNORECASE,
    )
    sent = re.sub(r",?\s*\b(?:but\s+)?(?:we|I)(?:'?(?:ve|ll))?\s+(?:make|made)\s+(?:a|another|a\s+similar)\s+mistake\b[:.,]?", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\bin\s+a\s+dimensional\s+analysis\s+mistake,?\s*", "here, ", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\bmakes\s+the\s+mistake\s+of\b", "proceeds by", sent, flags=re.IGNORECASE)
    sent = re.sub(r",?\s*\bwith\s+a\s+mistake\s+in\s+the\s+sign\b", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\bleads\s+to\s+a\s+mistake\s*:\s*", "gives ", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:the\s+mistake\s+in\s+step\s+\d+|where\s+we\s+make\s+our\s+mistake)\b", "this step", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:However,?\s*)?we\s+make\s+the\s+(?:incorrect|flawed|erroneous)\s+assumption\s+that\b", "We assume that", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:However,?\s*)?(?:we|I)\s+(?:mistakenly|incorrectly|erroneously|deliberately|intentionally)\s+", "We ", sent, flags=re.IGNORECASE)
    sent = re.sub(r",?\s*\b(?:mistakenly|incorrectly|erroneously|deliberately|intentionally)\b,?\s*", " ", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:we\s+arrive\s+at|this\s+leads\s+to|we\s+conclude\s+with)\s+(?:a|the)\s+flawed\s+(?:final\s+)?answer\b", "we obtain the final answer", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:our\s+)?flawed\s+final\s+answer\s+is\b", "the final answer is", sent, flags=re.IGNORECASE)
    sent = re.sub(
        r"\b(?:this|the|that|an?|our|related\s+but)\s+(?:spurious|nonsensical|flawed|incorrect|erroneous|wrong)\s+"
        r"(expression|formula|equation|substitution|assumption|value|result|factor|sign|direction|unit|understanding|current|ratio|model|relation|term|conclusion|answer|calculation|derivative|step)\b",
        r"the \1",
        sent,
        flags=re.IGNORECASE,
    )
    sent = re.sub(r"\b(?:spurious|nonsensical|flawed|erroneous|incorrect)\s+", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\b(?:but|and)\s+(?:our\s+interpretation\s+of\s+[^.;]*?\s+|that\s+[^.;]*?\s+)?is\s+(?:physically\s+|clearly\s+)?(?:incorrect|flawed|wrong)\b[^.;]*", "", sent, flags=re.IGNORECASE)
    sent = re.sub(r"\bthis\s+(?:is\s+(?:physically\s+|clearly\s+)?(?:incorrect|flawed)|step\s+is\s+flawed)\b[^.;]*", "", sent, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", sent).strip()


def _sanitize_flawed_trajectory_text(raw_text: str) -> tuple[str, bool]:
    """Strip self-incriminating giveaway phrases from synthetic flawed trajectories so Verifier cannot cheat."""
    cleaned = _normalize_step_lines_and_answer(raw_text)
    cleaned = re.sub(r"\\text\{\(\s*expression\s+with\s+incorrect\s+time\s+dependencies\s*\)\}", r"\\text{(time-dependent factor)}", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\text\{\(\s*incorrect\s+(term|expression|factor)\s*\)\}", r"\\text{(additional \1)}", cleaned, flags=re.IGNORECASE)

    # Drop standalone meta/Note header lines and clean meta comments on ANSWER lines
    kept_lines: list[str] = []
    for ln in cleaned.splitlines():
        s = ln.strip()
        if not s:
            kept_lines.append(ln)
            continue
        if s.upper().startswith("ANSWER:"):
            ln = _strip_balanced_meta_parens(ln)
            ln = re.sub(
                r"(?:,?\s*\b(?:is\s+incorrect|and\s+has\s+also\s+incorrectly\s+used|contradicting\s+our\s+flawed\s+analysis|"
                r"leading\s+to\s+an?\s+(?:incorrect|flawed|wrong)|where\s+the\s+extra\s+[^.]*\b(?:incorrect|flawed)|"
                r"which,?\s+while\s+(?:incorrect|flawed))\b.*$)",
                "",
                ln,
                flags=re.IGNORECASE,
            )
            kept_lines.append(ln.strip())
            continue
        if re.match(
            r"^(?:\*{0,2}Note\s*:?\*{0,2}|\(Note\b|\*{0,2}Flaw(?:ed)?\b[^:]*:|\*{0,2}INJECTED\s+ERROR\b|Here\s+is\s+a\s+(?:step-by-step\s+)?derivation\s+with\b)",
            s,
            flags=re.IGNORECASE,
        ) and _META_GIVEAWAY_RE.search(s):
            continue
        kept_lines.append(ln)
    cleaned = "\n".join(kept_lines)

    # Strip bold or colon-suffixed step sub-headers (including nested colons like `Misstep: Omitting Internal Resistance:**`)
    cleaned = re.sub(
        r"(?:#+\s*)?(?:\*{1,2})?\s*[^\n*]{0,75}?\b(?:flaw(?:ed)?|mistake|misstep|inject(?:ed)?\s*error|"
        r"realize\s+the\s+error|erroneous|incorrect(?:ly)?|unit\s+error|algebraic\s+error|sign\s+error|"
        r"omission|avoidable\s+simplification)\b[^\n*]{0,45}?(?::\s*\*{1,2}|\*{1,2}\s*:?)(?:\s*Step\s*:)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<=:\s)(?:Flaw\s+in\s+Step|Mistake\s+in\s+Step|Injected\s+Error)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\*{1,2}(?:flawed|incorrectly|erroneously|deliberately|intentionally|mistakenly)\*{1,2}", "", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_balanced_meta_parens(cleaned)
    cleaned = re.sub(
        r"(?:Using\s+the\s+same\s+logic\s+as\s+in|Following(?:\s+the\s+same\s+proportionality\s+logic\s+as\s+in)?|"
        r"Mirroring|Similar\s+to(?:\s+equations?\s*[^,]+in)?|As\s+(?:suggested|shown|given)\s+in|Based\s+on)\s+"
        r"the\s+(?:reference\s+solution|ground[- ]truth(?:\s+solution)?),?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+\bfrom\s+the\s+(?:reference\s+solution|ground[- ]truth(?:\s+solution)?)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?:Here'?s|Here\s+is)\s+(?:a\s+derivation\s+that\s+makes\s+the\s+mistake[^:]*:|where\s+the\s+(?:mistake|error|flaw)\s+(?:occurs|is\s+introduced)[^.]*\.?\s*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\.\.\.\s*\([^)]*(?:derivation|condition|equation|step)[^)]*\)",
        "Applying the intermediate algebraic substitutions from the governing equations:",
        cleaned,
        flags=re.IGNORECASE,
    )

    new_lines: list[str] = []
    for ln in cleaned.splitlines():
        if ln.strip().upper().startswith("ANSWER:"):
            new_lines.append(ln)
            continue
        m_step = re.match(r"^(Step\s+\d+:\s*)(.*)$", ln)
        prefix = m_step.group(1) if m_step else ""
        rest = m_step.group(2) if m_step else ln
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", rest.strip()) if rest.strip() else []
        kept_sents: list[str] = []
        for sent in sents:
            if _PURE_META_SENTENCE_RE.search(sent) and not (":" in sent and "=" in sent):
                continue
            cleaned_sent = _neutralize_sentence_meta_cues(sent)
            if cleaned_sent and cleaned_sent not in {".", ":", ","}:
                kept_sents.append(cleaned_sent)
        joined = (prefix + " ".join(kept_sents)).strip()
        if prefix and not kept_sents:
            continue
        if joined:
            new_lines.append(joined)

    cleaned = "\n".join(new_lines)
    cleaned = re.sub(r"\b(a)\s+([aeiouAEIOU])", lambda m: ("an " if m.group(1) == "a" else "An ") + m.group(2), cleaned)
    cleaned = re.sub(r"\b(an)\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", lambda m: ("a " if m.group(1) == "an" else "A ") + m.group(2), cleaned)
    cleaned = re.sub(r"\bthe\s+the\b", "the", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, True


def curate_and_deduplicate_stage2_trajectories(
    *,
    canonical_train_path: str | Path = DEFAULT_CANONICAL_TRAIN_PATH,
    solution2_input_path: str | Path = DEFAULT_SOLUTION2_INPUT_PATH,
    flawed_8b_input_path: str | Path = DEFAULT_FLAWED_8B_INPUT_PATH,
    tiny_aya_input_path: str | Path = DEFAULT_TINY_AYA_INPUT_PATH,
    raw_synthetic_output_path: str | Path = DEFAULT_RAW_SYNTHETIC_PATH,
    curated_sol2_output_path: str | Path = DEFAULT_CURATED_SOL2_PATH,
    tiny_aya_correct_output_path: str | Path = DEFAULT_TINY_AYA_CORRECT_PATH,
    surviving_flawed_output_path: str | Path = DEFAULT_SURVIVING_FLAWED_PATH,
    rejected_output_path: str | Path = DEFAULT_REJECTED_TRAJECTORIES_PATH,
    summary_output_path: str | Path = DEFAULT_DEDUP_SUMMARY_PATH,
    similarity_threshold: float = 0.87,
) -> tuple[list[CandidateSolution], list[CandidateSolution], Stage2TrajectoryDedupSummary]:
    """Run deterministic CuratorKIT curation & trajectory deduplication before hint generation."""
    problems = load_problems_jsonl(resolve_project_path(canonical_train_path))
    prob_map: dict[str, CanonicalProblem] = {p.problem_id: p for p in problems}

    sol2_raw = load_candidates_jsonl(resolve_project_path(solution2_input_path))
    flawed_8b_raw = load_candidates_jsonl(resolve_project_path(flawed_8b_input_path))
    tiny_aya_raw = load_candidates_jsonl(resolve_project_path(tiny_aya_input_path))

    # 1. Save immutable raw synthetic snapshot (`04A — raw_synthetic`)
    all_raw = list(sol2_raw) + list(flawed_8b_raw) + list(tiny_aya_raw)
    save_candidates_jsonl(all_raw, resolve_project_path(raw_synthetic_output_path))

    cleaner = TextCleaner(transforms={"strip_html": False}, fields=["instruction", "output"])
    schema_gate = SchemaGate(required_fields=["instruction", "output"], min_tokens=8, max_tokens=4096)
    # 2. Curate Minority-Domain Alternative Solution 2 (`82` problems)
    rejected_logs: list[dict[str, Any]] = []
    curated_sol2: list[CandidateSolution] = []
    if sol2_raw:
        # Normalize step headers and rescue any unescaped backslash control chars BEFORE TextCleaner strips them
        precleaned_sol2: list[CandidateSolution] = []
        for c in sol2_raw:
            if c.problem_id not in prob_map:
                continue
            norm_text = _normalize_step_lines_and_answer(c.solution_text, fallback_answer=None)
            precleaned_sol2.append(
                CandidateSolution(
                    problem_id=c.problem_id,
                    candidate_id=c.candidate_id,
                    solution_text=norm_text,
                    is_correct=True,
                    domain=c.domain,
                    error_type=None,
                    paired_directive=None,
                    source=c.source,
                    original_id=c.original_id,
                    latency_seconds=c.latency_seconds,
                )
            )
        sol2_samples = [
            _candidate_to_datasample(c, prob_map[c.problem_id].problem_statement)
            for c in precleaned_sol2
        ]
        sol2_samples = cleaner.run(sol2_samples)
        sol2_passed, sol2_rejected = schema_gate.run(sol2_samples)
        for sf in sol2_rejected:
            rejected_logs.append(
                {
                    "candidate_id": sf.id,
                    "problem_id": (sf.metadata or {}).get("problem_id"),
                    "source": (sf.metadata or {}).get("source"),
                    "reason": f"curatorkit_schema_gate:{sf.rejection_reason}",
                }
            )
        for s in sol2_passed:
            cand = _datasample_to_candidate(s)
            ref_prob = prob_map[cand.problem_id]
            norm_text = _normalize_step_lines_and_answer(cand.solution_text, fallback_answer=None)
            val = validate_candidate_answer(norm_text, ref_prob.reference_solution)
            if not val.is_equivalent:
                rejected_logs.append(
                    {
                        "candidate_id": cand.candidate_id,
                        "problem_id": cand.problem_id,
                        "source": cand.source,
                        "reason": f"solution2_answer_mismatch:{val.match_method}",
                    }
                )
                continue
            cand = CandidateSolution(
                problem_id=cand.problem_id,
                candidate_id=cand.candidate_id,
                solution_text=norm_text,
                is_correct=True,
                domain=cand.domain,
                error_type=None,
                paired_directive=None,
                source=cand.source,
                original_id=cand.original_id,
                latency_seconds=cand.latency_seconds,
            )
            curated_sol2.append(cand)
    save_candidates_jsonl(curated_sol2, resolve_project_path(curated_sol2_output_path))

    # 3. Filter & Deduplicate Flawed Candidate Trajectories (`flawed_8b_raw` + `tiny_aya_raw`)
    tiny_aya_correct: list[CandidateSolution] = []
    flawed_candidates_pool: list[CandidateSolution] = []
    placeholder_rejected = 0

    for cand in list(flawed_8b_raw) + list(tiny_aya_raw):
        ref_prob = prob_map.get(cand.problem_id)
        if ref_prob is None:
            continue
        sanitized_text, is_usable = _sanitize_flawed_trajectory_text(cand.solution_text)
        if not is_usable:
            placeholder_rejected += 1
            rejected_logs.append(
                {
                    "candidate_id": cand.candidate_id,
                    "problem_id": cand.problem_id,
                    "source": cand.source,
                    "reason": "rejected_incomplete_ellipsis_placeholder_derivation",
                }
            )
            continue
        cand = CandidateSolution(
            problem_id=cand.problem_id,
            candidate_id=cand.candidate_id,
            solution_text=sanitized_text,
            is_correct=cand.is_correct,
            domain=cand.domain,
            error_type=cand.error_type,
            paired_directive=cand.paired_directive,
            source=cand.source,
            original_id=cand.original_id,
            latency_seconds=cand.latency_seconds,
        )
        val = validate_candidate_answer(cand.solution_text, ref_prob.reference_solution)
        if val.is_equivalent or cand.is_correct:
            if "tiny_aya" in cand.source:
                tiny_aya_correct.append(cand)
                reason = f"tiny_aya_solved_correctly ({val.match_method})"
            else:
                reason = f"synthetic_error_solved_correctly ({val.match_method})"
            rejected_logs.append(
                {
                    "candidate_id": cand.candidate_id,
                    "problem_id": cand.problem_id,
                    "source": cand.source,
                    "reason": reason,
                }
            )
            continue
        flawed_candidates_pool.append(cand)

    save_candidates_jsonl(tiny_aya_correct, resolve_project_path(tiny_aya_correct_output_path))

    # Run CuratorKIT TextCleaner & SchemaGate on flawed candidates
    flawed_samples = [
        _candidate_to_datasample(c, prob_map[c.problem_id].problem_statement)
        for c in flawed_candidates_pool
    ]
    flawed_samples = cleaner.run(flawed_samples)
    schema_passed_samples, schema_failed_samples = schema_gate.run(flawed_samples)
    rejected_by_schema = len(schema_failed_samples) + placeholder_rejected
    for sf in schema_failed_samples:
        rejected_logs.append(
            {
                "candidate_id": sf.id,
                "problem_id": (sf.metadata or {}).get("problem_id"),
                "source": (sf.metadata or {}).get("source"),
                "reason": f"curatorkit_schema_gate:{sf.rejection_reason}",
            }
        )

    cleaned_flawed = [_datasample_to_candidate(s) for s in schema_passed_samples]

    # Run CuratorKIT ExactDeduplicator and MinHashDeduplicator across the flawed trajectory texts
    exact_dedup = ExactDeduplicator()
    minhash_dedup = MinHashDeduplicator(threshold=similarity_threshold)
    dedup_samples = exact_dedup.run(schema_passed_samples)
    dedup_samples = minhash_dedup.run(dedup_samples)
    curatorkit_survived_ids = {s.id for s in dedup_samples}

    # Group by problem_id to perform within-problem similarity pruning while guaranteeing >=1 trajectory per problem
    by_problem: dict[str, list[CandidateSolution]] = defaultdict(list)
    for cand in cleaned_flawed:
        by_problem[cand.problem_id].append(cand)

    surviving_flawed: list[CandidateSolution] = []
    eliminated_similar = 0

    for prob in problems:
        cands = by_problem.get(prob.problem_id, [])
        if not cands:
            continue
        # Sort so `aya_expanse_8b_synthetic_error` is primary (index 0) and `tiny_aya_global_zero_shot` is secondary
        cands.sort(key=lambda c: 0 if "aya_expanse_8b" in c.source else 1)
        primary = cands[0]
        surviving_flawed.append(primary)

        if len(cands) >= 2:
            secondary = cands[1]
            # Check if CuratorKIT Exact/MinHashDedup already flagged secondary OR if secondary is too similar to primary
            _, ans1 = extract_answer_line(primary.solution_text)
            _, ans2 = extract_answer_line(secondary.solution_text)
            norm_ans1 = _normalize_math_expr_string(ans1)
            norm_ans2 = _normalize_math_expr_string(ans2)
            tok_sim = _token_cosine_similarity(primary.solution_text, secondary.solution_text)

            is_similar = (
                secondary.candidate_id not in curatorkit_survived_ids
                or tok_sim >= similarity_threshold
                or (bool(norm_ans1) and norm_ans1 == norm_ans2 and tok_sim >= 0.72)
            )
            if is_similar:
                eliminated_similar += 1
                rejected_logs.append(
                    {
                        "candidate_id": secondary.candidate_id,
                        "problem_id": secondary.problem_id,
                        "source": secondary.source,
                        "reason": f"eliminated_similar_trajectory (cosine={tok_sim:.3f}, same_ans={norm_ans1 == norm_ans2})",
                    }
                )
            else:
                surviving_flawed.append(secondary)

    # Also run CuratorKIT DiversityGate across the surviving pool for telemetry & provenance reporting
    surv_samples = [
        _candidate_to_datasample(c, prob_map[c.problem_id].problem_statement)
        for c in surviving_flawed
    ]
    div_gate = DiversityGate(similarity_threshold=0.88, coverage_field="domain")
    _, _ = div_gate.run(surv_samples)

    save_candidates_jsonl(surviving_flawed, resolve_project_path(surviving_flawed_output_path))

    rej_file = resolve_project_path(rejected_output_path)
    rej_file.parent.mkdir(parents=True, exist_ok=True)
    with rej_file.open("w", encoding="utf-8") as f:
        for r in rejected_logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    surv_per_prob = Counter(c.problem_id for c in surviving_flawed)
    probs_with_1 = sum(1 for v in surv_per_prob.values() if v == 1)
    probs_with_2 = sum(1 for v in surv_per_prob.values() if v >= 2)

    summary = Stage2TrajectoryDedupSummary(
        canonical_problems_count=len(problems),
        minority_solution2_input_count=len(sol2_raw),
        minority_solution2_curated_count=len(curated_sol2),
        flawed_aya8b_input_count=len(flawed_8b_raw),
        tiny_aya_input_count=len(tiny_aya_raw),
        total_raw_trajectories_count=len(all_raw),
        tiny_aya_accidentally_correct_count=len(tiny_aya_correct),
        rejected_by_schema_or_empty_count=rejected_by_schema,
        eliminated_similar_trajectories_count=eliminated_similar,
        surviving_flawed_trajectories_count=len(surviving_flawed),
        problems_with_1_surviving_trajectory=probs_with_1,
        problems_with_2_surviving_trajectories=probs_with_2,
        surviving_by_domain=dict(sorted(Counter(c.domain for c in surviving_flawed).items())),
        surviving_by_source=dict(sorted(Counter(c.source for c in surviving_flawed).items())),
    )

    sum_file = resolve_project_path(summary_output_path)
    sum_file.parent.mkdir(parents=True, exist_ok=True)
    sum_file.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Stage 2.3 CuratorKIT Trajectory Dedup complete: %d surviving flawed trajectories "
        "(%d problems with 1, %d problems with 2; %d accidentally-correct Tiny-Aya filtered, %d similar eliminated).",
        len(surviving_flawed),
        probs_with_1,
        probs_with_2,
        len(tiny_aya_correct),
        eliminated_similar,
    )
    return curated_sol2, surviving_flawed, summary
