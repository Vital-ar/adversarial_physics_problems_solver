"""Substage 1.3: Problem-level and Anti-Leakage Deduplication (`cosine_similarity >= 0.85`).

Provides two distinct deduplication procedures required by the blueprint:
1. **Anti-Leakage Gate (`run_anti_leakage_gate`)**:
   Compares every candidate textbook problem against all 76 frozen held-out PhysGym problems
   (38 Validation + 38 Test) using:
     * Exact normalized statement hash
     * Word 5-gram Jaccard & containment overlap
     * Sublinear TF-IDF / dense embedding cosine similarity with explicit default
       ``cosine_threshold = 0.85`` (overriding CuratorKIT's 0.92 default).
2. **Within-Pool Problem Deduplication (`deduplicate_candidate_pool`)**:
   Ensures no two problems admitted into ``canonical_train_250.jsonl`` are exact or near-duplicates
   of one another, while keeping related subproblems from the same ``original_id`` tracked.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from adversarial_physics.data.curation.curatorkit_stage1 import (
    run_curatorkit_embedding_dedup,
    run_curatorkit_within_pool_dedup,
)
from adversarial_physics.data.schemas.problem import CanonicalProblem
from adversarial_physics.utils.logging import get_logger

logger = get_logger("stage_1_3_deduplication")

DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD: float = 0.85
DEFAULT_ANTI_LEAKAGE_NGRAM_CONTAINMENT_THRESHOLD: float = 0.60
DEFAULT_WITHIN_POOL_COSINE_THRESHOLD: float = 0.88

_LATEX_NOISE_RE = re.compile(
    r"(\\mathrm|\\mathbf|\\pmb|\\boldsymbol|\\textstyle|\\displaystyle|\\scriptstyle|\\left|\\right|\\biggl|\\biggr|\\Big|\\bigl|\\bigr)"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class LeakageMatch:
    """Details of a candidate problem rejected by the anti-leakage or deduplication gate."""

    candidate_id: str
    candidate_original_id: Any
    matched_problem_id: str
    matched_split: str
    cosine_similarity: float
    ngram_containment: float
    exact_hash_match: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text_for_dedup(text: str) -> str:
    """Normalize physics problem text for hashing and token n-gram comparison."""
    cleaned = _LATEX_NOISE_RE.sub(" ", text.lower())
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_normalized_hash(text: str) -> str:
    """Compute SHA-256 hex digest of normalized problem statement text."""
    norm = normalize_text_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def tokenize_for_dedup(text: str) -> list[str]:
    """Tokenize problem statement into lowercase alphanumeric tokens."""
    norm = _LATEX_NOISE_RE.sub(" ", text.lower())
    return _TOKEN_RE.findall(norm)


def extract_word_ngrams(tokens: list[str], n: int = 5) -> set[tuple[str, ...]]:
    """Extract unique word n-grams from a token sequence."""
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_containment(tokens_a: list[str], tokens_b: list[str], n: int = 5) -> float:
    """Compute maximum directional n-gram containment between two token sequences."""
    ngrams_a = extract_word_ngrams(tokens_a, n=n)
    ngrams_b = extract_word_ngrams(tokens_b, n=n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    inter = len(ngrams_a & ngrams_b)
    denom = min(len(ngrams_a), len(ngrams_b))
    return inter / denom if denom > 0 else 0.0


def _build_tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """Build L2-normalized sublinear TF-IDF vectors combining word unigrams/bigrams and char 4-grams."""
    doc_term_counts: list[Counter[str]] = []
    df: Counter[str] = Counter()
    n_docs = len(texts)

    for text in texts:
        tokens = tokenize_for_dedup(text)
        counts: Counter[str] = Counter()
        for tok in tokens:
            if len(tok) >= 2:
                counts[f"w1:{tok}"] += 1
        for i in range(len(tokens) - 1):
            counts[f"w2:{tokens[i]}_{tokens[i + 1]}"] += 1
        norm_compact = "".join(tokens)
        for i in range(0, max(0, len(norm_compact) - 4), 2):
            counts[f"c4:{norm_compact[i : i + 4]}"] += 1

        doc_term_counts.append(counts)
        for term in counts.keys():
            df[term] += 1

    vectors: list[dict[str, float]] = []
    for counts in doc_term_counts:
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for term, tf in counts.items():
            # Smooth IDF with sublinear TF
            idf = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
            weight = (1.0 + math.log(tf)) * idf
            vec[term] = weight
            norm_sq += weight * weight
        norm = math.sqrt(norm_sq) if norm_sq > 0.0 else 1.0
        for term in vec:
            vec[term] /= norm
        vectors.append(vec)
    return vectors


def _sparse_cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute dot product between two L2-normalized sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(val * vec_b.get(term, 0.0) for term, val in vec_a.items())


def run_anti_leakage_gate(
    candidates: list[CanonicalProblem],
    *,
    val_problems: list[CanonicalProblem],
    test_problems: list[CanonicalProblem],
    cosine_threshold: float = DEFAULT_ANTI_LEAKAGE_COSINE_THRESHOLD,
    ngram_containment_threshold: float = DEFAULT_ANTI_LEAKAGE_NGRAM_CONTAINMENT_THRESHOLD,
    ngram_size: int = 5,
) -> tuple[list[CanonicalProblem], list[LeakageMatch]]:
    """Filter candidate textbook problems against all held-out Validation and Test problems.

    Any candidate with:
      * exact normalized hash match,
      * cosine similarity >= ``cosine_threshold`` (default 0.85), or
      * 5-gram containment >= ``ngram_containment_threshold`` (default 0.60)
    to any held-out Validation or Test problem is purged immediately.
    """
    if not candidates:
        return [], []

    heldout: list[tuple[CanonicalProblem, str]] = [
        *((p, "validation") for p in val_problems),
        *((p, "test") for p in test_problems),
    ]
    if not heldout:
        return list(candidates), []

    # 1. Run CuratorKIT persistent cross-run EmbeddingDeduplicator (configured at threshold=0.85)
    _, ck_purged = run_curatorkit_embedding_dedup(
        candidates, heldout, cosine_threshold=cosine_threshold
    )
    ck_purged_ids: dict[str, dict[str, Any]] = {
        str(item["candidate_id"]): item for item in ck_purged
    }

    all_texts = [p.problem_statement for p, _ in heldout] + [c.problem_statement for c in candidates]
    all_vecs = _build_tfidf_vectors(all_texts)
    heldout_vecs = all_vecs[: len(heldout)]
    cand_vecs = all_vecs[len(heldout) :]

    heldout_hashes = [compute_normalized_hash(p.problem_statement) for p, _ in heldout]
    heldout_tokens = [tokenize_for_dedup(p.problem_statement) for p, _ in heldout]

    accepted: list[CanonicalProblem] = []
    leaked: list[LeakageMatch] = []

    for cand, cand_vec in zip(candidates, cand_vecs):
        cand_hash = compute_normalized_hash(cand.problem_statement)
        cand_tokens = tokenize_for_dedup(cand.problem_statement)

        best_cosine = -1.0
        best_containment = -1.0
        best_idx = 0
        exact_match = False

        for idx, ((hp, _split_name), h_vec, h_hash, h_tokens) in enumerate(
            zip(heldout, heldout_vecs, heldout_hashes, heldout_tokens)
        ):
            if cand_hash == h_hash:
                exact_match = True
                best_cosine = 1.0
                best_containment = 1.0
                best_idx = idx
                break

            cos_sim = _sparse_cosine(cand_vec, h_vec)
            cont_sim = compute_ngram_containment(cand_tokens, h_tokens, n=ngram_size)
            if (cos_sim, cont_sim) > (best_cosine, best_containment):
                best_cosine = cos_sim
                best_containment = cont_sim
                best_idx = idx

        matched_prob, matched_split = heldout[best_idx]
        if (
            cand.problem_id in ck_purged_ids
            or exact_match
            or best_cosine >= cosine_threshold
            or best_containment >= ngram_containment_threshold
        ):
            reason = (
                "exact_heldout_duplicate"
                if exact_match
                else (
                    f"curatorkit_embedding_dedup>={cosine_threshold:.2f}"
                    if cand.problem_id in ck_purged_ids
                    else (
                        f"anti_leakage_cosine>={cosine_threshold:.2f} ({best_cosine:.3f})"
                        if best_cosine >= cosine_threshold
                        else f"anti_leakage_{ngram_size}gram>={ngram_containment_threshold:.2f} ({best_containment:.3f})"
                    )
                )
            )
            leaked.append(
                LeakageMatch(
                    candidate_id=cand.problem_id,
                    candidate_original_id=cand.original_id,
                    matched_problem_id=matched_prob.problem_id,
                    matched_split=matched_split,
                    cosine_similarity=round(best_cosine, 4),
                    ngram_containment=round(best_containment, 4),
                    exact_hash_match=exact_match,
                    reason=reason,
                )
            )
        else:
            accepted.append(cand)

    logger.info(
        "Anti-leakage gate checked %d candidates against %d held-out Val/Test problems: %d accepted, %d purged.",
        len(candidates),
        len(heldout),
        len(accepted),
        len(leaked),
    )
    return accepted, leaked


def deduplicate_candidate_pool(
    candidates: list[CanonicalProblem],
    *,
    seed_problems: list[CanonicalProblem] | None = None,
    cosine_threshold: float = DEFAULT_WITHIN_POOL_COSINE_THRESHOLD,
    ngram_containment_threshold: float = 0.70,
    ngram_size: int = 5,
) -> tuple[list[CanonicalProblem], list[LeakageMatch]]:
    """Deduplicate candidate problems against existing training seeds and each other.

    Sibling subproblems split from the same parent exercise (sharing the same base
    ``original_id`` prefix before ``:part_``) are exempted from cross-subproblem
    rejection so independently solvable ``(a)`` and ``(b)`` subproblems can coexist
    in the training split.
    """
    if not candidates:
        return [], []

    seeds = list(seed_problems or [])
    _, ck_pool_purged = run_curatorkit_within_pool_dedup(
        candidates,
        seed_problems=seeds,
        minhash_threshold=0.85,
        diversity_threshold=cosine_threshold,
    )
    ck_pool_purged_map: dict[str, dict[str, Any]] = {
        str(item["candidate_id"]): item for item in ck_pool_purged
    }

    all_texts = [p.problem_statement for p in seeds] + [c.problem_statement for c in candidates]
    all_vecs = _build_tfidf_vectors(all_texts)
    seed_vecs = all_vecs[: len(seeds)]
    cand_vecs = all_vecs[len(seeds) :]

    accepted_probs: list[CanonicalProblem] = list(seeds)
    accepted_vecs: list[dict[str, float]] = list(seed_vecs)
    accepted_hashes: list[str] = [compute_normalized_hash(p.problem_statement) for p in seeds]
    accepted_tokens: list[list[str]] = [tokenize_for_dedup(p.problem_statement) for p in seeds]

    unique_candidates: list[CanonicalProblem] = []
    duplicates: list[LeakageMatch] = []

    def _base_parent_id(orig_id: Any) -> str:
        s = str(orig_id or "")
        return s.split(":part_")[0]

    for cand, cand_vec in zip(candidates, cand_vecs):
        cand_hash = compute_normalized_hash(cand.problem_statement)
        cand_tokens = tokenize_for_dedup(cand.problem_statement)
        cand_parent = _base_parent_id(cand.original_id)

        dup_match: LeakageMatch | None = None
        if cand.problem_id in ck_pool_purged_map:
            ck_item = ck_pool_purged_map[cand.problem_id]
            dup_match = LeakageMatch(
                candidate_id=cand.problem_id,
                candidate_original_id=cand.original_id,
                matched_problem_id="curatorkit_pool",
                matched_split="train_pool",
                cosine_similarity=1.0,
                ngram_containment=1.0,
                exact_hash_match="exact" in str(ck_item.get("reason", "")),
                reason=str(ck_item.get("reason") or "curatorkit_within_pool_duplicate"),
            )

        for existing_prob, existing_vec, existing_hash, existing_tokens in zip(
            accepted_probs, accepted_vecs, accepted_hashes, accepted_tokens
        ):
            if dup_match is not None:
                break
            if cand_hash == existing_hash:
                dup_match = LeakageMatch(
                    candidate_id=cand.problem_id,
                    candidate_original_id=cand.original_id,
                    matched_problem_id=existing_prob.problem_id,
                    matched_split="train_pool",
                    cosine_similarity=1.0,
                    ngram_containment=1.0,
                    exact_hash_match=True,
                    reason="within_pool_exact_duplicate",
                )
                break

            # Allow sibling subproblems from the same parent exercise unless identical
            if cand_parent and cand_parent == _base_parent_id(existing_prob.original_id):
                continue

            cos_sim = _sparse_cosine(cand_vec, existing_vec)
            cont_sim = compute_ngram_containment(cand_tokens, existing_tokens, n=ngram_size)
            if cos_sim >= cosine_threshold or cont_sim >= ngram_containment_threshold:
                dup_match = LeakageMatch(
                    candidate_id=cand.problem_id,
                    candidate_original_id=cand.original_id,
                    matched_problem_id=existing_prob.problem_id,
                    matched_split="train_pool",
                    cosine_similarity=round(cos_sim, 4),
                    ngram_containment=round(cont_sim, 4),
                    exact_hash_match=False,
                    reason=f"within_pool_near_duplicate (cos={cos_sim:.3f}, ngram={cont_sim:.3f})",
                )
                break

        if dup_match is not None:
            duplicates.append(dup_match)
        else:
            unique_candidates.append(cand)
            accepted_probs.append(cand)
            accepted_vecs.append(cand_vec)
            accepted_hashes.append(cand_hash)
            accepted_tokens.append(cand_tokens)

    return unique_candidates, duplicates
