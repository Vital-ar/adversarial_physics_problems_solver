"""Offline leak checks, verifier/grader Cohen's kappa calibration, and compute/cost telemetry audits."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from adversarial_physics.evaluation.auditkit_bridge import (
    _safe_aligntune_grade,
    grade_math_answer,
)
from adversarial_physics.orchestration.loop import extract_problem_statement_from_prompt


def extract_question_text(record: dict[str, Any]) -> str:
    """Extract normalized problem statement from SFT or split JSONL records."""
    for key in ("problem_statement", "question", "prompt"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return extract_problem_statement_from_prompt(val)
    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return extract_problem_statement_from_prompt(content)
    return ""


def normalize_text_for_leak(text: str) -> str:
    """Canonicalize whitespace, case, and punctuation noise for leak comparisons."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return cleaned


def compute_word_shingles(text: str, n: int = 5) -> set[str]:
    """Compute word n-gram shingles for Jaccard near-duplicate leak checks."""
    tokens = re.findall(r"[a-z0-9\\]+", normalize_text_for_leak(text))
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute exact Jaccard similarity between two shingle sets."""
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}
_EMBEDDING_TEXT_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _get_sentence_transformer(model_name: str) -> Any:
    """Lazily load and cache SentenceTransformer, trying local cache first."""
    if model_name in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _encode_texts_normalized(
    texts: list[str],
    *,
    model_name: str,
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> np.ndarray:
    """Encode unique texts into L2-normalized vectors with process-level caching."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if embed_fn is not None:
        return _l2_normalize_rows(embed_fn(texts))

    unseen = sorted({t for t in texts if (model_name, t) not in _EMBEDDING_TEXT_CACHE})
    if unseen:
        model = _get_sentence_transformer(model_name)
        encoded = model.encode(
            unseen,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False,
        )
        normed = _l2_normalize_rows(encoded)
        for txt, vec in zip(unseen, normed):
            _EMBEDDING_TEXT_CACHE[(model_name, txt)] = vec
    return np.vstack([_EMBEDDING_TEXT_CACHE[(model_name, t)] for t in texts])


def check_embedding_train_eval_leakage(
    train_paths: list[Path],
    eval_paths: list[Path],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    cosine_threshold: float = 0.80,
    embed_fn: Callable[[list[str]], Any] | None = None,
    eval_rows_override: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Audit semantic train-eval leakage using dense cosine similarity (`cos >= threshold`)."""
    # Deduplicate identical normalized problem strings on train side to avoid redundant matmul rows
    train_unique: dict[str, tuple[str, str, str]] = {}
    train_rows_checked = 0
    for tp in train_paths:
        src_label = tp.parent.name if tp.name == "train.jsonl" else tp.name
        for row in _load_jsonl(tp):
            train_rows_checked += 1
            pid = str(row.get("problem_id", "")).strip()
            q = extract_question_text(row)
            if q and q not in train_unique:
                train_unique[q] = (src_label, pid, q)

    eval_items: list[tuple[str, str, str]] = []
    if eval_rows_override is not None:
        for split_label, row in eval_rows_override:
            pid = str(row.get("problem_id", "")).strip()
            q = extract_question_text(row)
            eval_items.append((split_label, pid, q))
    else:
        for ep in eval_paths:
            for row in _load_jsonl(ep):
                pid = str(row.get("problem_id", "")).strip()
                q = extract_question_text(row)
                eval_items.append((ep.name, pid, q))

    train_list = list(train_unique.values())
    leaked_pairs: list[dict[str, Any]] = []
    leaked_by_split: dict[str, list[str]] = {}
    clean_by_split: dict[str, list[str]] = {}
    max_cosine_overall = 0.0
    max_cosine_pair: dict[str, Any] | None = None
    max_clean_cosine = 0.0
    max_clean_pair: dict[str, Any] | None = None

    valid_eval_indices = [i for i, (_, _, q) in enumerate(eval_items) if q]
    if train_list and valid_eval_indices:
        t_emb = _encode_texts_normalized([x[2] for x in train_list], model_name=model_name, embed_fn=embed_fn)
        e_emb = _encode_texts_normalized(
            [eval_items[i][2] for i in valid_eval_indices],
            model_name=model_name,
            embed_fn=embed_fn,
        )
        sims = e_emb @ t_emb.T
        for row_idx, orig_idx in enumerate(valid_eval_indices):
            split_label, epid, _ = eval_items[orig_idx]
            best_j = int(np.argmax(sims[row_idx]))
            max_cos = float(sims[row_idx, best_j])
            t_src, t_pid, _ = train_list[best_j]
            pair_info = {
                "eval_split": split_label,
                "eval_problem_id": epid,
                "closest_train_source": t_src,
                "closest_train_problem_id": t_pid,
                "cosine": round(max_cos, 4),
            }
            if max_cos > max_cosine_overall:
                max_cosine_overall = max_cos
                max_cosine_pair = pair_info
            if max_cos >= cosine_threshold:
                leaked_pairs.append(pair_info)
                leaked_by_split.setdefault(split_label, []).append(epid)
            else:
                clean_by_split.setdefault(split_label, []).append(epid)
                if max_cos > max_clean_cosine:
                    max_clean_cosine = max_cos
                    max_clean_pair = pair_info

    for split_label, epid, q in eval_items:
        if not q:
            clean_by_split.setdefault(split_label, []).append(epid)

    leaked_pairs.sort(key=lambda d: (-float(d["cosine"]), str(d["eval_split"]), str(d["eval_problem_id"])))
    return {
        "train_files": [str(p).replace("\\", "/") for p in train_paths],
        "eval_files": [str(p).replace("\\", "/") for p in eval_paths],
        "embedding_model": model_name,
        "cosine_threshold": cosine_threshold,
        "train_rows_checked": train_rows_checked,
        "train_unique_texts": len(train_list),
        "eval_rows_checked": len(eval_items),
        "leaked_eval_count": len(leaked_pairs),
        "leaked_by_split": {k: sorted(set(v)) for k, v in sorted(leaked_by_split.items())},
        "clean_by_split": {k: sorted(set(v)) for k, v in sorted(clean_by_split.items())},
        "leaked_pairs": leaked_pairs,
        "max_cosine_overall": round(max_cosine_overall, 4),
        "max_cosine_pair": max_cosine_pair,
        "max_clean_cosine": round(max_clean_cosine, 4),
        "max_clean_pair": max_clean_pair,
        "leak_free": len(leaked_pairs) == 0,
    }


def check_train_eval_leakage(
    train_paths: list[Path],
    eval_paths: list[Path],
    *,
    shingle_n: int = 5,
    jaccard_threshold: float = 0.75,
    embedding_cosine_threshold: float | None = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Check ID overlap, exact normalized question matches, and word n-gram near-duplicates."""
    train_records: dict[str, tuple[str, str, set[str]]] = {}
    for tp in train_paths:
        for row in _load_jsonl(tp):
            pid = str(row.get("problem_id", "")).strip()
            q = extract_question_text(row)
            norm = normalize_text_for_leak(q)
            key = f"{tp.parent.name}:{pid}:{hash(norm)}"
            if norm:
                train_records[key] = (pid, norm, compute_word_shingles(q, n=shingle_n))

    train_pids = {v[0] for v in train_records.values() if v[0]}
    train_norms = {v[1] for v in train_records.values() if v[1]}
    train_shingle_list = [(k, v[0], v[2]) for k, v in train_records.items()]

    eval_total = 0
    eval_unique_pids: set[str] = set()
    id_overlaps: list[str] = []
    exact_matches: list[str] = []
    near_duplicates: list[dict[str, Any]] = []
    max_jaccard_overall = 0.0
    max_jaccard_pair: dict[str, Any] | None = None

    for ep in eval_paths:
        for row in _load_jsonl(ep):
            eval_total += 1
            pid = str(row.get("problem_id", "")).strip()
            eval_unique_pids.add(pid)
            q = extract_question_text(row)
            norm = normalize_text_for_leak(q)
            if pid and pid in train_pids:
                id_overlaps.append(pid)
            if norm and norm in train_norms:
                exact_matches.append(pid)
            e_shingles = compute_word_shingles(q, n=shingle_n)
            for _, t_pid, t_shingles in train_shingle_list:
                sim = jaccard_similarity(e_shingles, t_shingles)
                if sim > max_jaccard_overall:
                    max_jaccard_overall = sim
                    max_jaccard_pair = {
                        "eval_problem_id": pid,
                        "train_problem_id": t_pid,
                        "jaccard": round(sim, 4),
                    }
                if sim >= jaccard_threshold:
                    near_duplicates.append(
                        {
                            "eval_problem_id": pid,
                            "train_problem_id": t_pid,
                            "jaccard": round(sim, 4),
                        }
                    )

    unique_id_overlaps = sorted(set(id_overlaps))
    unique_exact_matches = sorted(set(exact_matches))
    result: dict[str, Any] = {
        "train_files": [str(p).replace("\\", "/") for p in train_paths],
        "eval_files": [str(p).replace("\\", "/") for p in eval_paths],
        "train_unique_problem_ids": len(train_pids),
        "train_rows_checked": len(train_records),
        "eval_rows_checked": eval_total,
        "eval_unique_problem_ids": len(eval_unique_pids),
        "shingle_n": shingle_n,
        "jaccard_threshold": jaccard_threshold,
        "id_overlap_count": len(unique_id_overlaps),
        "id_overlaps": unique_id_overlaps,
        "exact_normalized_text_overlap_count": len(unique_exact_matches),
        "exact_text_overlaps": unique_exact_matches,
        "near_duplicate_pair_count": len(near_duplicates),
        "near_duplicates": near_duplicates[:20],
        "max_shingle_jaccard": round(max_jaccard_overall, 4),
        "max_shingle_pair": max_jaccard_pair,
        "leak_free": (
            len(unique_id_overlaps) == 0
            and len(unique_exact_matches) == 0
            and len(near_duplicates) == 0
        ),
    }
    if embedding_cosine_threshold is not None:
        emb_audit = check_embedding_train_eval_leakage(
            train_paths,
            eval_paths,
            model_name=embedding_model_name,
            cosine_threshold=embedding_cosine_threshold,
            embed_fn=embed_fn,
        )
        result["embedding_leak_audit"] = emb_audit
        result["leak_free"] = bool(result["leak_free"] and emb_audit["leak_free"])
    return result


def compute_confusion_kappa(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Compute accuracy, chance-corrected Cohen's kappa, precision, recall, and F1."""
    n = tp + fp + tn + fn
    if n <= 0:
        return {
            "n": 0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "observed_agreement": None,
            "expected_agreement": None,
            "cohens_kappa": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    po = (tp + tn) / n
    pred_pos = tp + fp
    pred_neg = tn + fn
    true_pos = tp + fn
    true_neg = tn + fp
    pe = ((pred_pos * true_pos) + (pred_neg * true_neg)) / (n * n)
    kappa = (po - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else 0.0
    precision = tp / pred_pos if pred_pos > 0 else None
    recall = tp / true_pos if true_pos > 0 else None
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else 0.0
    )
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4),
    }


def audit_verifier_from_loop_summary(summary_path: Path, label: str) -> dict[str, Any]:
    """Compute Verifier agreement and Cohen's kappa from a saved loop summary JSON."""
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    ver = payload.get("metrics", {}).get("verifier") or payload.get("verifier", {})
    tp = int(ver.get("tp_true_challenge", 0))
    fp = int(ver.get("fp_false_challenge", 0))
    tn = int(ver.get("tn_true_approve", 0))
    fn = int(ver.get("fn_false_approve", 0))
    unknown = int(ver.get("unknown_no_sentinel", 0))
    valid_stats = compute_confusion_kappa(tp=tp, fp=fp, tn=tn, fn=fn)
    total_reviews = tp + fp + tn + fn + unknown
    return {
        "label": label,
        "source": str(summary_path).replace("\\", "/"),
        "total_reviews": total_reviews,
        "unknown_no_sentinel": unknown,
        "overall_sentinel_accuracy": round((tp + tn) / total_reviews, 4) if total_reviews else None,
        "valid_sentinel_stats": valid_stats,
        "false_approval_rate": ver.get("verifier_false_approval_rate"),
        "false_challenge_rate": ver.get("verifier_false_challenge_rate"),
        "directive_actionability": ver.get("auditkit_directive_actionability_score"),
        "directive_no_leak_rate": ver.get("directive_self_containment_no_leak_rate"),
    }


def audit_grader_agreement_from_eval_jsons(eval_paths: list[Path]) -> dict[str, Any]:
    """Measure agreement and Cohen's kappa between SymPy/Exact physics grading and AlignTune math grader."""
    at_available = grade_math_answer is not None
    tp = fp = tn = fn = 0
    evaluated_files: list[str] = []
    for path in eval_paths:
        if not path.is_file():
            continue
        evaluated_files.append(str(path).replace("\\", "/"))
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in data.get("per_sample", []):
            sympy_ok = bool(row.get("is_equivalent", False))
            pred_ans = str(row.get("predicted_answer") or "").strip()
            ref_ans = str(row.get("reference_answer") or "").strip()
            at_ok = bool(_safe_aligntune_grade(pred_ans, ref_ans)) if (at_available and pred_ans and ref_ans) else False
            if sympy_ok and at_ok:
                tp += 1
            elif (not sympy_ok) and at_ok:
                fp += 1
            elif (not sympy_ok) and (not at_ok):
                tn += 1
            else:
                fn += 1
    return {
        "files_audited": evaluated_files,
        "aligntune_math_grader_available": at_available,
        "agreement_stats": compute_confusion_kappa(tp=tp, fp=fp, tn=tn, fn=fn),
    }


def summarize_training_and_inference_compute(root_dir: Path) -> dict[str, Any]:
    """Aggregate training runtime across v1/v2/v3 adapters and loop inference throughput metrics."""
    by_version: dict[str, dict[str, Any]] = {
        "v1": {"runs": [], "total_seconds": 0.0, "total_steps": 0},
        "v2": {"runs": [], "total_seconds": 0.0, "total_steps": 0},
        "v3": {"runs": [], "total_seconds": 0.0, "total_steps": 0},
    }
    adapters_root = root_dir / "models/adapters"
    if adapters_root.is_dir():
        for ts_path in sorted(adapters_root.glob("*/training_summary.json")):
            adapter_name = ts_path.parent.name
            ver = "v3" if "_v3_" in adapter_name else ("v2" if "_v2_" in adapter_name else "v1")
            ts = json.loads(ts_path.read_text(encoding="utf-8"))
            tm = ts.get("train_metrics", {})
            secs = float(tm.get("training_time") or tm.get("metrics", {}).get("train_runtime") or 0.0)
            steps = int(tm.get("steps") or 0)
            by_version[ver]["runs"].append(
                {
                    "adapter": adapter_name,
                    "experiment_id": ts.get("experiment_id"),
                    "lora_r": ts.get("lora_r"),
                    "lora_alpha": ts.get("lora_alpha"),
                    "learning_rate": ts.get("learning_rate"),
                    "training_seconds": round(secs, 2),
                    "training_hours": round(secs / 3600.0, 4),
                    "steps": steps,
                    "epochs_run": tm.get("metrics", {}).get("epoch"),
                    "final_loss": round(float(tm.get("final_loss") or 0.0), 4),
                }
            )
            by_version[ver]["total_seconds"] = round(by_version[ver]["total_seconds"] + secs, 2)
            by_version[ver]["total_steps"] += steps

    total_train_seconds = sum(v["total_seconds"] for v in by_version.values())
    for v in by_version.values():
        v["total_hours"] = round(v["total_seconds"] / 3600.0, 4)

    inference_loops: list[dict[str, Any]] = []
    for label, p in [
        ("v3_val_60", root_dir / "results/eval_e4_loop_v3_val/summary.json"),
        ("v3_test_60", root_dir / "results/eval_e4_loop_v3_test/summary.json"),
    ]:
        if p.is_file():
            s = json.loads(p.read_text(encoding="utf-8"))
            eff = s.get("metrics", {}).get("loop_efficiency", {})
            n = int(s.get("metrics", {}).get("total_problems", 0))
            mean_lat = float(eff.get("mean_latency_s") or 0.0)
            mean_tok = float(eff.get("mean_tokens_per_problem") or 0.0)
            tok_per_s = mean_tok / mean_lat if mean_lat > 0 else 0.0
            gpu_hours_per_1k = (mean_lat * 1000.0) / 3600.0 if mean_lat > 0 else 0.0
            inference_loops.append(
                {
                    "label": label,
                    "problems": n,
                    "mean_iterations": eff.get("mean_iterations_used"),
                    "mean_latency_s": round(mean_lat, 2),
                    "total_latency_s": round(float(eff.get("total_latency_s") or 0.0), 2),
                    "mean_tokens_per_problem": round(mean_tok, 2),
                    "effective_tokens_per_sec": round(tok_per_s, 2),
                    "gpu_hours_per_1k_loops": round(gpu_hours_per_1k, 2),
                }
            )

    return {
        "training_by_version": by_version,
        "total_training_seconds": round(total_train_seconds, 2),
        "total_training_hours": round(total_train_seconds / 3600.0, 4),
        "inference_loop_telemetry": inference_loops,
    }
