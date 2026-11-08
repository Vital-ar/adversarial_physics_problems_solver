"""Normalize raw PhysGym and textbook records into the CanonicalProblem schema."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from adversarial_physics.data.schemas.problem import CanonicalProblem

DEFAULT_DOMAIN_ALIASES: dict[str, str] = {
    "MECHANICS": "MECHANICS",
    "THERMODYNAMICS": "THERMODYNAMICS",
    "ELECTRICITY": "ELECTROMAGNETISM",
    "ELECTROMAGNETISM": "ELECTROMAGNETISM",
    "OPTICS": "OPTICS",
    "MODERN": "MODERN_PHYSICS",
    "MODERN_PHYSICS": "MODERN_PHYSICS",
    "ADVANCED": "RELATIVITY",
    "RELATIVITY": "RELATIVITY",
}


def normalize_domain(
    raw_domain: str,
    *,
    canonicalize: bool = False,
    aliases: dict[str, str] | None = None,
) -> str:
    """Normalize a domain tag, optionally mapping PhysGym tags to blueprint names."""
    cleaned = str(raw_domain).strip().upper()
    if not canonicalize:
        return cleaned
    mapping = aliases if aliases is not None else DEFAULT_DOMAIN_ALIASES
    return mapping.get(cleaned, cleaned)


def normalize_physgym_record(
    row: dict[str, Any],
    *,
    canonicalize_domains: bool = False,
    domain_aliases: dict[str, str] | None = None,
    id_prefix: str = "physgym",
) -> CanonicalProblem:
    """Convert a single PhysGym record into a CanonicalProblem instance."""
    raw_id = row.get("id", row.get("problem_id"))
    if raw_id is None:
        raise ValueError("PhysGym record is missing 'id' or 'problem_id'")

    raw_tag = row.get("tag", row.get("domain", "UNKNOWN"))
    domain = normalize_domain(
        raw_tag,
        canonicalize=canonicalize_domains,
        aliases=domain_aliases,
    )

    if isinstance(raw_id, int) or (isinstance(raw_id, str) and raw_id.isdigit()):
        problem_id = f"{id_prefix}_{int(raw_id):04d}" if id_prefix else str(raw_id)
    else:
        problem_id = str(raw_id)

    statement = str(row.get("content", row.get("problem_statement", ""))).strip()
    solution = str(row.get("solution", row.get("reference_solution", ""))).strip()

    return CanonicalProblem(
        problem_id=problem_id,
        domain=domain,
        problem_statement=statement,
        reference_solution=solution,
        source=str(row.get("source") or "physgym"),
        original_id=raw_id,
    )


def normalize_physgym_records(
    rows: Iterable[dict[str, Any]],
    *,
    canonicalize_domains: bool = False,
    domain_aliases: dict[str, str] | None = None,
    id_prefix: str = "physgym",
) -> list[CanonicalProblem]:
    """Normalize an iterable of PhysGym records into sorted CanonicalProblem objects."""
    problems = [
        normalize_physgym_record(
            dict(row),
            canonicalize_domains=canonicalize_domains,
            domain_aliases=domain_aliases,
            id_prefix=id_prefix,
        )
        for row in rows
    ]
    return sorted(problems, key=lambda p: p.problem_id)
