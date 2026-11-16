from pathlib import Path
import importlib.util
import sys

import yaml


def _load_card_module():
    path = Path("scripts/10_build_and_push_model_cards.py")
    spec = importlib.util.spec_from_file_location("build_model_cards", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["build_model_cards"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_all_15_model_cards_valid_and_concise() -> None:
    mod = _load_card_module()
    root = Path(".")
    if not (root / "results/comparison_tables/experiment_comparison_v1_v2_v3.json").is_file():
        return

    assert len(mod.ADAPTER_SPECS) == 15
    for spec in mod.ADAPTER_SPECS:
        card = mod.build_model_card(spec, root)
        lines = card.splitlines()
        assert 40 <= len(lines) <= 95
        assert "More Information Needed" not in card
        assert "Model Card for Model ID" not in card
        assert spec["dataset_repo"] in card
        assert "CohereLabs/tiny-aya-global" in card
        assert "Training Data & Provenance" in card

        # Verify clean YAML frontmatter parses properly
        parts = card.split("---\n", 2)
        assert len(parts) == 3
        meta = yaml.safe_load(parts[1])
        assert meta["base_model"] == "CohereLabs/tiny-aya-global"
        assert meta["library_name"] == "peft"
        assert meta["license"] == "cc-by-nc-4.0"
        assert spec["dataset_repo"] in meta["datasets"]
