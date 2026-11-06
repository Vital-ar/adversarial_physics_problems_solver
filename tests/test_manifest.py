from pathlib import Path

import yaml

from adversarial_physics.utils.manifest import _write_yaml_atomic, update_experiment_manifest


def test_update_manifest_records_status_and_revision(tmp_path: Path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump({"status": "prepared", "output": {}, "evaluation": {}}),
        encoding="utf-8",
    )
    updated = update_experiment_manifest(
        path,
        status="evaluated",
        output_hf_revision="0123456789abcdef",
        metrics_file="experiments/E2_solver_curated/metrics.json",
    )
    assert updated["status"] == "evaluated"
    assert updated["output"]["hf_revision"] == "0123456789abcdef"
    assert updated["evaluation"]["metrics_file"].endswith("metrics.json")


class LibraryString(str):
    """Stand-in for str subclasses such as torch.torch_version.TorchVersion."""


def test_manifest_writer_normalizes_string_subclasses(tmp_path: Path):
    path = tmp_path / "manifest.yaml"
    _write_yaml_atomic(
        path,
        {
            "runtime": {
                "torch_version": LibraryString("2.13.0+cpu"),
                "packages": [LibraryString("torch"), "PyYAML"],
            }
        },
    )

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["runtime"]["torch_version"] == "2.13.0+cpu"
    assert type(loaded["runtime"]["torch_version"]) is str
    assert loaded["runtime"]["packages"] == ["torch", "PyYAML"]
