from pathlib import Path

import pytest
import yaml

from auric.config import fingerprint, formulation, load_config

BASE = {
    "experiment": {"name": "x", "seed": 0},
    "paths": {"split_file": "splits/s.json", "data_root": None, "work_dir": None},
    "runtime": {"device": 0, "workers": 2},
    "tiling": {"size": 1024, "overlap": 200},
    "inference": {"tile": 1024, "overlap": 200},
    "train": {"args": {"imgsz": 1024, "epochs": 50}},
}


def _setup(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "classmap.txt").write_text("0 A\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(yaml.safe_dump(BASE))
    (tmp_path / "configs" / "child.yaml").write_text(yaml.safe_dump({"_base_": "base.yaml",
                                                                      "train": {"args": {"epochs": 3}}}))
    (tmp_path / "env.yaml").write_text(yaml.safe_dump({"paths": {"data_root": "data", "work_dir": "work"},
                                                       "runtime": {"workers": 8}}))
    return tmp_path


def test_inheritance_paths_and_overrides(tmp_path):
    root = _setup(tmp_path)
    cfg = load_config(root / "configs" / "child.yaml", root / "env.yaml", overrides=["train.args.imgsz=640"])
    assert cfg["train"]["args"] == {"imgsz": 640, "epochs": 3}
    assert Path(cfg["paths"]["split_file"]) == (root / "configs" / "splits" / "s.json").resolve()
    assert Path(cfg["paths"]["data_root"]) == (root / "data").resolve()


def test_missing_paths_raise(tmp_path):
    root = _setup(tmp_path)
    with pytest.raises(ValueError, match="paths not set"):
        load_config(root / "configs" / "base.yaml")


def test_fingerprint_covers_only_the_given_sections(tmp_path):
    root = _setup(tmp_path)
    a = load_config(root / "configs" / "base.yaml", root / "env.yaml")
    b = load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["runtime.workers=1"])
    c = load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["train.args.epochs=1"])
    sections = ("experiment", "train")
    assert fingerprint(a, sections) == fingerprint(b, sections) != fingerprint(c, sections)


def test_a_new_optional_section_leaves_old_fingerprints_unchanged(tmp_path):
    root = _setup(tmp_path)
    a = load_config(root / "configs" / "base.yaml", root / "env.yaml")
    assert fingerprint(a, ("experiment", "train")) == fingerprint(a, ("experiment", "train", "formulation"))
    b = load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["formulation.mode=scenes"])
    assert fingerprint(b, ("experiment", "train", "formulation")) != fingerprint(a, ("experiment", "train", "formulation"))


def test_formulation_defaults_to_tiles_and_is_validated(tmp_path):
    root = _setup(tmp_path)
    assert formulation(load_config(root / "configs" / "base.yaml", root / "env.yaml")) == "tiles"
    with pytest.raises(ValueError, match="formulation.mode"):
        load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["formulation.mode=chips"])
    with pytest.raises(ValueError, match="cls_pw"):
        load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["train.args.cls_pw=1.5"])
    with pytest.raises(ValueError, match="alpha"):
        load_config(root / "configs" / "base.yaml", root / "env.yaml", overrides=["train.sampler={alpha: 2}"])


def test_shipped_configs_load_and_pre_existing_run_fingerprints_are_unchanged(tmp_path):
    from auric.runs import RUN_SECTIONS

    (tmp_path / "classmap.txt").write_text("0 A\n")
    configs = Path(__file__).resolve().parents[1] / "src" / "auric" / "configs"
    old = RUN_SECTIONS[:5]                      # the sections that identified a run before `formulation` and `subset` existed
    for path in sorted(configs.glob("*.yaml")):
        cfg = load_config(path, overrides=[f"paths.data_root={tmp_path.as_posix()}", "paths.work_dir=work"])
        if "formulation" not in cfg and "subset" not in cfg:
            assert fingerprint(cfg, old) == fingerprint(cfg, RUN_SECTIONS), path.name


def test_the_subset_section_is_validated_and_part_of_the_run_identity(tmp_path):
    from auric.runs import RUN_SECTIONS

    root = _setup(tmp_path)
    full = load_config(root / "configs" / "base.yaml", root / "env.yaml")
    sub = load_config(root / "configs" / "base.yaml", root / "env.yaml",
                      overrides=["subset={method: coverage, budget_frac: 0.25, seed: 1}"])
    assert fingerprint(sub, RUN_SECTIONS) != fingerprint(full, RUN_SECTIONS)
    assert fingerprint(full, RUN_SECTIONS) == fingerprint(full, RUN_SECTIONS[:6])          # absent section: old runs keep their id
    for bad, msg in (("subset.method=hard", "method"), ("subset.budget_frac=1.5", "budget_frac"), ("subset.budget_frac=0", "budget_frac")):
        with pytest.raises(ValueError, match=msg):
            load_config(root / "configs" / "base.yaml", root / "env.yaml",
                        overrides=["subset={method: random, budget_frac: 0.5}", bad])
