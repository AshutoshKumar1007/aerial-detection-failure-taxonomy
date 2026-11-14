import pytest

from auric.runs import create_run, log_stage, open_run, read_record

CFG = {"experiment": {"name": "x", "seed": 0}, "split": {}, "tiling": {"size": 1024}, "model": {},
       "train": {"args": {"epochs": 1}}, "runtime": {"device": 0}}


def test_existing_run_requires_an_explicit_decision(tmp_path):
    d = tmp_path / "runs" / "x_s0"
    run = create_run(d, CFG)
    log_stage(run, "train", {"seed": 0})
    with pytest.raises(FileExistsError):
        create_run(d, CFG)
    assert create_run(d, CFG, if_exists="resume").dir == d
    other = {**CFG, "train": {"args": {"epochs": 2}}}
    with pytest.raises(RuntimeError, match="different config"):
        create_run(d, other, if_exists="resume")
    with pytest.raises(RuntimeError, match="different config"):
        open_run(d, other)
    open_run(d, {**CFG, "runtime": {"device": "cpu"}})      # runtime is not part of the run identity
    assert [e["stage"] for e in read_record(run)] == ["train"]
    create_run(d, other, if_exists="overwrite")
    assert read_record(run) == [] and open_run(d, other)


def test_import_checkpoint_checks_arguments_and_keeps_its_state(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from auric.model.train import import_checkpoint

    args = {"epochs": 50, "imgsz": 1024}
    src = tmp_path / "last.pt"
    torch.save({"epoch": 9, "train_args": {**args, "seed": 0, "data": "elsewhere"},
                "train_results": {"epoch": [1, 2], "train/box_loss": [2.3, 2.1]}}, src)
    with pytest.raises(ValueError, match="other arguments"):
        import_checkpoint(src, tmp_path / "a", {**args, "imgsz": 640}, seed=0)
    with pytest.raises(ValueError, match="other arguments"):
        import_checkpoint(src, tmp_path / "a", args, seed=1)
    info = import_checkpoint(src, tmp_path / "a", args, seed=0)
    assert (info.epochs_completed, info.epochs_planned, info.finished) == (10, 50, False)
    assert (tmp_path / "a" / "weights" / "last.pt").is_file() and (tmp_path / "a" / "args.yaml").is_file()
    assert (tmp_path / "a" / "results.csv").read_text().startswith("epoch,train/box_loss")
    with pytest.raises(FileExistsError):
        import_checkpoint(src, tmp_path / "a", args, seed=0)
