import csv
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from rsfm.preprocessing import cli, status, worker
from rsfm.preprocessing.inventory import build_manifest, load_manifest, save_manifest, update_manifest
from rsfm.preprocessing.split import make_split, read_split, write_split
from rsfm.preprocessing.voxel_store import VoxelRun, build_group_mask, load_grid, save_grid
from test_preprocessing import AFFINE, CONF_CFG, SHAPE, make_run

CLASSES = {**{f"{i:06d}": "complete" for i in range(900)}, **{f"9{i:05d}": "partial" for i in range(80)},
           **{f"8{i:05d}": "unusable" for i in range(5)}}
BOLD_GLOB = "sub-*/ses-*/func/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
FMRIPREP_COLUMNS = ["subject", "class", "runs", "collect", "state"]


# ---------------------------------------------------------------- split
def test_split_sizes_and_membership():
    split = make_split(CLASSES)
    assert len(split) == 980 and not any(CLASSES[s] == "unusable" for s in split)
    assert sorted(set(split.values())) == ["test", "train", "val"]
    for cls, n in (("complete", 900), ("partial", 80)):
        got = [split[s] for s in split if CLASSES[s] == cls]
        assert got.count("train") == round(0.8 * n) and got.count("val") == round(0.1 * n)
        assert got.count("test") == n - round(0.8 * n) - round(0.1 * n)


def test_split_is_reproducible_and_depends_on_seed():
    assert make_split(CLASSES, seed=3) == make_split(CLASSES, seed=3)
    assert make_split(CLASSES, seed=3) != make_split(CLASSES, seed=4)


def test_split_files_and_no_overwrite(tmp_path):
    split = make_split(CLASSES)
    table = write_split(split, CLASSES, tmp_path, "v1")
    assert read_split(table) == split
    lists = {name: (tmp_path / f"v1_{name}.txt").read_text().split() for name in ("train", "val", "test")}
    assert sum(len(x) for x in lists.values()) == 980
    assert not set(lists["train"]) & set(lists["test"]) and not set(lists["val"]) & set(lists["test"])
    with pytest.raises(FileExistsError):
        write_split(split, CLASSES, tmp_path, "v1")


# ---------------------------------------------------------------- pipeline fixtures
def make_subject(root, sub, dirs=("LR", "RL"), seed=0):
    for k, d in enumerate(dirs):
        make_run(root / f"sub-{sub}" / "ses-1" / "func", f"sub-{sub}_ses-1_task-rest_dir-{d}", seed=seed + k)


def make_cfg(tmp_path):
    return {"fmriprep_dir": tmp_path / "fmriprep", "output_dir": tmp_path / "out", "bold_glob": BOLD_GLOB,
            "qc": {"fd_threshold": 0.5, "max_mean_fd": 0.5}, "confounds": CONF_CFG, "io": {"chunk_trs": 50},
            "denoise": {"low_pass": 0.1, "detrend": True, "standardize": "zscore_sample"}}


def write_fmriprep_status(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, FMRIPREP_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def paths(tmp_path):
    write_fmriprep_status(tmp_path / "fmriprep.tsv", [
        {"subject": "01", "class": "complete", "runs": "A,B", "collect": "DONE", "state": "DONE"},
        {"subject": "02", "class": "partial", "runs": "A", "collect": "RUN", "state": "RUNNING"},
        {"subject": "03", "class": "complete", "runs": "A,B", "collect": "-", "state": "DONE"},
        {"subject": "04", "class": "complete", "runs": "A,B", "collect": "DONE", "state": "DONE"},
        {"subject": "09", "class": "unusable", "runs": "", "collect": "SKIP", "state": "SKIPPED(unusable)"},
    ])
    (tmp_path / "split.tsv").write_text("subject\tclass\tset\n01\tcomplete\ttrain\n02\tpartial\tval\n"
                                        "03\tcomplete\ttest\n04\tcomplete\ttrain\n")
    p = status.Paths(tmp_path / "jobs", tmp_path / "fmriprep.tsv", tmp_path / "split.tsv", tmp_path / "STATUS.md")
    status.main(["--dir", str(p.dir), "--fmriprep-status", str(p.fmriprep_status), "--split", str(p.split),
                 "--md", str(p.md), "init"])
    return p


# ---------------------------------------------------------------- manifest and mask
def test_manifest_for_one_subject_and_merge(tmp_path):
    make_subject(tmp_path / "fmriprep", "01")
    make_subject(tmp_path / "fmriprep", "02", dirs=("LR",))
    cfg = make_cfg(tmp_path)
    one = build_manifest(cfg, subjects=["01"])
    assert one["sub"].tolist() == ["01", "01"]

    update_manifest(one, cfg["output_dir"])
    update_manifest(build_manifest(cfg, subjects=["02"]), cfg["output_dir"])
    update_manifest(build_manifest(cfg, subjects=["01"]), cfg["output_dir"])       # a rerun adds nothing
    merged = load_manifest(cfg["output_dir"])
    assert merged["sub"].tolist() == ["01", "01", "02"] and merged["run_id"].is_unique


def test_mask_from_a_subject_list(tmp_path, monkeypatch):
    make_subject(tmp_path / "fmriprep", "01")
    make_subject(tmp_path / "fmriprep", "02")
    cfg = make_cfg(tmp_path)
    save_manifest(build_manifest(cfg), cfg["output_dir"])
    # Subject 02's masks are wider, so a mask that used them would be larger.
    wide = np.ones(SHAPE, np.uint8)
    for path in (tmp_path / "fmriprep" / "sub-02").rglob("*_desc-brain_mask.nii.gz"):
        nib.save(nib.Nifti1Image(wide, AFFINE), path)
    template = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(wide, AFFINE), template)
    listed = tmp_path / "train.txt"
    listed.write_text("01\n")
    monkeypatch.setattr(cli, "load_config", lambda path, out=None: {
        **cfg, "template_mask": template, "group_mask": {"min_run_fraction": 0.5}})
    cli.main(["mask", "--config", "x", "--subjects-file", str(listed)])
    assert load_grid(cfg["output_dir"]).n_voxels == 36        # subject 01's 4 x 3 x 3 mask, not all 120
    with pytest.raises(FileExistsError):                      # frozen
        cli.main(["mask", "--config", "x"])


# ---------------------------------------------------------------- status
def test_init_reads_fmriprep_and_split(paths):
    rows = status.read_table(paths)
    assert rows["01"]["set"] == "train" and rows["01"]["expected_runs"] == "2" and rows["01"]["fmriprep"] == "DONE"
    assert rows["09"]["state"] == "SKIPPED(unusable)" and rows["09"]["set"] == "-"
    assert rows["03"]["fmriprep"] == "DONE(not collected)"
    assert paths.md.exists()


def test_only_finished_and_collected_subjects_are_ready(paths):
    assert status.next_subjects(status.read_table(paths)) == ["01", "04"]


def test_never_writes_the_fmriprep_status(paths, monkeypatch):
    before = paths.fmriprep_status.read_text()
    monkeypatch.setattr(status, "sacct_states", lambda ids: {})
    status.reconcile(paths)
    status.job_started(paths, "01", "11", "n1")
    assert paths.fmriprep_status.read_text() == before


def test_new_fmriprep_results_become_ready(paths, monkeypatch):
    monkeypatch.setattr(status, "sacct_states", lambda ids: {})
    write_fmriprep_status(paths.fmriprep_status, [
        {"subject": "02", "class": "partial", "runs": "A", "collect": "DONE", "state": "DONE"},
        {"subject": "01", "class": "complete", "runs": "A,B", "collect": "DONE", "state": "DONE"}])
    status.reconcile(paths)
    assert "02" in status.next_subjects(status.read_table(paths))


@pytest.mark.parametrize("sacct,state,ready", [("PREEMPTED", "PREEMPTED", True), ("NODE_FAIL", "PREEMPTED", True),
                                               ("OUT_OF_MEMORY", "FAILED", False), ("TIMEOUT", "FAILED", False),
                                               ("CANCELLED", "FAILED", False), ("COMPLETED", "FAILED", False),
                                               ("RUNNING", "RUNNING", False)])
def test_reconcile(paths, monkeypatch, sacct, state, ready):
    status.job_started(paths, "01", "11", "n1")
    monkeypatch.setattr(status, "sacct_states", lambda ids: {"11": sacct})
    status.reconcile(paths)
    row = status.read_table(paths)["01"]
    assert row["state"] == state and ("01" in status.next_subjects(status.read_table(paths))) == ready
    assert row["preemptions"] == ("1" if state == "PREEMPTED" else "0")


def test_preempted_first_and_capped(paths, monkeypatch):
    status.job_started(paths, "04", "11", "n1")
    monkeypatch.setattr(status, "sacct_states", lambda ids: {"11": "PREEMPTED"})
    status.reconcile(paths)
    assert status.next_subjects(status.read_table(paths))[0] == "04"
    with status.locked(paths):
        rows = status.read_table(paths)
        rows["04"]["preemptions"] = str(status.MAX_PREEMPTIONS)
        status.write_table(paths, rows)
        status.render(paths, rows)
    assert status.next_subjects(status.read_table(paths)) == ["01"]
    assert "no longer resubmitted" in paths.md.read_text()


def test_a_signal_is_interrupted_and_an_error_needs_a_human(paths):
    status.job_failed(paths, "01", "SIGTERM")
    assert status.read_table(paths)["01"]["state"] == "INTERRUPTED(SIGTERM)"
    status.job_failed(paths, "04", "ValueError: bad")
    row = status.read_table(paths)["04"]
    assert row["state"] == "FAILED" and row["note"] == "ValueError: bad"
    assert status.next_subjects(status.read_table(paths)) == []
    assert "04" in paths.md.read_text()


def test_failed_subjects_keep_the_manager_alive(paths, capsys):
    args = ["--dir", str(paths.dir), "--fmriprep-status", str(paths.fmriprep_status), "--split", str(paths.split),
            "--md", str(paths.md)]
    with status.locked(paths):
        rows = status.read_table(paths)
        for s in rows:
            if rows[s]["class"] != "unusable":
                rows[s]["state"] = "DONE"
        status.write_table(paths, rows)
    capsys.readouterr()
    status.main(args + ["remaining"])
    assert capsys.readouterr().out.strip() == "0"
    status.job_failed(paths, "01", "ValueError: bad")
    status.main(args + ["remaining"])
    assert capsys.readouterr().out.strip() == "1"
    status.main(args + ["reset", "01"])
    assert status.read_table(paths)["01"]["state"] == "PENDING"


def test_summary_counts_per_set(paths):
    status.job_done(paths, "01", 2, 1, 12.5, note="peak 9.0 GiB")
    text = paths.md.read_text()
    assert "| train | 2 | 2 | 1 | 2 / 4 | 0 |" in text
    assert "| **all** | 4 | 2 | 1 | 2 / 7 | 0 |" in text
    assert "12.5 (mean of 1)" in text


# ---------------------------------------------------------------- worker
@pytest.fixture
def stored_grid(tmp_path):
    make_subject(tmp_path / "fmriprep", "01")
    cfg = make_cfg(tmp_path)
    runs = build_manifest(cfg)
    grid, info = build_group_mask(runs, _template(tmp_path), 0.5)
    save_grid(cfg["output_dir"], grid, info)
    return cfg


def _template(tmp_path):
    path = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.uint8), AFFINE), path)
    return path


def test_worker_stores_the_runs_and_resumes(stored_grid, monkeypatch):
    cfg = stored_grid
    assert worker.process_subject(cfg, "01", 2) == (2, 0)
    manifest = load_manifest(cfg["output_dir"])
    run = VoxelRun.load(cfg["output_dir"], manifest.iloc[0])
    assert run.data.shape == (120, 36) and run.data.dtype == np.float16

    calls = []
    monkeypatch.setattr(worker, "denoise_run", lambda *a: calls.append(a))
    assert worker.process_subject(cfg, "01", 2) == (2, 0)     # all stored: no work
    assert calls == []


def test_worker_refuses_a_wrong_run_count(stored_grid):
    with pytest.raises(ValueError, match="expected 4"):
        worker.process_subject(stored_grid, "01", 4)


def test_worker_main_records_success_and_failure(stored_grid, paths, monkeypatch):
    monkeypatch.setattr(worker, "load_config", lambda path: stored_grid)
    monkeypatch.setattr(worker, "Paths", lambda d: paths)
    monkeypatch.setenv("SLURM_JOB_ID", "55")
    worker.main(["01", "--config", "x", "--dir", "x"])
    row = status.read_table(paths)["01"]
    assert row["state"] == "DONE" and row["stored_runs"] == "2" and row["attempts"] == "1"
    assert row["last_jobid"] == "55" and row["note"].startswith("peak")
    with pytest.raises(SystemExit):
        worker.main(["04", "--config", "x", "--dir", "x"])    # has no files
    row = status.read_table(paths)["04"]
    assert row["state"] == "FAILED" and "FileNotFoundError" in row["note"]


def test_sbatch_scripts_parse():
    for script in ("preprocess_manager.sbatch", "preprocess_subject.sbatch"):
        subprocess.run(["bash", "-n", f"scripts/{script}"], check=True)
