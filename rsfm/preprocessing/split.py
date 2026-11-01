"""Subject split, fixed in advance over every subject with rest data, processed or not.

Input is the fMRIPrep run's status.tsv, which lists every S3 subject and its class. "complete"
and "partial" subjects are split within each class, so subjects with missing runs are spread
proportionally. "unusable" subjects have no rest run and are left out.

Families are ignored so far (see STATUS.md): relatives can land in different sets, so a split made
here is provisional until a family-aware one replaces it. Versions are never overwritten.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

USABLE_CLASSES = ("complete", "partial")
SETS = ("train", "val", "test")


def read_fmriprep_status(path: Path) -> dict[str, dict]:
    """The fMRIPrep run's status.tsv as {subject: row}. Read-only; that file is replaced atomically."""
    with open(path, newline="") as f:
        return {row["subject"]: row for row in csv.DictReader(f, delimiter="\t")}


def make_split(classes: dict[str, str], fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> dict[str, str]:
    """{subject: class} -> {subject: set}, for the usable classes only."""
    assert len(fractions) == 3 and abs(sum(fractions) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    assignment = {}
    for cls in USABLE_CLASSES:
        subjects = sorted(s for s, c in classes.items() if c == cls)
        order = rng.permutation(subjects).tolist()
        n_train = round(fractions[0] * len(order))
        n_val = round(fractions[1] * len(order))
        for i, subject in enumerate(order):
            assignment[subject] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def write_split(assignment: dict[str, str], classes: dict[str, str], out_dir: Path, name: str) -> Path:
    """Writes <name>.tsv (subject, class, set) and <name>_<set>.txt, one subject per line."""
    out_dir = Path(out_dir)
    table = out_dir / f"{name}.tsv"
    if table.exists():
        raise FileExistsError(f"{table} exists; a split is never overwritten, use a new name")
    out_dir.mkdir(parents=True, exist_ok=True)
    for set_name in SETS:
        subjects = sorted(s for s, x in assignment.items() if x == set_name)
        (out_dir / f"{name}_{set_name}.txt").write_text("\n".join(subjects) + "\n")
    with open(table, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["subject", "class", "set"])
        for subject in sorted(assignment):
            writer.writerow([subject, classes[subject], assignment[subject]])
    return table


def read_split(path: Path) -> dict[str, str]:
    """<name>.tsv as {subject: set}."""
    with open(path, newline="") as f:
        return {row["subject"]: row["set"] for row in csv.DictReader(f, delimiter="\t")}


def read_subject_list(path: Path) -> list[str]:
    return Path(path).read_text().split()
