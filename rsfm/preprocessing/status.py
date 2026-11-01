"""Per-subject status of the denoising run, in the style of the fMRIPrep run's hcp_status.py.

This is the only writer of <jobs>/status.tsv (one row per subject) and of the generated
PREPROCESSING_STATUS.md. It reads the fMRIPrep run's status.tsv but never writes to it.

    python -m rsfm.preprocessing.status <command>
      init       create the table from the fMRIPrep status and the split
      reconcile  refresh fMRIPrep states; compare QUEUED/RUNNING/INTERRUPTED rows with sacct
      next       print subjects that are ready to submit
      queued     mark a subject as submitted (manager)
      remaining  print how many subjects aren't finished; FAILED counts, since it needs a human
      reset      FAILED/INTERRUPTED/PREEMPTED -> PENDING, once a human has fixed the cause
      render     regenerate PREPROCESSING_STATUS.md
      show       print one subject's row

Writes take a POSIX lock and replace the file atomically, so jobs on different nodes can update it
and a job killed mid-write never leaves it half-written.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .split import read_fmriprep_status, read_split

HCP_ROOT = Path("/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200")
DEFAULT_DIR = HCP_ROOT / "derivatives" / "rsfm_voxel" / "jobs"
DEFAULT_FMRIPREP_STATUS = HCP_ROOT / "jobs" / "full" / "status.tsv"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = REPO / "splits" / "hcp_interim_v1.tsv"
DEFAULT_MD = REPO / "PREPROCESSING_STATUS.md"

COLUMNS = ["subject", "class", "set", "expected_runs", "fmriprep", "stored_runs", "excluded_runs", "state",
           "attempts", "preemptions", "last_jobid", "node", "updated", "minutes", "note"]
MAX_PREEMPTIONS = 5
GB_PER_RUN = 0.56
ALIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING"}
PREEMPT_LIKE = {"PREEMPTED", "NODE_FAIL"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Paths:
    def __init__(self, directory=DEFAULT_DIR, fmriprep_status=DEFAULT_FMRIPREP_STATUS, split=DEFAULT_SPLIT,
                 md=DEFAULT_MD):
        self.dir = Path(directory)
        self.status = self.dir / "status.tsv"
        self.lock = self.dir / "status.tsv.lock"
        self.conf = self.dir / "manager.conf"
        self.stop = self.dir / "STOP"
        self.logs = self.dir / "logs"
        self.fmriprep_status = Path(fmriprep_status)
        self.split = Path(split)
        self.md = Path(md)


# ---------------------------------------------------------------- table I/O
@contextmanager
def locked(p: Paths):
    p.dir.mkdir(parents=True, exist_ok=True)
    with open(p.lock, "a+") as fh:
        fcntl.lockf(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)


def read_table(p: Paths) -> dict[str, dict]:
    if not p.status.exists():
        return {}
    with open(p.status, newline="") as f:
        return {row["subject"]: row for row in csv.DictReader(f, delimiter="\t")}


def write_table(p: Paths, rows: dict[str, dict]) -> None:
    tmp = p.status.with_name(f"status.tsv.tmp.{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for subject in sorted(rows):
            writer.writerow({c: str(rows[subject].get(c, "")).replace("\t", " ").replace("\n", " ") for c in COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p.status)


def read_conf(p: Paths) -> dict:
    conf = {"max_active": 8, "interval_sec": 900}
    if p.conf.exists():
        for line in p.conf.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                key, value = (x.strip() for x in line.split("=", 1))
                conf[key] = int(value) if value.isdigit() else value
    return conf


def refresh_fmriprep(p: Paths, rows: dict[str, dict]) -> None:
    """Copy each subject's fMRIPrep state. "ready" needs both state DONE and collect DONE, so a run
    is never read while it is being copied into fmriprep_rest."""
    source = read_fmriprep_status(p.fmriprep_status)
    for subject, row in rows.items():
        src = source.get(subject)
        if src is None:
            row["fmriprep"] = "unknown"
        elif src["state"] == "DONE" and src["collect"] != "DONE":
            row["fmriprep"] = "DONE(not collected)"
        else:
            row["fmriprep"] = src["state"]


def is_ready(row: dict) -> bool:
    """fMRIPrep finished, and this subject is waiting or was preempted (under the cap)."""
    if row["fmriprep"] != "DONE":
        return False
    return row["state"] == "PENDING" or (row["state"] == "PREEMPTED"
                                         and int(row["preemptions"] or 0) < MAX_PREEMPTIONS)


def next_subjects(rows: dict[str, dict]) -> list[str]:
    """Preempted subjects first, then pending ones in subject order."""
    ready = [s for s in sorted(rows) if is_ready(rows[s])]
    return [s for s in ready if rows[s]["state"] == "PREEMPTED"] + [s for s in ready if rows[s]["state"] == "PENDING"]


def is_finished(row: dict) -> bool:
    return row["state"] == "DONE" or row["state"].startswith("SKIPPED")


# ---------------------------------------------------------------- transitions used by the worker
def job_started(p: Paths, subject: str, jobid: str, node: str) -> None:
    with locked(p):
        rows = read_table(p)
        rows[subject].update(state="RUNNING", attempts=str(int(rows[subject]["attempts"] or 0) + 1),
                             last_jobid=jobid, node=node, updated=now_iso())
        write_table(p, rows)
        render(p, rows)


def job_done(p: Paths, subject: str, stored: int, excluded: int, minutes: float, note: str = "") -> None:
    with locked(p):
        rows = read_table(p)
        rows[subject].update(state="DONE", stored_runs=str(stored), excluded_runs=str(excluded),
                             minutes=f"{minutes:.1f}", note=note, updated=now_iso())
        write_table(p, rows)
        render(p, rows)


def job_failed(p: Paths, subject: str, reason: str, stored: int | None = None) -> None:
    """A signal means killed from outside (reconcile decides how); anything else needs a human."""
    with locked(p):
        rows = read_table(p)
        state = f"INTERRUPTED({reason})" if reason.startswith("SIG") else "FAILED"
        rows[subject].update(state=state, updated=now_iso())
        if not reason.startswith("SIG"):
            rows[subject]["note"] = reason
        if stored is not None:
            rows[subject]["stored_runs"] = str(stored)
        write_table(p, rows)
        render(p, rows)


# ---------------------------------------------------------------- commands
def cmd_init(p: Paths, a) -> None:
    if p.status.exists() and not a.force:
        sys.exit(f"{p.status} exists; --force adds new subjects and keeps existing rows")
    source = read_fmriprep_status(p.fmriprep_status)
    split = read_split(p.split)
    with locked(p):
        rows = read_table(p)
        for subject, src in source.items():
            row = rows.setdefault(subject, {c: "" for c in COLUMNS})
            usable = src["class"] != "unusable"
            row.update(subject=subject, **{"class": src["class"]}, set=split.get(subject, "-"),
                       expected_runs=str(len([r for r in src["runs"].split(",") if r])))
            if not row["state"]:
                row.update(state="PENDING" if usable else "SKIPPED(unusable)", stored_runs="0", excluded_runs="0",
                           attempts="0", preemptions="0", updated=now_iso())
            if usable and subject not in split:
                sys.exit(f"{subject} has rest data but isn't in {p.split}")
        refresh_fmriprep(p, rows)
        write_table(p, rows)
        render(p, rows)
    print(f"{len(rows)} subjects in {p.status}")


def sacct_states(jobids: set[str]) -> dict[str, str]:
    jobids = {j for j in jobids if j.isdigit()}
    if not jobids:
        return {}
    r = subprocess.run(["sacct", "-a", "-n", "-X", "-P", "-o", "JobID,State,ExitCode", "-j", ",".join(sorted(jobids))],
                       capture_output=True, text=True, timeout=120)
    states = {}
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            states[parts[0]] = parts[1].split()[0]            # "CANCELLED by 123" -> "CANCELLED"
    return states


def reconcile(p: Paths) -> list[str]:
    """Jobs that died without updating their row become PREEMPTED (resubmitted) or FAILED (a human)."""
    with locked(p):
        rows = read_table(p)
        refresh_fmriprep(p, rows)
        todo = {s: r for s, r in rows.items()
                if (r["state"] in ("QUEUED", "RUNNING") or r["state"].startswith("INTERRUPTED")) and r["last_jobid"]}
        states = sacct_states({r["last_jobid"] for r in todo.values()})
        changes = []
        for subject, row in todo.items():
            state = states.get(row["last_jobid"])
            if state is None or state in ALIVE:
                continue
            if state in PREEMPT_LIKE:
                row["state"] = "PREEMPTED"
                row["preemptions"] = str(int(row["preemptions"] or 0) + 1)
            elif state == "COMPLETED":
                row["state"] = "FAILED"
                row["note"] = "job completed without reaching DONE"
            else:
                row["state"] = "FAILED"
                row["note"] = f"job {state}"
            row["updated"] = now_iso()
            changes.append(f"{subject}: job {row['last_jobid']} {state} -> {row['state']}")
        write_table(p, rows)
        render(p, rows)
    return changes


def cmd_reconcile(p: Paths, a) -> None:
    changes = reconcile(p)
    for change in changes:
        print(change)
    print(f"reconcile: {len(changes)} changed")


def cmd_next(p: Paths, a) -> None:
    subjects = next_subjects(read_table(p))
    for subject in subjects[: a.n] if a.n else subjects:
        print(subject)


def cmd_queued(p: Paths, a) -> None:
    with locked(p):
        rows = read_table(p)
        rows[a.subject].update(state="QUEUED", last_jobid=a.jobid, updated=now_iso())
        write_table(p, rows)
        render(p, rows)


def cmd_remaining(p: Paths, a) -> None:
    print(sum(not is_finished(r) for r in read_table(p).values()))


def cmd_reset(p: Paths, a) -> None:
    with locked(p):
        rows = read_table(p)
        row = rows.get(a.subject)
        if row is None:
            sys.exit(f"{a.subject} is not in {p.status}")
        if row["state"] in ("RUNNING", "QUEUED", "DONE") and not a.force:
            sys.exit(f"{a.subject} is {row['state']}; refusing without --force")
        row.update(state="PENDING", updated=now_iso())
        if a.clear_preemptions:
            row["preemptions"] = "0"
        write_table(p, rows)
        render(p, rows)
    print(f"{a.subject}: reset to PENDING")


def cmd_render(p: Paths, a) -> None:
    render(p, read_table(p))
    print(f"wrote {p.md}")


def cmd_show(p: Paths, a) -> None:
    row = read_table(p).get(a.subject)
    if row is None:
        sys.exit(f"{a.subject} is not in {p.status}")
    for column in COLUMNS:
        print(f"{column:14} {row[column]}")


# ---------------------------------------------------------------- PREPROCESSING_STATUS.md
def _summary_row(name: str, rows: list[dict]) -> str:
    expected = sum(int(r["expected_runs"] or 0) for r in rows)
    stored = sum(int(r["stored_runs"] or 0) for r in rows)
    fmriprep = sum(r["fmriprep"] == "DONE" for r in rows)
    done = sum(r["state"] == "DONE" for r in rows)
    failed = sum(r["state"].startswith("FAILED") for r in rows)
    return f"| {name} | {len(rows)} | {fmriprep} | {done} | {stored} / {expected} | {failed} |"


def render(p: Paths, rows: dict[str, dict]) -> None:
    conf = read_conf(p)
    table = list(rows.values())
    usable = [r for r in table if r["class"] != "unusable"]
    by_state: dict[str, int] = {}
    for row in table:
        key = re.sub(r"\(.*", "", row["state"])
        by_state[key] = by_state.get(key, 0) + 1
    running = sorted((r for r in table if r["state"] in ("RUNNING", "QUEUED")), key=lambda r: r["subject"])
    failed = sorted((r for r in table if r["state"].startswith(("FAILED", "INTERRUPTED"))), key=lambda r: r["subject"])
    preempted = sorted((r for r in table if r["state"] == "PREEMPTED"), key=lambda r: r["subject"])
    stuck = [r for r in preempted if int(r["preemptions"] or 0) >= MAX_PREEMPTIONS]
    minutes = [float(r["minutes"]) for r in usable if r["state"] == "DONE" and r["minutes"]]
    mean_minutes = sum(minutes) / len(minutes) if minutes else None
    stored = sum(int(r["stored_runs"] or 0) for r in usable)
    expected = sum(int(r["expected_runs"] or 0) for r in usable)
    ready = len(next_subjects(rows))
    waiting = sum(r["fmriprep"] != "DONE" and not is_finished(r) for r in usable)

    L = ["# HCP denoising: status\n",
         f"_Generated by `rsfm/preprocessing/status.py` at {now_iso()} from `{p.status}`. Do not edit by hand._\n",
         "Design: `docs/PREPROCESSING_PLAN.md`. fMRIPrep run: "
         f"`{p.fmriprep_status.parent / 'STATUS.md'}`. Split: `{p.split.name}` (provisional, see `STATUS.md`).\n"]
    if p.stop.exists():
        L.append(f"`{p.stop}` exists: the manager submits no new jobs.\n")
    L.append("## Summary\n")
    L.append("| Set | Subjects | fMRIPrep done | Denoised | Runs stored / expected | Failed |\n|---|---|---|---|---|---|")
    for set_name in ("train", "val", "test"):
        L.append(_summary_row(set_name, [r for r in usable if r["set"] == set_name]))
    L.append(_summary_row("**all**", usable))
    L.append("")
    L.append("| | |\n|---|---|")
    L.append("| States | " + ", ".join(f"{k} {v}" for k, v in sorted(by_state.items())) + " |")
    L.append(f"| Ready now / waiting for fMRIPrep | {ready} / {waiting} |")
    L.append(f"| Stored data | about {stored * GB_PER_RUN:.0f} GB of {expected * GB_PER_RUN:.0f} GB |")
    if mean_minutes is None:
        L.append("| Minutes per subject | not measured yet |")
    else:
        hours = ready * mean_minutes / max(1, int(conf["max_active"])) / 60
        L.append(f"| Minutes per subject | {mean_minutes:.1f} (mean of {len(minutes)}) |")
        L.append(f"| Time to clear what's ready | about {hours:.1f} h at {conf['max_active']} jobs; "
                 "the rest depends on fMRIPrep |")
    L.append("")
    if running:
        L.append("## Running / queued\n")
        L.append("| subject | set | state | job | node | attempts | since |\n|---|---|---|---|---|---|---|")
        for r in running:
            L.append(f"| {r['subject']} | {r['set']} | {r['state']} | {r['last_jobid']} | {r['node']} | "
                     f"{r['attempts']} | {r['updated']} |")
        L.append("")
    if failed:
        L.append("## Failed / interrupted (need a human)\n")
        L.append("| subject | state | job | log | note |\n|---|---|---|---|---|")
        for r in failed:
            log = p.logs / f"rsfmd_{r['subject']}_{r['last_jobid']}.log"
            L.append(f"| {r['subject']} | {r['state']} | {r['last_jobid']} | `{log}` | {r['note']} |")
        L.append("")
    if preempted:
        L.append("## Preempted (resubmitted automatically)\n")
        L.append(", ".join(f"{r['subject']} ({r['preemptions']}×)" for r in preempted) + "\n")
    if stuck:
        L.append(f"**Preempted {MAX_PREEMPTIONS}+ times, no longer resubmitted:** "
                 + ", ".join(r["subject"] for r in stuck) + "\n")
    L.append("## Next actions\n")
    actions = []
    if failed:
        actions.append("Read each failed subject's log, fix the cause, then "
                       "`python -m rsfm.preprocessing.status reset <S>`.")
    if stuck:
        actions.append("Decide what to do with the repeatedly preempted subjects, then reset them with "
                       "`--clear-preemptions`.")
    actions.append("Is the manager alive? `squeue -u $USER -n rsfm_manager`. If not, and work is left: "
                   "`sbatch scripts/preprocess_manager.sbatch`.")
    actions.append(f"Pause new submissions: `touch {p.stop}`. Settings, read every loop: `{p.conf}`.")
    for i, action in enumerate(actions, 1):
        L.append(f"{i}. {action}")
    L.append("")
    L.append("## Legend\n")
    L.append("States: PENDING, QUEUED, RUNNING, DONE, FAILED, INTERRUPTED(signal), PREEMPTED, SKIPPED(reason). "
             f"One row per subject in `{p.status}`; logs in `{p.logs}/rsfmd_<S>_<jobid>.log`.\n")
    tmp = p.md.with_name(f"{p.md.name}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(L))
    os.replace(tmp, p.md)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--fmriprep-status", default=str(DEFAULT_FMRIPREP_STATUS))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").add_argument("--force", action="store_true")
    sub.add_parser("reconcile")
    sub.add_parser("next").add_argument("--n", type=int, default=0)
    s = sub.add_parser("queued")
    s.add_argument("subject")
    s.add_argument("--jobid", required=True)
    sub.add_parser("remaining")
    s = sub.add_parser("reset")
    s.add_argument("subject")
    s.add_argument("--force", action="store_true")
    s.add_argument("--clear-preemptions", action="store_true")
    sub.add_parser("render")
    sub.add_parser("show").add_argument("subject")
    a = ap.parse_args(argv)
    p = Paths(a.dir, a.fmriprep_status, a.split, a.md)
    {"init": cmd_init, "reconcile": cmd_reconcile, "next": cmd_next, "queued": cmd_queued, "remaining": cmd_remaining,
     "reset": cmd_reset, "render": cmd_render, "show": cmd_show}[a.cmd](p, a)


if __name__ == "__main__":
    main()
