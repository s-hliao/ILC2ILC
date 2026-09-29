"""One folder per ILC run, one self-contained npz per trial.

    <root>/<run_name>/
        meta.json          who made the run: task config, ILC weights, controller
                           parameters, sim conditions, and its parent (a resumed or
                           transferred-from trial), if any
        reference.npz      the TO reference the run flew (JumpILC.save_reference)
        trial_001.npz      one per trial, written right after that trial's ILC update
        trial_002.npz
        ...

Each trial file stands on its own -- it carries the reference and the learning state
as well as the recording -- so a single file is enough to resume learning from that
trial, to branch a new run off it, or to transfer what it learned to another task
(JumpILC.restore / JumpILC.transfer_from / JumpILC.from_trial):

    trial, stage          1-based trial number and the stage it was flown in
    U_flown, U_next       the forces this trial flew, and the ILC's forces for the next
    result, history       json: this trial's result, and every result up to it
    config                json: JumpILC.config (robot, task, phases, TO settings)
    goal                  the CoM landing target
    x_ref, u_ref, info_*  the reference (as in reference.npz)
    log_*                 the trial on the TO grid, as JumpILC.update takes it
    rec_*                 the raw recording at the control rate (optional)
    extra                 json: anything else the caller passes

    rec = TrialRecorder.load("/ilc_ws/log/go1_f40/")            # last trial of a run
    rec = TrialRecorder.load("/ilc_ws/log/go1_f40/", trial=7)    # or any trial
    ilc.restore(rec)                                             # continue from it
"""

from __future__ import annotations

import datetime
import glob
import json
import os

import numpy as np


def _json(obj) -> str:
    return json.dumps(obj, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))


class TrialRecord(dict):
    """A loaded trial file: the npz arrays, with the json fields decoded."""

    JSON_KEYS = ("result", "history", "config", "extra")

    @property
    def log(self) -> dict:
        """The trial on the TO grid, the dict JumpILC.update takes."""
        return {k[4:]: v for k, v in self.items() if k.startswith("log_")}

    @property
    def reference(self) -> dict:
        """The reference as JumpILC.save_reference writes it."""
        keys = ("x_ref", "u_ref") + tuple(k for k in self if k.startswith("info_"))
        return dict({k: self[k] for k in keys}, config=_json(self["config"]))


class TrialRecorder:
    """Writes a run folder: meta.json, reference.npz, then trial_NNN.npz per trial."""

    def __init__(self, root: str, run_name: str = "", meta: dict | None = None):
        if not run_name:
            run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(root, run_name)
        if os.path.exists(os.path.join(self.run_dir, "meta.json")):
            raise FileExistsError(f"{self.run_dir} already holds a run; pick another run_name")
        os.makedirs(self.run_dir, exist_ok=True)
        self.meta = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                         **(meta or {}))
        self._write_meta()

    def annotate(self, **fields) -> None:
        """Add fields to meta.json (e.g. sim conditions the controller is not told)."""
        self.meta.update(fields)
        self._write_meta()

    def _write_meta(self) -> None:
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            f.write(json.dumps(json.loads(_json(self.meta)), indent=1))

    def save_reference(self, ilc) -> str:
        path = os.path.join(self.run_dir, "reference.npz")
        ilc.save_reference(path)
        return path

    def record(self, ilc, U_flown, log: dict, rec: dict | None = None,
               extra: dict | None = None) -> str:
        """Write the trial ilc.update just scored; call it right after the update."""
        result = ilc.history[-1]
        path = os.path.join(self.run_dir, f"trial_{result['trial']:03d}.npz")
        np.savez(
            path, trial=result["trial"], stage=result["stage"], U_flown=U_flown,
            U_next=ilc.U, goal=ilc.goal, result=_json(result), history=_json(ilc.history),
            extra=_json(extra or {}),
            **ilc.reference_arrays(),                     # the reference, config included
            **{f"log_{k}": v for k, v in log.items()},
            **{f"rec_{k}": v for k, v in (rec or {}).items()})
        return path

    # -- reading ---------------------------------------------------------------------------

    @staticmethod
    def list_trials(run_dir: str) -> list[str]:
        return sorted(glob.glob(os.path.join(run_dir, "trial_*.npz")))

    @staticmethod
    def load(path: str, trial: int | None = None) -> TrialRecord:
        """A trial file, or a run folder's trial `trial` (default: its last)."""
        if os.path.isdir(path):
            if trial is None:
                files = TrialRecorder.list_trials(path)
                if not files:
                    raise FileNotFoundError(f"no trials in {path}")
                path = files[-1]
            else:
                path = os.path.join(path, f"trial_{int(trial):03d}.npz")
        with np.load(path) as data:
            rec = TrialRecord({k: data[k] for k in data.files})
        for k in TrialRecord.JSON_KEYS:
            if k in rec:
                rec[k] = json.loads(str(rec[k]))
        rec["trial"], rec["stage"] = int(rec["trial"]), int(rec["stage"])
        rec["path"] = os.path.abspath(path)
        return rec
