from __future__ import annotations
import json, os, platform, subprocess, sys, time, uuid
import numpy as np


def provenance() -> dict:
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL,
                                           cwd=os.path.dirname(os.path.dirname(__file__))).decode().strip()
        except Exception:
            return "unavailable"
    import torch, sklearn, scipy
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain") != "",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


class JsonlWriter:
    def __init__(self, path: str, meta: dict | None = None, append: bool = False):
        """append=True keeps rows written by earlier, interrupted runs of the
        same experiment and reuses their run id, so a restartable experiment
        still has a single provenance record."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        meta_path = path.replace(".jsonl", ".meta.json")
        prev = None
        if append and os.path.exists(meta_path):
            try:
                prev = json.load(open(meta_path)).get("run_id")
            except Exception:
                prev = None
        self.run_id = prev or uuid.uuid4().hex[:12]
        self.prov = provenance()
        if meta is not None:
            with open(meta_path, "w") as f:
                json.dump({"run_id": self.run_id, "provenance": self.prov,
                           "config": meta}, f, indent=2, default=str)

    def write(self, row: dict):
        row = dict(row); row["run_id"] = self.run_id
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=_conv) + "\n")


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
