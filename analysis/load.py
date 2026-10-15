import json, glob, os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load(name: str, strict_run: bool = True) -> pd.DataFrame:
    """Load a results file.

    Each run writes a companion ``*.meta.json`` naming its run id.  A results
    file can accumulate rows from an aborted earlier run if the same path is
    reused, so by default we keep only the rows belonging to the run recorded
    in the metadata, and report how many stale rows were dropped.
    """
    path = os.path.join(RESULTS, name)
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    meta_path = path.replace(".jsonl", ".meta.json")
    if strict_run and os.path.exists(meta_path) and "run_id" in df:
        run_id = json.load(open(meta_path))["run_id"]
        stale = int((df.run_id != run_id).sum())
        if stale:
            print(f"[load] {name}: dropped {stale} rows from earlier run(s)")
        df = df[df.run_id == run_id].reset_index(drop=True)
    return df

def load_multi(names, strict_run: bool = True):
    """Concatenate several results files that belong to one logical experiment.

    Used where an experiment was extended with additional seeds after the first
    run: each file keeps its own run id and metadata, and the rows are pooled
    only at analysis time.
    """
    import pandas as _pd
    return _pd.concat([load(n, strict_run) for n in names], ignore_index=True)


LABELS = {
    "spcrl_amortized_soft": "SP-CRL (soft prior)",
    "spcrl_amortized_hard": "hard structural mask",
    "spcrl_amortized_none": "no structural prior",
    "spcrl_amortized_soft_noenc": "SP-CRL, no prior encoder",
    "spcrl_direct_soft": "SP-CRL-D (soft)",
    "spcrl_direct_hard": "SP-CRL-D (hard)",
    "spcrl_direct_none": "SP-CRL-D (none)",
    "notears_linear": "NOTEARS (linear)",
    "notears_linear_hardmask": "NOTEARS + hard mask",
    "pc": "PC", "pc_prior": "PC + prior",
    "pcorr": "partial correlation", "pcorr_prior": "partial corr. + prior",
    "prior_only": "structural prior only",
    "oracle": "true-graph reference",
    "spcrl_soft": "SP-CRL (soft prior)", "spcrl_hard": "hard structural mask",
    "spcrl_none": "no structural prior",
    "anomaly_no_structure": "anomaly ranking (no structure)",
    "spcrl_amortized_soft_noenc": "prior features only (no prior attention)",
    "spcrl_amortized_soft_gate": "prior features + reliability gate",
}
REGIME = {"iid": "in-distribution", "mech_ood": "unseen mechanisms",
          "graph_ood": "unseen graph family", "size_ood": "unseen system size",
          "noise_ood": "unseen noise", "joint_ood": "joint shift"}
