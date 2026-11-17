import sys, json, time
import numpy as np
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS
OOD_FAMILY = sys.argv[2]
c = float(sys.argv[1]); T = 400; n_test = 40
out = {}
for regime, seed, fams in (("iid", 5000 + int(c*100), TRAIN_FAMILIES),
                           ("graph_ood", 6000 + int(c*100), [OOD_FAMILY])):
    t0 = time.time()
    pool = make_pool(n_test, seed, fams, TRAIN_MECHS, tag=regime, T=T,
                     corruption=c, corrupt_op="delete")
    for s in pool:
        A = np.asarray(s.y).reshape(-1)
        true = np.asarray(s.y).astype(int).reshape(-1)
        S = np.asarray(s.S if hasattr(s, "S") else s.sysinst.S).reshape(-1)
        m = int(true.sum()); n = int(((true == 1) & (S <= 0.0)).sum())
        out[f"{regime}|{s.sysinst.graph_id}"] = (m, n)
    print(regime, "pool built in %.0fs" % (time.time()-t0), flush=True)
json.dump(out, open(f"{sys.argv[3]}/ceil_{c}.json", "w"))
print("wrote", len(out), "entries")
