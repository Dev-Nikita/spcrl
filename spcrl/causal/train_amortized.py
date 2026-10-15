from __future__ import annotations
import numpy as np, torch, torch.nn as nn, time
from sklearn.metrics import average_precision_score, roc_auc_score
from .amortized import AmortizedSPCRL, build_variant


def _tensors(s):
    return (torch.tensor(s.Fp), torch.tensor(s.Fn), torch.tensor(s.S),
            torch.tensor(s.y))


def train_amortized(train_pool, val_pool, mode="soft", use_encoder=True,
                    epochs=30, lr=1e-3, seed=0, pos_weight=None, verbose=False,
                    patience=6, variant=None):
    """`variant` (see amortized.build_variant) overrides mode/use_encoder."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_variant(variant) if variant else AmortizedSPCRL(mode=mode, use_encoder=use_encoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cache = [_tensors(s) for s in train_pool]
    vcache = [_tensors(s) for s in val_pool]
    if pos_weight is None:
        pos = sum(float(s.y.sum()) for s in train_pool)
        tot = sum(s.y.size - s.y.shape[0] for s in train_pool)
        pos_weight = max(1.0, (tot - pos) / max(pos, 1.0)) ** 0.5
    pw = torch.tensor(float(pos_weight))
    best, best_state, bad = -np.inf, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(cache))
        tot_loss = 0.0
        for k in idx:
            Fp, Fn, S, y = cache[k]
            opt.zero_grad()
            logit = model(Fp, Fn, S)
            d = y.shape[0]
            m = ~torch.eye(d, dtype=torch.bool)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit[m], y[m], pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.detach())
        vs = evaluate_pool(model, vcache)
        hist.append({"epoch": ep, "train_loss": tot_loss / len(cache), "val_auprc": vs})
        if verbose:
            print(f"  ep{ep} loss={tot_loss/len(cache):.4f} val_auprc={vs:.4f}", flush=True)
        if vs > best + 1e-4:
            best, bad = vs, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auprc": best, "history": hist}


def evaluate_pool(model, cache) -> float:
    model.eval()
    aps = []
    with torch.no_grad():
        for Fp, Fn, S, y in cache:
            lg = model(Fp, Fn, S)
            d = y.shape[0]
            m = ~torch.eye(d, dtype=torch.bool)
            yv = y[m].numpy()
            if yv.sum() == 0:
                continue
            aps.append(average_precision_score(yv, torch.sigmoid(lg)[m].numpy()))
    return float(np.mean(aps)) if aps else 0.0


def score_pool(model, pool):
    """Per-system AUPRC / AUROC on a pool of Samples."""
    model.eval()
    rows = []
    with torch.no_grad():
        for s in pool:
            Fp, Fn, S, y = _tensors(s)
            p = torch.sigmoid(model(Fp, Fn, S)).numpy()
            d = s.y.shape[0]
            m = ~np.eye(d, dtype=bool)
            yv = s.y[m].ravel(); pv = p[m].ravel()
            if yv.sum() == 0:
                continue
            rows.append({"graph_id": s.sysinst.graph_id, "d": d,
                         "family": s.sysinst.family, "mechanism": s.sysinst.mechanism,
                         "auprc": float(average_precision_score(yv, pv)),
                         "auroc": float(roc_auc_score(yv, pv)),
                         "P": p, "tag": s.tag})
    return rows
