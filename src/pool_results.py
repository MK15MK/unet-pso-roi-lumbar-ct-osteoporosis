import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

CLASSES = ["normal", "osteopenia", "osteoporosis"]
PROB_COLS = ["p_normal", "p_osteopenia", "p_osteoporosis"]
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_ovr_macro", "mcc",
           "cohen_kappa", "brier_score"]
PERM_METRICS = ["f1_macro", "balanced_accuracy", "auc_ovr_macro", "mcc"]


def metric_row(y, p):
    pred = p.argmax(1)
    try:
        auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, pred, average="macro", zero_division=0), "auc_ovr_macro": auc,
            "mcc": matthews_corrcoef(y, pred), "cohen_kappa": cohen_kappa_score(y, pred),
            "brier_score": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))}


def load_method(out_dir, tag, method):
    files = sorted(glob.glob(os.path.join(out_dir, f"pred_{tag}_{method}_fold*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df["patient_id"].duplicated().any():
        raise ValueError(f"{method}: a patient appears in more than one test fold")
    return df.sort_values("patient_id").reset_index(drop=True)


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def pool(out_dir, tag, methods=None, reference="unsegmented", n_boot=2000, n_perm=2000, seed=0):
    if methods is None:
        found = glob.glob(os.path.join(out_dir, f"pred_{tag}_*_fold*.csv"))
        methods = sorted({re.match(rf"pred_{re.escape(tag)}_(.+)_fold\d+\.csv", os.path.basename(f)).group(1) for f in found})
    data = {m: load_method(out_dir, tag, m) for m in methods}
    data = {m: d for m, d in data.items() if d is not None}
    if not data:
        raise SystemExit(f"no prediction files for tag {tag} in {out_dir}")
    ids = None
    for d in data.values():
        s = set(d["patient_id"])
        ids = s if ids is None else ids & s
    ids = sorted(ids)
    for m in data:
        data[m] = data[m][data[m]["patient_id"].isin(ids)].reset_index(drop=True)
    y = next(iter(data.values()))["true_label"].to_numpy()
    for m, d in data.items():
        if not np.array_equal(d["true_label"].to_numpy(), y):
            raise ValueError(f"{m}: labels differ from the other methods for the same patients")
    P = {m: d[PROB_COLS].to_numpy() for m, d in data.items()}
    folds = next(iter(data.values()))["fold"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))

    pooled, boots = [], {}
    for m in data:
        point = metric_row(y, P[m])
        bs = [metric_row(y[i], P[m][i]) for i in idx]
        boots[m] = pd.DataFrame(bs)
        row = {"method": m, "n_patients": n}
        for k in METRICS:
            lo, hi = np.nanpercentile(boots[m][k], [2.5, 97.5])
            row[k] = point[k]
            row[k + "_ci_lo"], row[k + "_ci_hi"] = lo, hi
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(os.path.join(out_dir, f"pooled_metrics_{tag}.csv"), index=False)

    per_fold = []
    for m in data:
        for k in sorted(set(folds)):
            msk = folds == k
            r = metric_row(y[msk], P[m][msk])
            per_fold.append({"method": m, "fold": int(k), "n_patients": int(msk.sum()), "f1_macro": r["f1_macro"],
                             "balanced_accuracy": r["balanced_accuracy"]})
    pd.DataFrame(per_fold).to_csv(os.path.join(out_dir, f"per_fold_metrics_{tag}.csv"), index=False)

    counts = [{"fold": int(k), **{c: int(((folds == k) & (y == i)).sum()) for i, c in enumerate(CLASSES)}} for k in sorted(set(folds))]
    pd.DataFrame(counts).to_csv(os.path.join(out_dir, f"class_counts_per_fold_{tag}.csv"), index=False)

    cms = {m: {"confusion_matrix": confusion_matrix(y, P[m].argmax(1), labels=[0, 1, 2]).tolist(),
               "per_class_recall": {c: float(r) for c, r in zip(CLASSES, recall_score(y, P[m].argmax(1), labels=[0, 1, 2], average=None, zero_division=0))}}
           for m in data}
    json.dump(cms, open(os.path.join(out_dir, f"confusion_matrices_{tag}.json"), "w"), indent=2)

    if reference in data:
        rows = []
        for m in data:
            if m == reference:
                continue
            ref_pt, m_pt = metric_row(y, P[reference]), metric_row(y, P[m])
            row = {"method": m, "reference": reference}
            for k in METRICS:
                d = (boots[m][k] - boots[reference][k]).to_numpy()
                row[k + "_diff"] = m_pt[k] - ref_pt[k]
                row[k + "_diff_ci_lo"], row[k + "_diff_ci_hi"] = np.nanpercentile(d, [2.5, 97.5])
            for k in PERM_METRICS:
                obs = abs(m_pt[k] - ref_pt[k])
                hits = 0
                for _ in range(n_perm):
                    sw = rng.random(n) < 0.5
                    pa = np.where(sw[:, None], P[reference], P[m])
                    pb = np.where(sw[:, None], P[m], P[reference])
                    hits += abs(metric_row(y, pa)[k] - metric_row(y, pb)[k]) >= obs - 1e-12
                row[k + "_perm_p"] = (1 + hits) / (1 + n_perm)
            rows.append(row)
        out = pd.DataFrame(rows)
        for k in PERM_METRICS:
            out[k + "_perm_p_holm"] = holm(out[k + "_perm_p"].to_numpy())
        out.to_csv(os.path.join(out_dir, f"paired_vs_{reference}_{tag}.csv"), index=False)
    cols = ["method", "n_patients", "f1_macro", "balanced_accuracy", "auc_ovr_macro", "mcc", "brier_score"]
    print("\npooled over folds (patient level):")
    print(pd.DataFrame(pooled)[cols].round(4).to_string(index=False), flush=True)
    return pooled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", "outputs"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reference", default="unsegmented")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool(a.out_dir, a.tag, None, a.reference, a.n_boot, a.n_perm, a.seed)
