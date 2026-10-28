import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.ops import roi_align
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG, STORE = 224, 256
CT_WINDOW = (-200.0, 1000.0)
BONE_HU = 150.0
MEAN, STD = 0.449, 0.226
LABEL_SMOOTHING = 0.1
DEV = torch.device("cuda")
METHODS = ["unsegmented", "rule_centroid", "unet_default", "pso_roi", "woa_roi", "mpa_roi"]
COLORS = {"unsegmented": "#888888", "rule_centroid": "#1f77b4", "unet_default": "#ff7f0e", "pso_roi": "#2ca02c",
          "woa_roi": "#d62728", "mpa_roi": "#9467bd"}
DEFAULT_ROI = dict(tau=0.5, side=0.55, dx=0.0, dy=0.0)
CLIN_TABLE, CLIN_NAMES = {}, []


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available")
    z = torch.randn(64, 64, device="cuda")
    float((z @ z).sum().item())
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)


def read_labels(data_dir):
    xl = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".xlsx"))
    if not xl:
        raise FileNotFoundError(f"no .xlsx in {data_dir}")
    df = pd.read_excel(os.path.join(data_dir, xl[0]))
    labels = {}
    for pid, lab in zip(df["patient_id"], df["Osteoporosis"]):
        m = re.search(r"\d+", str(pid))
        if m and not pd.isna(lab):
            labels[int(m.group())] = int(lab)
    return df, labels


def find_ct_patients(data_dir):
    pts = {}
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        m = re.fullmatch(r"lumos_ct_0*(\d+)", os.path.basename(root), re.I)
        if m:
            dcm = [os.path.join(root, f) for f in files if f.lower().endswith(".dcm") and not f.startswith("._")]
            if dcm:
                pts.setdefault(int(m.group(1)), []).extend(dcm)
    return pts


def pick_central(files):
    metas = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        iop, ipp = getattr(ds, "ImageOrientationPatient", None), getattr(ds, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
            normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            key = float(np.dot(np.array(ipp, float), normal))
        else:
            key = float(getattr(ds, "InstanceNumber", 0) or 0)
        metas.append((key, os.path.basename(f), f))
    metas.sort()
    n = len(metas)
    lo, hi = int(0.2 * n), int(0.8 * n)
    if hi <= lo:
        lo, hi = 0, n
    return [m[2] for m in metas[lo:hi]]


def slice_to_u8(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if arr.ndim > 2:
        arr = arr[0]
    lo, hi = CT_WINDOW
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    t = F.interpolate(torch.from_numpy(arr)[None, None], size=(STORE, STORE), mode="bilinear", align_corners=False)
    return (t[0, 0] * 255).round().clamp(0, 255).to(torch.uint8).numpy()


def load_slices(args):
    tag = f"_{args.subset_patients}" if args.subset_patients > 0 else ""
    cache = os.path.join(args.out_dir, f"cache_ct_slices{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        print(f"loaded cached slices from {cache}", flush=True)
        return z["X"], z["y"], z["pids"]
    _, labels = read_labels(args.data_dir)
    ct = find_ct_patients(args.data_dir)
    pids = sorted(set(labels) & set(ct))
    if args.subset_patients > 0:
        rs = np.random.default_rng(args.seed)
        per = args.subset_patients // 3
        chosen = []
        for c in (0, 1, 2):
            pool = [p for p in pids if labels[p] == c]
            chosen += list(rs.choice(pool, min(per, len(pool)), replace=False))
        pids = sorted(int(p) for p in chosen)
    print(f"patients with label and CT: {len(pids)}", flush=True)

    def work(pid):
        return pid, [slice_to_u8(p) for p in pick_central(ct[pid])]

    imgs, ys, ids, seen = [], [], [], set()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for pid, arrs in ex.map(work, pids):
            for a in arrs:
                h = hashlib.sha1(a.tobytes()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                imgs.append(a)
                ys.append(labels[pid])
                ids.append(pid)
            done += 1
            if done % 20 == 0 or done == len(pids):
                print(f"  read {done}/{len(pids)} patients ({len(imgs)} unique slices)", flush=True)
    X = np.stack(imgs)
    y, ids = np.array(ys), np.array(ids)
    np.savez(cache, X=X, y=y, pids=ids)
    return X, y, ids


def load_clinical(args, used):
    global CLIN_TABLE, CLIN_NAMES
    df, _ = read_labels(args.data_dir)
    cols = ["age", "gender", "height", "weight", "BMI"]
    sub = df[["patient_id"] + cols].copy()
    sub["_pid"] = sub["patient_id"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    sub = sub.dropna(subset=["_pid"]).drop_duplicates("_pid").set_index(sub["_pid"].astype(int))
    enc = pd.get_dummies(sub[cols], drop_first=True, dtype=float)
    enc = enc.loc[:, enc.std(skipna=True) > 0]
    CLIN_NAMES = list(enc.columns)
    CLIN_TABLE = {int(p): enc.loc[p].to_numpy(dtype=float) for p in used}
    print(f"clinical input: {CLIN_NAMES}", flush=True)


def make_fold(pids, y, k, folds, seed):
    uniq = np.array(sorted(set(pids.tolist())))
    lab_of = {}
    for p, l in zip(pids.tolist(), y.tolist()):
        lab_of.setdefault(p, l)
    lab = np.array([lab_of[p] for p in uniq])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(skf.split(uniq, lab))
    rest_i, te_i = splits[k]
    rest, te = uniq[rest_i], uniq[te_i]
    tr, va = train_test_split(rest, test_size=0.1765, stratify=lab[rest_i], random_state=seed)
    return tr, va, te


def build_fold(X, y, pids, parts):
    tr, va, te = parts
    out = {}
    for name, ps in (("tr", tr), ("va", va), ("te", te)):
        idx = np.where(np.isin(pids, ps))[0]
        out["X" + name] = torch.from_numpy(X[idx])[:, None]
        out["y" + name] = torch.from_numpy(y[idx]).long().to(DEV)
        out["p" + name] = pids[idx]
    if CLIN_TABLE:
        M = np.stack([CLIN_TABLE[int(p)] for p in tr]).astype(np.float64)
        med = np.nanmedian(M, 0)
        M = np.where(np.isnan(M), med, M)
        mu, sd = M.mean(0), M.std(0)
        sd[sd < 1e-8] = 1.0
        for name in ("tr", "va", "te"):
            A = np.stack([CLIN_TABLE[int(p)] for p in out["p" + name]]).astype(np.float64)
            A = np.where(np.isnan(A), med, A)
            out["C" + name] = torch.from_numpy(((A - mu) / sd).astype(np.float32)).to(DEV)
    print(f"fold sizes: train={len(out['ytr'])} ({len(tr)} pts) val={len(out['yva'])} ({len(va)} pts) "
          f"test={len(out['yte'])} ({len(te)} pts)", flush=True)
    return out


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.e1, self.e2, self.e3, self.e4, self.bott = blk(1, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3]), blk(c[3], c[4])
        self.up4, self.d4 = nn.ConvTranspose2d(c[4], c[3], 2, 2), blk(c[3] * 2, c[3])
        self.up3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), blk(c[2] * 2, c[2])
        self.up2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), blk(c[1] * 2, c[1])
        self.up1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), blk(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.bott(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.up4(b), e4], 1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class ViTCNN(nn.Module):
    def __init__(self, dim, depth, heads, dropout, n_cls, clin_dim=0):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        self.frozen = nn.ModuleList([m.conv1, m.bn1, m.layer1, m.layer2])
        for p in self.frozen.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(256, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 14 * 14 + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.clin = None
        if clin_dim > 0:
            self.clin = nn.Sequential(nn.Linear(clin_dim, 32), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                      nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim + (32 if clin_dim > 0 else 0), n_cls))

    def set_train(self):
        self.train()
        for m in self.frozen:
            m.eval()

    def forward(self, x, c=None):
        x = (x * 0.5 + 0.5 - MEAN) / STD
        x = x.repeat(1, 3, 1, 1)
        t = self.stem(x).flatten(2).transpose(1, 2)
        t = self.proj(t)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1) + self.pos
        t = self.enc(t)
        f = self.norm(t[:, 0])
        if self.clin is not None:
            f = torch.cat([f, self.clin(c)], 1)
        return self.head(f)


def prep(xb):
    x = xb.float().div_(255.0)
    if x.shape[-1] != IMG:
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def bone_mask_u8(xg):
    hu = xg.float() / 255.0 * (CT_WINDOW[1] - CT_WINDOW[0]) + CT_WINDOW[0]
    m = (hu > BONE_HU).float()
    m = -F.max_pool2d(-m, 3, 1, 1)
    return F.max_pool2d(m, 3, 1, 1)


def seg_target(xb):
    m = F.interpolate(bone_mask_u8(xb), size=(IMG, IMG), mode="bilinear", align_corners=False)
    return (m > 0.5).float()


def seg_loss(logit, t):
    bce = F.binary_cross_entropy_with_logits(logit, t)
    p = torch.sigmoid(logit)
    inter = (p * t).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def unet_dice(net, X, bs=64):
    net.eval()
    tp = fp = fn = 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(DEV)
        pr = (net(prep(xb)).float() > 0).float()
        t = seg_target(xb)
        tp += (pr * t).sum().item()
        fp += (pr * (1 - t)).sum().item()
        fn += ((1 - pr) * t).sum().item()
    return 2 * tp / (2 * tp + fp + fn + 1e-6)


def train_unet(fold, args, path):
    net = UNet().to(DEV)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=DEV))
        print(f"loaded U-Net weights {path}", flush=True)
        return net.eval()
    torch.manual_seed(args.seed + 1000)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = fold["Xtr"]
    n, bs = len(Xtr), args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.unet_epochs * math.ceil(n / bs))
    best, best_state = -1.0, None
    Xtr_g = Xtr.to(DEV)
    for ep in range(1, args.unet_epochs + 1):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            xb = Xtr_g[perm[i:i + bs]]
            loss = seg_loss(net(prep(xb)).float(), seg_target(xb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        d = unet_dice(net, fold["Xva"])
        print(f"U-Net epoch {ep}/{args.unet_epochs} val_dice={d:.4f}", flush=True)
        if d > best:
            best, best_state = d, {k: v.detach().clone() for k, v in net.state_dict().items()}
    del Xtr_g
    net.load_state_dict(best_state)
    torch.save(best_state, path)
    return net.eval()


@torch.no_grad()
def unet_probs(net, X, bs=64):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        p = torch.sigmoid(net(prep(X[i:i + bs].to(DEV))).float())
        out.append((p * 255).round().to(torch.uint8).cpu())
    return torch.cat(out)


def _windows(size):
    xs = torch.linspace(0, 1, size, device=DEV)
    win = ((xs >= 0.20) & (xs <= 0.80)).float()[:, None] * ((xs >= 0.25) & (xs <= 0.75)).float()[None, :]
    return win, xs


def centre_of_mass(m, size):
    win, xs = _windows(size)
    m = m * win
    mass = m.sum((1, 2))
    ok = mass >= 50
    cx = (m * xs[None, None, :]).sum((1, 2)) / mass.clamp(min=1)
    cy = (m * xs[None, :, None]).sum((1, 2)) / mass.clamp(min=1)
    half = torch.full_like(cx, 0.5)
    return torch.where(ok, cx, half), torch.where(ok, cy, half)


def geometry(kind, xg, pc, r):
    n = len(xg)
    if kind == "unsegmented":
        c = torch.full((n,), 0.5, device=DEV)
        return c, c.clone(), torch.full((n,), 0.5, device=DEV)
    if kind == "rule":
        cx, cy = centre_of_mass(bone_mask_u8(xg)[:, 0], STORE)
    else:
        p = pc.to(DEV).float()[:, 0] / 255.0
        cx, cy = centre_of_mass((p > r["tau"]).float(), IMG)
    half = torch.full((n,), r["side"] / 2, device=DEV)
    cx = torch.minimum(torch.maximum(cx + r["dx"], half), 1 - half)
    cy = torch.minimum(torch.maximum(cy + r["dy"], half), 1 - half)
    return cx, cy, half


@torch.no_grad()
def make_crops(kind, X, P, r, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xg = X[i:i + bs].to(DEV)
        if kind == "unsegmented":
            out.append(F.interpolate(xg.float(), size=(IMG, IMG), mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8))
            continue
        cx, cy, half = geometry(kind, xg, None if P is None else P[i:i + bs], r)
        S = float(STORE)
        boxes = torch.stack([torch.arange(len(cx), device=DEV, dtype=torch.float32), (cx - half) * S, (cy - half) * S,
                             (cx + half) * S, (cy + half) * S], 1)
        crop = roi_align(xg.float(), boxes, (IMG, IMG), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        out.append(crop.round().clamp(0, 255).to(torch.uint8))
    return torch.cat(out)


def decode_roi(v):
    return dict(tau=0.30 + 0.40 * v[0], side=0.30 + 0.60 * v[1], dx=-0.25 + 0.50 * v[2], dy=-0.35 + 0.50 * v[3])


def class_metrics(y_true, probs):
    pred = probs.argmax(1)
    n_cls = probs.shape[1]
    onehot = np.eye(n_cls)[y_true]
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(n_cls)))
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(y_true, pred), "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0), "auc_ovr_macro": auc,
            "mcc": matthews_corrcoef(y_true, pred), "cohen_kappa": cohen_kappa_score(y_true, pred),
            "brier_score": float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))}


@torch.no_grad()
def forward_logits(model, X, C=None, bs=128):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(prep(X[i:i + bs]), None if C is None else C[i:i + bs]).float())
    return torch.cat(out)


def patient_pool(probs, y, slice_pids):
    uniq = np.unique(slice_pids)
    p = np.stack([probs[slice_pids == u].mean(0) for u in uniq])
    yy = np.array([y[slice_pids == u][0] for u in uniq])
    return uniq, yy, p


def train_classifier(data, epochs, args, seed, quiet=False):
    torch.manual_seed(seed)
    clin_dim = data["Ctr"].shape[1] if "Ctr" in data else 0
    model = ViTCNN(96, 2, 4, args.dropout, 3, clin_dim).to(DEV)
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("stem.")]
    stem_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("stem.")]
    opt = torch.optim.AdamW([{"params": head_p, "lr": args.lr}, {"params": stem_p, "lr": args.lr * 0.1}], weight_decay=args.wd)
    Xtr, ytr, Ctr = data["Xtr"], data["ytr"], data.get("Ctr")
    n, bs = len(ytr), args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 0.1], total_steps=epochs * math.ceil(n / bs))
    best_loss, best_state, best_ep = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.set_train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            yb = ytr[idx]
            logits = model(prep(Xtr[idx]), None if Ctr is None else Ctr[idx])
            loss = F.cross_entropy(logits, yb, label_smoothing=LABEL_SMOOTHING)
            pr = torch.softmax(logits.float(), 1)
            dist = (torch.arange(3, device=DEV)[None, :] - yb[:, None]).abs().float()
            loss = loss + 0.5 * (pr * dist).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        lv = forward_logits(model, data["Xva"], data.get("Cva"))
        vl = F.cross_entropy(lv, data["yva"]).item()
        va = (lv.argmax(1) == data["yva"]).float().mean().item()
        if not quiet:
            print(f"  epoch {ep}/{epochs} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        if vl < best_loss:
            best_loss, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_ep, best_loss


def pso(f, d, n, T, rng):
    X = rng.random((n, d))
    V = (rng.random((n, d)) - 0.5) * 0.2
    P = X.copy()
    pf = np.array([f(x) for x in X])
    g, gf = P[pf.argmin()].copy(), pf.min()
    for t in range(T):
        w = 0.9 - (0.9 - 0.4) * t / T
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        V = np.clip(w * V + 1.5 * r1 * (P - X) + 1.5 * r2 * (g - X), -0.2, 0.2)
        X = np.clip(X + V, 0, 1)
        fx = np.array([f(x) for x in X])
        better = fx < pf
        P[better], pf[better] = X[better], fx[better]
        if pf.min() < gf:
            gf, g = pf.min(), P[pf.argmin()].copy()
    return g, gf


def woa(f, d, n, T, rng, b=1.0):
    X = rng.random((n, d))
    fx = np.array([f(x) for x in X])
    best, gf = X[fx.argmin()].copy(), fx.min()
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(n):
            r1, r2 = rng.random(2)
            A, C = 2 * a * r1 - a, 2 * r2
            p, l = rng.random(), rng.uniform(-1, 1)
            if p < 0.5:
                if abs(A) < 1:
                    cand = best - A * np.abs(C * best - X[i])
                else:
                    rnd = X[rng.integers(n)]
                    cand = rnd - A * np.abs(C * rnd - X[i])
            else:
                cand = np.abs(best - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            cand = np.clip(cand, 0, 1)
            X[i], fx[i] = cand, f(cand)
        if fx.min() < gf:
            gf, best = fx.min(), X[fx.argmin()].copy()
    return best, gf


def levy(d, rng, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(d) * sig
    v = rng.standard_normal(d)
    return 0.05 * u / np.abs(v) ** (1 / beta)


def mpa(f, d, n, T, rng, FADs=0.2, Pm=0.5):
    X = rng.random((n, d))
    fx = np.array([f(x) for x in X])
    elite, gf = X[fx.argmin()].copy(), fx.min()
    for t in range(T):
        CF = (1 - t / T) ** (2 * t / T)
        for i in range(n):
            R = rng.random(d)
            RB = rng.standard_normal(d)
            if t < T / 3:
                cand = X[i] + Pm * R * (RB * (elite - RB * X[i]))
            elif t < 2 * T / 3:
                if i < n / 2:
                    RL = levy(d, rng)
                    cand = X[i] + Pm * R * (RL * (elite - RL * X[i]))
                else:
                    cand = elite + Pm * CF * (RB * (RB * elite - X[i]))
            else:
                cand = elite + Pm * CF * (RB * (RB * elite - X[i]))
            if rng.random() < FADs:
                U = (rng.random(d) < FADs).astype(float)
                cand = cand + CF * rng.random(d) * U
            else:
                r = rng.random()
                r1, r2 = rng.integers(n, size=2)
                cand = cand + (FADs * (1 - r) + r) * (X[r1] - X[r2])
            cand = np.clip(cand, 0, 1)
            fc = f(cand)
            if fc < fx[i]:
                X[i], fx[i] = cand, fc
        if fx.min() < gf:
            gf, elite = fx.min(), X[fx.argmin()].copy()
    return elite, gf


OPTIMIZERS = {"pso_roi": pso, "woa_roi": woa, "mpa_roi": mpa}


def run_search(name, fold, P, args, out_path):
    if os.path.exists(out_path):
        res = json.load(open(out_path))
        print(f"loaded search result for {name}: {res['roi']}", flush=True)
        return res
    itr = np.arange(0, len(fold["ytr"]), args.proxy_train_stride)
    iva = np.arange(0, len(fold["yva"]), args.proxy_val_stride)
    Xtr_s, Xva_s = fold["Xtr"][torch.from_numpy(itr)], fold["Xva"][torch.from_numpy(iva)]
    Ptr_s, Pva_s = P["tr"][torch.from_numpy(itr)], P["va"][torch.from_numpy(iva)]
    ytr_s, yva_s = fold["ytr"][torch.from_numpy(itr).to(DEV)], fold["yva"][torch.from_numpy(iva).to(DEV)]
    Ctr_s = fold["Ctr"][torch.from_numpy(itr).to(DEV)] if "Ctr" in fold else None
    Cva_s = fold["Cva"][torch.from_numpy(iva).to(DEV)] if "Cva" in fold else None
    cache, log = {}, []

    def fitness(v):
        key = tuple(np.round(v, 3))
        if key in cache:
            return cache[key]
        r = decode_roi(v)
        data = {"Xtr": make_crops("unet", Xtr_s, Ptr_s, r), "ytr": ytr_s, "Xva": make_crops("unet", Xva_s, Pva_s, r), "yva": yva_s}
        if Ctr_s is not None:
            data["Ctr"], data["Cva"] = Ctr_s, Cva_s
        model, _, _ = train_classifier(data, args.proxy_epochs, args, args.seed, quiet=True)
        lv = forward_logits(model, data["Xva"], data.get("Cva"))
        pr = lv.argmax(1).cpu().numpy()
        f1 = f1_score(yva_s.cpu().numpy(), pr, average="macro", zero_division=0)
        del model, data
        torch.cuda.empty_cache()
        cache[key] = 1.0 - f1
        log.append({"candidate": len(log) + 1, "val_f1_macro": f1, **{a: float(b) for a, b in r.items()}})
        print(f"    {name} candidate {len(log):3d}  val_f1={f1:.4f}  " + "  ".join(f"{a}={b:.3f}" for a, b in r.items()), flush=True)
        return 1.0 - f1

    rng = np.random.default_rng(args.seed + 5)
    best_v, best_f = OPTIMIZERS[name](fitness, 4, args.pop, args.iters, rng)
    res = {"method": name, "roi": {a: float(b) for a, b in decode_roi(best_v).items()}, "best_val_f1": 1.0 - float(best_f),
           "evaluations": len(log), "log": log}
    json.dump(res, open(out_path, "w"), indent=2)
    return res


def draw_figure(fold_data, P_test, params, args, path, title_methods):
    te_p, te_y = fold_data["pte"], fold_data["yte"].cpu().numpy()
    picks = []
    for cls in (0, 1, 2):
        pat = [p for p in np.unique(te_p) if te_y[te_p == p][0] == cls]
        if pat:
            idx = np.where(te_p == pat[0])[0]
            picks.append(int(idx[len(idx) // 2]))
    labels = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}
    rows = len(picks)
    fig, axes = plt.subplots(2 * rows, len(title_methods), figsize=(2.6 * len(title_methods), 2.7 * 2 * rows))
    axes = np.atleast_2d(axes)
    for ri, si in enumerate(picks):
        xg = fold_data["Xte"][si:si + 1]
        pc = P_test[si:si + 1]
        for ci, m in enumerate(title_methods):
            kind = {"unsegmented": "unsegmented", "rule_centroid": "rule"}.get(m, "unet")
            cx, cy, half = geometry(kind, xg.to(DEV), pc, params[m])
            a1, a2 = axes[2 * ri, ci], axes[2 * ri + 1, ci]
            img = xg[0, 0].numpy()
            a1.imshow(img, cmap="gray", vmin=0, vmax=255)
            if m != "unsegmented":
                S = STORE
                a1.add_patch(plt.Rectangle(((cx - half).item() * S, (cy - half).item() * S), 2 * half.item() * S, 2 * half.item() * S,
                                           fill=False, ec=COLORS[m], lw=2.2))
                a1.plot(cx.item() * S, cy.item() * S, "+", color=COLORS[m], ms=9)
            crop = make_crops(kind, xg, pc, params[m])[0, 0].cpu().numpy()
            a2.imshow(crop, cmap="gray", vmin=0, vmax=255)
            for a in (a1, a2):
                a.set_xticks([])
                a.set_yticks([])
            if ri == 0:
                a1.set_title(m, fontsize=9, color=COLORS[m], fontweight="bold")
            if ci == 0:
                a1.set_ylabel(f"{labels[int(te_y[si])]}\nslice", fontsize=8)
                a2.set_ylabel("crop fed to\nthe classifier", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("LUMOS_DIR", "data/LUMOS"))
    ap.add_argument("--out_dir", default=os.environ.get("OUT_DIR", "outputs"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--unet_epochs", type=int, default=10)
    ap.add_argument("--final_epochs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--proxy_epochs", type=int, default=4)
    ap.add_argument("--proxy_train_stride", type=int, default=4)
    ap.add_argument("--proxy_val_stride", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clinical", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--subset_patients", type=int, default=0)
    ap.add_argument("--figure_only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.unet_epochs, args.final_epochs, args.pop, args.iters, args.proxy_epochs = 1, 1, 2, 1, 1
        args.proxy_train_stride = args.proxy_val_stride = 1
        args.out_dir = os.path.join(args.out_dir, "smoke")
    if args.quick:
        args.subset_patients, args.figure_only = 90, True
        args.unet_epochs, args.pop, args.iters = 6, 4, 2
        args.proxy_epochs, args.proxy_train_stride, args.proxy_val_stride = 2, 2, 1
        args.out_dir = os.path.join(args.out_dir, "quick")
    os.makedirs(args.out_dir, exist_ok=True)
    require_gpu()
    X, y, pids = load_slices(args)
    if args.smoke:
        X, y, pids = X[::15], y[::15], pids[::15]
    if args.clinical:
        load_clinical(args, sorted(set(int(p) for p in pids.tolist())))
    parts = make_fold(pids, y, args.fold, args.folds, args.seed)
    fold = build_fold(X, y, pids, parts)
    for k in ("Ctr", "Cva", "Cte"):
        if k in fold:
            fold[k] = fold[k].to(DEV)
    net = train_unet(fold, args, os.path.join(args.out_dir, f"unet_fold{args.fold}.pt"))
    P = {s: unet_probs(net, fold["X" + s]) for s in ("tr", "va", "te")}
    del net
    torch.cuda.empty_cache()

    methods = [m for m in args.methods.split(",") if m]
    params = {"unsegmented": {}, "rule_centroid": dict(DEFAULT_ROI), "unet_default": dict(DEFAULT_ROI)}
    searches = {}
    for name in OPTIMIZERS:
        if name in methods:
            print(f"\n=== search: {name} (pop {args.pop}, iters {args.iters}) ===", flush=True)
            searches[name] = run_search(name, fold, P, args, os.path.join(args.out_dir, f"search_{name}_fold{args.fold}.json"))
            params[name] = searches[name]["roi"]
    for name in OPTIMIZERS:
        params.setdefault(name, dict(DEFAULT_ROI))
    draw_figure(fold, P["te"], params, args, os.path.join(args.out_dir, f"roi_same_slice_comparison_fold{args.fold}.png"), methods)
    print("saved the same-slice comparison figure", flush=True)
    json.dump(params, open(os.path.join(args.out_dir, f"roi_parameters_fold{args.fold}.json"), "w"), indent=2)
    if args.figure_only:
        return

    rows = []
    for m in methods:
        res_path = os.path.join(args.out_dir, f"result_{m}_fold{args.fold}.json")
        if os.path.exists(res_path):
            rows.append(json.load(open(res_path)))
            print(f"loaded result for {m}", flush=True)
            continue
        print(f"\n=== final ViT-CNN: {m} ({args.final_epochs} epochs) ===", flush=True)
        t0 = time.time()
        kind = {"unsegmented": "unsegmented", "rule_centroid": "rule"}.get(m, "unet")
        data = {"Xtr": make_crops(kind, fold["Xtr"], P["tr"], params[m]), "ytr": fold["ytr"],
                "Xva": make_crops(kind, fold["Xva"], P["va"], params[m]), "yva": fold["yva"]}
        xte = make_crops(kind, fold["Xte"], P["te"], params[m])
        for k in ("Ctr", "Cva"):
            if k in fold:
                data[k] = fold[k]
        model, best_ep, best_loss = train_classifier(data, args.final_epochs, args, args.seed)
        lt = forward_logits(model, xte, fold.get("Cte"))
        probs = torch.softmax(lt, 1).cpu().numpy()
        _, yy, pp = patient_pool(probs, fold["yte"].cpu().numpy(), fold["pte"])
        met = class_metrics(yy, pp)
        row = {"method": m, "best_epoch": best_ep, "best_val_loss": best_loss, "minutes": round((time.time() - t0) / 60, 1),
               "roi": params[m], **{k: float(v) for k, v in met.items()}}
        json.dump(row, open(res_path, "w"), indent=2)
        rows.append(row)
        print(f"  patient-level test: macro-F1={met['f1_macro']:.4f} acc={met['accuracy']:.4f} auc={met['auc_ovr_macro']:.4f}", flush=True)
        del data, xte, model
        gc.collect()
        torch.cuda.empty_cache()
        flat = [{k: v for k, v in r.items() if k != "roi"} for r in rows]
        pd.DataFrame(flat).to_csv(os.path.join(args.out_dir, f"roi_comparison_fold{args.fold}.csv"), index=False)
    flat = [{k: v for k, v in r.items() if k != "roi"} for r in rows]
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(args.out_dir, f"roi_comparison_fold{args.fold}.csv"), index=False)
    print("\n" + df[["method", "accuracy", "f1_macro", "balanced_accuracy", "auc_ovr_macro", "mcc", "cohen_kappa", "brier_score"]]
          .round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
