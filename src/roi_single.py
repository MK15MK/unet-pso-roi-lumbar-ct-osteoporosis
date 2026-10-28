import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
rc.require_gpu()
DEV, S = rc.DEV, rc.STORE
OUT = OUT_DIR

df, labels = rc.read_labels(DATA_DIR)
ct = rc.find_ct_patients(DATA_DIR)
PID = 1
files = rc.pick_central(ct[PID])
arr = rc.slice_to_u8(files[len(files) // 2])
xg = torch.from_numpy(arr)[None, None].to(DEV)
cls = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}[labels[PID]]

hu = arr.astype(np.float32) / 255.0 * 1200.0 - 200.0
raw = torch.from_numpy((hu > rc.BONE_HU).astype(np.float32))[None, None].to(DEV)
eroded = -F.max_pool2d(-raw, 3, 1, 1)
opened = F.max_pool2d(eroded, 3, 1, 1)
mask = opened[0, 0].cpu().numpy() > 0.5
win = np.zeros((S, S), bool)
win[int(0.2 * S):int(0.8 * S), int(0.25 * S):int(0.75 * S)] = True
inside = mask & win

soft = ndi.gaussian_filter(mask.astype(np.float32), 2.0)
P224 = F.interpolate(torch.from_numpy(soft)[None, None], size=(rc.IMG, rc.IMG), mode="bilinear", align_corners=False)
P = (P224 * 255).round().clamp(0, 255).to(torch.uint8)

ys, xs = np.where(inside)
bx0, bx1, by0, by1 = xs.min() / S, (xs.max() + 1) / S, ys.min() / S, (ys.max() + 1) / S
target_area = (bx1 - bx0) * (by1 - by0)


def box_of(kind, r, pc):
    cx, cy, half = rc.geometry(kind, xg, pc, r)
    cx, cy, half = cx.item(), cy.item(), half.item()
    return cx - half, cy - half, cx + half, cy + half


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


target_box = (bx0, by0, bx1, by1)
trails, curves, finals = {}, {}, {}


def make_fitness(name):
    trails[name] = []

    def fitness(v):
        r = rc.decode_roi(v)
        b = box_of("unet", r, P)
        s = iou(b, target_box)
        trails[name].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s))
        return 1.0 - s
    return fitness


for name, opt in rc.OPTIMIZERS.items():
    rng = np.random.default_rng(11)
    best_v, best_f = opt(make_fitness(name), 4, 8, 5, rng)
    finals[name] = rc.decode_roi(best_v)
    run = np.array([t[2] for t in trails[name]])
    curves[name] = np.maximum.accumulate(run)

methods = ["unsegmented", "rule_centroid", "unet_proxy", "pso_roi", "woa_roi", "mpa_roi"]
colors = {"unsegmented": "#888888", "rule_centroid": "#1f77b4", "unet_proxy": "#ff7f0e", "pso_roi": "#2ca02c",
          "woa_roi": "#d62728", "mpa_roi": "#9467bd"}
boxes = {"unsegmented": (0.0, 0.0, 1.0, 1.0), "rule_centroid": box_of("rule", rc.DEFAULT_ROI, None),
         "unet_proxy": box_of("unet", rc.DEFAULT_ROI, P)}
params = {"unsegmented": {}, "rule_centroid": rc.DEFAULT_ROI, "unet_proxy": rc.DEFAULT_ROI}
for n in rc.OPTIMIZERS:
    boxes[n] = box_of("unet", finals[n], P)
    params[n] = finals[n]


def crop_of(m):
    kind = {"unsegmented": "unsegmented", "rule_centroid": "rule"}.get(m, "unet")
    return rc.make_crops(kind, xg.cpu(), P, params[m])[0, 0].cpu().numpy()


fig, axes = plt.subplots(2, 6, figsize=(21, 7.6))
for ci, m in enumerate(methods):
    a1, a2 = axes[0, ci], axes[1, ci]
    a1.imshow(arr, cmap="gray", vmin=0, vmax=255)
    b = boxes[m]
    if m != "unsegmented":
        if m in trails:
            t = np.array(trails[m])
            a1.scatter(t[:, 0] * S, t[:, 1] * S, c=np.arange(len(t)), cmap="viridis", s=14, alpha=0.8, zorder=3)
        a1.add_patch(Rectangle((b[0] * S, b[1] * S), (b[2] - b[0]) * S, (b[3] - b[1]) * S, fill=False, ec=colors[m], lw=2.8, zorder=4))
    a1.set_title(m.replace("_", " "), fontsize=12, color=colors[m], fontweight="bold")
    a2.imshow(crop_of(m), cmap="gray", vmin=0, vmax=255)
    side = (b[2] - b[0])
    a2.set_xlabel(f"box side {side:.2f} of the slice" if m != "unsegmented" else "whole slice, no ROI", fontsize=10)
    for a in (a1, a2):
        a.set_xticks([])
        a.set_yticks([])
axes[0, 0].set_ylabel(f"patient {PID} ({cls})\nCT slice with ROI", fontsize=11)
axes[1, 0].set_ylabel("crop fed to\nthe classifier", fontsize=11)
fig.suptitle("Same slice, six ROI strategies (dots on the optimizer panels = the candidate ROI centres each algorithm tried, early to late)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "roi_single_slice_all_methods.png"), dpi=125, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(17.5, 9))
panels = [("a  CT slice", None), ("b  HU > 150 threshold (raw)", raw[0, 0].cpu().numpy() > 0.5), ("c  after erosion (3x3)", eroded[0, 0].cpu().numpy() > 0.5),
          ("d  after dilation = bone mask", mask), ("e  U-Net view: soft bone\nprobability (stand-in)", soft),
          ("f  keep bone inside the\ncentral window", inside), ("g  centre of mass and ROI box", None), ("h  the crop", None)]
for ax, (title, layer) in zip(axes.ravel(), panels):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    if layer is not None and layer.dtype == bool:
        ov = np.zeros((S, S, 4))
        ov[layer] = [1, 0.15, 0.15, 0.6]
        ax.imshow(ov)
    elif layer is not None:
        ax.imshow(layer, cmap="inferno", alpha=0.65, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
ax = axes.ravel()[5]
ax.add_patch(Rectangle((0.25 * S, 0.2 * S), 0.5 * S, 0.6 * S, fill=False, ec="#FFD400", lw=2, ls="--"))
ax = axes.ravel()[6]
b = boxes["rule_centroid"]
cxm, cym = (b[0] + b[2]) / 2 * S, (b[1] + b[3]) / 2 * S
ov = np.zeros((S, S, 4))
ov[inside] = [1, 0.15, 0.15, 0.45]
ax.imshow(ov)
ax.add_patch(Rectangle((b[0] * S, b[1] * S), (b[2] - b[0]) * S, (b[3] - b[1]) * S, fill=False, ec="#00E5FF", lw=2.8))
ax.plot(cxm, cym, "o", color="#00E5FF", ms=10, mec="black")
axes.ravel()[7].clear()
axes.ravel()[7].imshow(crop_of("rule_centroid"), cmap="gray", vmin=0, vmax=255)
axes.ravel()[7].set_title("h  the crop", fontsize=11, fontweight="bold")
axes.ravel()[7].set_xticks([])
axes.ravel()[7].set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "roi_masking_explained.png"), dpi=125, facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
for n, c in curves.items():
    ax.plot(np.arange(1, len(c) + 1), c, color=colors[n], lw=2.4, label=n.replace("_", " "))
ax.set_xlabel("candidate ROIs evaluated")
ax.set_ylabel("best IoU with the bone extent so far")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Same budget (48 evaluations each), same slice and objective", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "roi_optimizer_convergence.png"), dpi=130, facecolor="white")
plt.close(fig)

bone_in = lambda b: mask[int(b[1] * S):int(np.ceil(b[3] * S)), int(b[0] * S):int(np.ceil(b[2] * S))].sum() / max(mask.sum(), 1)
stats = []
for m in methods:
    b = boxes[m]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    stats.append({"method": m, "side": round(b[2] - b[0], 3), "centre_x": round(cx, 3), "centre_y": round(cy, 3),
                  "fraction_of_bone_mask_inside": round(float(bone_in(b)), 3), "iou_with_bone_extent": round(iou(b, target_box), 3),
                  "params": {k: round(v, 3) for k, v in params[m].items()}})
M = np.array([[iou(boxes[a], boxes[b]) for b in methods] for a in methods])
json.dump({"patient": PID, "label": cls, "stats": stats, "pairwise_box_iou": M.round(3).tolist(), "methods": methods},
          open(os.path.join(OUT, "roi_single_slice_stats.json"), "w"), indent=2)
for s in stats:
    print(s)
print("pairwise IoU between boxes:\n", np.round(M, 2))
print("bone-extent box (target):", [round(v, 3) for v in target_box], "| mask pixels:", int(mask.sum()))
