import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util
import json
import numpy as np
import pydicom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
OUT = OUT_DIR
PATH = os.path.join(DATA_DIR, "lumos_x_001_280_dcm", "lumos_x_002", "lumos_x_002_1.Dcm")

ds = pydicom.dcmread(PATH)
a = ds.pixel_array.astype(np.float32)
if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
    a = a.max() - a
H, W = 800, 505
lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.7)
img = np.asarray(Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0

enh = ndi.gaussian_filter(img, 3) - ndi.gaussian_filter(img, 40)
z = (enh - enh.mean()) / enh.std()
mask = ndi.binary_opening(z > 1.2, np.ones((3, 3)))
lab, n = ndi.label(mask)
sizes = ndi.sum(mask, lab, range(1, n + 1))
mask = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 400])

yy, xx = np.mgrid[0:H, 0:W]
win = (yy >= 0.12 * H) & (yy <= 0.88 * H) & (xx >= 0.30 * W) & (xx <= 0.70 * W)


def centroid(m):
    mm = m & win
    if mm.sum() < 50:
        return 0.5, 0.5
    cy, cx = ndi.center_of_mass(mm)
    return cx / W, cy / H


def clamp_box(cx, cy, w, h):
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    ua = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / ua if ua > 0 else 0.0


def spine_band(m):
    rows = (np.arange(H) >= 0.12 * H) & (np.arange(H) <= 0.88 * H)
    prof = ndi.gaussian_filter1d(m[rows].sum(0).astype(float), 6)
    cols = (np.arange(W) >= 0.25 * W) & (np.arange(W) <= 0.75 * W)
    prof_w = np.where(cols, prof, 0)
    c0 = int(prof_w.argmax())
    lo_c = c0
    while lo_c > 0 and prof[lo_c] >= 0.5 * prof[c0]:
        lo_c -= 1
    hi_c = c0
    while hi_c < W - 1 and prof[hi_c] >= 0.5 * prof[c0]:
        hi_c += 1
    pad = int(0.1 * (hi_c - lo_c))
    lo_c, hi_c = max(lo_c - pad, 0), min(hi_c + pad, W - 1)
    return (lo_c / W, 0.17, hi_c / W, 0.83)


target = spine_band(mask)
REF = (0.32, 0.15, 0.63, 0.82)
P = ndi.gaussian_filter(mask.astype(np.float32), 4)
P = P / P.max()
BW, BH = 0.38, 0.62

boxes, trails, curves = {}, {}, {}
cx, cy = centroid(mask)
boxes["threshold + centroid"] = clamp_box(cx, cy, BW, BH)
boxes["spine-band projection"] = target
ucx, ucy = centroid(P > 0.5)
boxes["U-Net"] = clamp_box(ucx, ucy, BW, BH)


def run(name, fit_builder, d, opt):
    trails[name] = []
    f = fit_builder(name)
    rng = np.random.default_rng(11)
    v, _ = opt(f, d, 8, 5, rng)
    curves[name] = np.maximum.accumulate([t[2] for t in trails[name]])
    return v


def global_fit(name):
    def f(v):
        b = clamp_box(0.2 + 0.6 * v[0], 0.2 + 0.6 * v[1], 0.15 + 0.55 * v[2], 0.30 + 0.65 * v[3])
        s = iou(b, target)
        trails[name].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s))
        return 1.0 - s
    return f


def hybrid_box(v):
    tau = 0.3 + 0.5 * v[0]
    c = centroid(P > tau)
    return clamp_box(c[0] + (-0.15 + 0.30 * v[3]), c[1] + (-0.15 + 0.30 * v[4]), 0.15 + 0.55 * v[1], 0.30 + 0.65 * v[2])


def hybrid_fit(name):
    def f(v):
        b = hybrid_box(v)
        s = iou(b, target)
        trails[name].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s))
        return 1.0 - s
    return f


for nm, opt in (("PSO", rc.pso), ("WOA", rc.woa), ("MPA", rc.mpa)):
    v = run(nm, global_fit, 4, opt)
    boxes[nm] = clamp_box(0.2 + 0.6 * v[0], 0.2 + 0.6 * v[1], 0.15 + 0.55 * v[2], 0.30 + 0.65 * v[3])
vh = run("U-Net + PSO (hybrid)", hybrid_fit, 5, rc.pso)
boxes["U-Net + PSO (hybrid)"] = hybrid_box(vh)

order = ["threshold + centroid", "spine-band projection", "U-Net", "PSO", "WOA", "MPA", "U-Net + PSO (hybrid)"]
colors = {"threshold + centroid": "#1f77b4", "spine-band projection": "#17becf", "U-Net": "#ff7f0e",
          "PSO": "#2ca02c", "WOA": "#d62728", "MPA": "#9467bd", "U-Net + PSO (hybrid)": "#e6007e"}
group = {"threshold + centroid": "traditional", "spine-band projection": "traditional",
         "U-Net": "segmentation", "PSO": "optimizer, no mask", "WOA": "optimizer, no mask", "MPA": "optimizer, no mask",
         "U-Net + PSO (hybrid)": "proposed hybrid"}


def draw(ax, name):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if name in trails:
        t = np.array(trails[name])
        ax.scatter(t[:, 0] * W, t[:, 1] * H, c=np.arange(len(t)), cmap="viridis", s=9, alpha=0.75, zorder=3)
    b = boxes[name]
    ax.add_patch(Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H, fill=False, ec=colors[name], lw=3, zorder=4))
    ax.set_title(f"{name}\n({group[name]})", fontsize=15, color=colors[name], fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(24, 17))
gs = GridSpec(2, 4, figure=fig, hspace=0.12, wspace=0.04, left=0.02, right=0.98, top=0.93, bottom=0.04)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(img, cmap="gray", vmin=0, vmax=1)
ax.set_title("original AP radiograph\n(LUMOS patient 2)", fontsize=15, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])
for slot, nm in zip([(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], order):
    draw(fig.add_subplot(gs[slot]), nm)

fig.suptitle("Same AP lumbar spine image, every ROI strategy", fontsize=22, fontweight="bold", y=0.985)
fig.text(0.5, 0.012, "Demonstration only, no model training. U-Net panel is a stand-in (soft bone mask). White dashed = reference spine box drawn by eye. Dots = candidate ROIs tried by each optimizer (early to late).", ha="center", fontsize=13, color="#444444")
fig.savefig(os.path.join(OUT, "roi_ap_all_methods_one_slide.png"), dpi=110, facecolor="white")
json.dump({n: [round(float(v), 4) for v in boxes[n]] for n in order} | {"spine_band_target": [round(float(v), 4) for v in target]},
          open(os.path.join(OUT, "roi_ap_boxes.json"), "w"), indent=2)
print("mask pixels:", int(mask.sum()), "| target band:", [round(float(v), 3) for v in target])
