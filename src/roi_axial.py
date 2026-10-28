import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
OUT = OUT_DIR
df, labels = rc.read_labels(DATA_DIR)
ct = rc.find_ct_patients(DATA_DIR)
PID = 1
files = rc.pick_central(ct[PID])
arr = rc.slice_to_u8(files[len(files) // 2])
S = arr.shape[0]
img = arr.astype(np.float32) / 255.0
hu = img * 1200.0 - 200.0
mask = ndi.binary_opening(hu > rc.BONE_HU, np.ones((3, 3)))
mask = ndi.binary_fill_holes(ndi.binary_closing(mask, np.ones((7, 7))))
lab, n = ndi.label(mask)
keep = [i + 1 for i in range(n) if ndi.sum(mask, lab, i + 1) >= 300]
mask = np.isin(lab, keep)
yy, xx = np.mgrid[0:S, 0:S]
win = (yy >= 0.2 * S) & (yy <= 0.8 * S) & (xx >= 0.25 * S) & (xx <= 0.75 * S)


def centroid(m):
    mm = m & win
    if mm.sum() < 20:
        return 0.5, 0.5
    cy, cx = ndi.center_of_mass(mm)
    return cx / S, cy / S


def clamp_box(cx, cy, w, h):
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    i = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
    return i / u if u > 0 else 0.0


def half_extent(prof, lo, hi):
    p = ndi.gaussian_filter1d(prof.astype(float), 2)
    q = np.where((np.arange(len(p)) >= lo) & (np.arange(len(p)) <= hi), p, 0)
    c = int(q.argmax())
    a = c
    while a > 0 and p[a] >= 0.3 * p[c]:
        a -= 1
    b = c
    while b < len(p) - 1 and p[b] >= 0.3 * p[c]:
        b += 1
    return a / len(p), b / len(p)


mw = mask & win
ys, xs = np.where(mw)
x0, x1 = np.percentile(xs, 2) / S, np.percentile(xs, 98) / S
y0, y1 = np.percentile(ys, 2) / S, np.percentile(ys, 98) / S
target = (x0, y0, x1, y1)
P = ndi.gaussian_filter(mask.astype(np.float32), 3)
P = P / P.max()
BW = BH = 0.5

boxes, trails = {}, {}
cx, cy = centroid(mask)
boxes["threshold + centroid"] = clamp_box(cx, cy, BW, BH)
boxes["projection (bone extent)"] = target
ucx, ucy = centroid(P > 0.5)
boxes["U-Net"] = clamp_box(ucx, ucy, BW, BH)


def gbox(v):
    return clamp_box(0.2 + 0.6 * v[0], 0.2 + 0.6 * v[1], 0.15 + 0.5 * v[2], 0.15 + 0.5 * v[3])


def hbox(v):
    c = centroid(P > 0.3 + 0.5 * v[0])
    return clamp_box(c[0] - 0.15 + 0.3 * v[3], c[1] - 0.15 + 0.3 * v[4], 0.15 + 0.5 * v[1], 0.15 + 0.5 * v[2])


def run(name, opt, d, mk):
    trails[name] = []

    def f(v):
        b = mk(v)
        trails[name].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        return 1.0 - iou(b, target)
    v, _ = opt(f, d, 8, 5, np.random.default_rng(11))
    boxes[name] = mk(v)


run("PSO", rc.pso, 4, gbox)
run("WOA", rc.woa, 4, gbox)
run("MPA", rc.mpa, 4, gbox)
run("U-Net + PSO (hybrid)", rc.pso, 5, hbox)
order = ["threshold + centroid", "projection (bone extent)", "U-Net", "PSO", "WOA", "MPA", "U-Net + PSO (hybrid)"]
colors = {"threshold + centroid": "#1f77b4", "projection (bone extent)": "#17becf", "U-Net": "#ff7f0e", "PSO": "#2ca02c",
          "WOA": "#d62728", "MPA": "#9467bd", "U-Net + PSO (hybrid)": "#e6007e"}
group = {"threshold + centroid": "traditional", "projection (bone extent)": "traditional", "U-Net": "segmentation",
         "PSO": "optimizer, no mask", "WOA": "optimizer, no mask", "MPA": "optimizer, no mask", "U-Net + PSO (hybrid)": "proposed hybrid"}

fig = plt.figure(figsize=(24, 12.5))
gs = GridSpec(2, 4, figure=fig, hspace=0.12, wspace=0.04, left=0.02, right=0.98, top=0.9, bottom=0.05)


def base(ax, title, color="black"):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=[0, 1, 1, 0])
    ax.set_title(title, fontsize=15, color=color, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


base(fig.add_subplot(gs[0, 0]), "original axial CT slice\n(LUMOS patient 1)")
for slot, nm in zip([(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], order):
    ax = fig.add_subplot(gs[slot])
    base(ax, f"{nm}\n({group[nm]})", colors[nm])
    if nm in trails:
        t = np.array(trails[nm])
        ax.scatter(t[:, 0], t[:, 1], c=np.arange(len(t)), cmap="viridis", s=22, alpha=0.8, zorder=3)
    b = boxes[nm]
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=colors[nm], lw=3.5, zorder=4))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
fig.suptitle("Same axial CT slice, every ROI strategy", fontsize=22, fontweight="bold", y=0.975)
fig.text(0.5, 0.015, "Demonstration only, no model training. U-Net panel is a stand-in (soft bone mask). Dots = candidate ROIs tried by each optimizer (early to late).",
         ha="center", fontsize=13, color="#444444")
fig.savefig(os.path.join(OUT, "roi_axial_all_methods_one_slide.png"), dpi=110, facecolor="white")
