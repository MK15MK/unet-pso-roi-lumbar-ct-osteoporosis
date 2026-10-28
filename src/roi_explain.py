import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
rc.require_gpu()
DEV = rc.DEV

df, labels = rc.read_labels(DATA_DIR)
ct = rc.find_ct_patients(DATA_DIR)
names = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}
picks = {}
for p in sorted(set(labels) & set(ct)):
    picks.setdefault(labels[p], p)
    if len(picks) == 3:
        break

slices = {}
for cls in (0, 1, 2):
    files = rc.pick_central(ct[picks[cls]])
    slices[cls] = rc.slice_to_u8(files[len(files) // 2])

S = rc.STORE
R = rc.DEFAULT_ROI


def stages(arr):
    xg = torch.from_numpy(arr)[None, None].to(DEV)
    mask = rc.bone_mask_u8(xg)[0, 0].cpu().numpy() > 0.5
    cx, cy, half = rc.geometry("rule", xg, None, R)
    crop = rc.make_crops("rule", xg, None, R)[0, 0].cpu().numpy()
    return mask, cx.item(), cy.item(), half.item(), crop


fig, axes = plt.subplots(3, 5, figsize=(15, 9.4))
titles = ["1  CT slice", "2  bone segmentation\n(HU > 150, cleaned)", "3  central window:\nbone inside it", "4  ROI box at the\nbone centre of mass", "5  crop fed to the classifier"]
for r, cls in enumerate((0, 1, 2)):
    arr = slices[cls]
    mask, cx, cy, half, crop = stages(arr)
    win = np.zeros((S, S), bool)
    win[int(0.2 * S):int(0.8 * S), int(0.25 * S):int(0.75 * S)] = True
    inside = mask & win
    axs = axes[r]
    axs[0].imshow(arr, cmap="gray", vmin=0, vmax=255)
    axs[1].imshow(arr, cmap="gray", vmin=0, vmax=255)
    ov = np.zeros((S, S, 4))
    ov[mask] = [1, 0.15, 0.15, 0.55]
    axs[1].imshow(ov)
    axs[2].imshow(arr, cmap="gray", vmin=0, vmax=255)
    ov2 = np.zeros((S, S, 4))
    ov2[inside] = [1, 0.15, 0.15, 0.6]
    axs[2].imshow(ov2)
    axs[2].add_patch(Rectangle((0.25 * S, 0.2 * S), 0.5 * S, 0.6 * S, fill=False, ec="#FFD400", lw=2, ls="--"))
    axs[2].plot(cx * S, cy * S, "o", color="#00E5FF", ms=9, mec="black")
    axs[3].imshow(arr, cmap="gray", vmin=0, vmax=255)
    axs[3].add_patch(Rectangle(((cx - half) * S, (cy - half) * S), 2 * half * S, 2 * half * S, fill=False, ec="#00E5FF", lw=2.6))
    axs[3].plot(cx * S, cy * S, "o", color="#00E5FF", ms=9, mec="black")
    axs[4].imshow(crop, cmap="gray", vmin=0, vmax=255)
    axs[0].set_ylabel(f"{names[cls]}\n(patient {picks[cls]})", fontsize=11)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    if r == 0:
        for a, t in zip(axs, titles):
            a.set_title(t, fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roi_segmentation_steps.png"), dpi=130, facecolor="white")
plt.close(fig)

arr = slices[1]
xg = torch.from_numpy(arr)[None, None].to(DEV)
mask = rc.bone_mask_u8(xg)[0, 0].cpu().numpy() > 0.5
cx0, cy0, _ = rc.geometry("rule", xg, None, R)
cx0, cy0 = cx0.item(), cy0.item()
sweeps = [("crop size (side)", "side", [0.35, 0.55, 0.85]), ("horizontal shift (dx)", "dx", [-0.15, 0.0, 0.15]),
          ("vertical shift (dy)", "dy", [-0.25, 0.0, 0.12])]
fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
for r, (label, key, vals) in enumerate(sweeps):
    for c, v in enumerate(vals):
        rr = dict(R)
        rr[key] = v
        cx, cy, half = rc.geometry("rule", xg, None, rr)
        cx, cy, half = cx.item(), cy.item(), half.item()
        ax = axes[r, c]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(Rectangle(((cx - half) * S, (cy - half) * S), 2 * half * S, 2 * half * S, fill=False, ec="#00E5FF", lw=2.6))
        ax.plot(cx * S, cy * S, "o", color="#00E5FF", ms=7, mec="black")
        ax.set_title(f"{key} = {v:+.2f}" if key != "side" else f"side = {v:.2f}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(label, fontsize=11)
fig.suptitle("What PSO, WOA and MPA tune: the crop size and shift around the bone centre (plus the mask threshold)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT_DIR, "roi_parameters_effect.png"), dpi=130, facecolor="white")
print("saved both figures; patients:", picks)
