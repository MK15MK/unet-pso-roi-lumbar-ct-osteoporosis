import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util, os, glob
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from sklearn.cluster import KMeans

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
ct = rc.find_ct_patients(DATA_DIR)
files = rc.pick_central(ct[1])
chosen = files[len(files) // 2]
d = os.path.dirname(chosen)
rd = sitk.ImageSeriesReader()
import pydicom
allf = [f for f in files_all] if False else sorted(ct[1])
zs = [float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in allf]
names = [f for _, f in sorted(zip(zs, allf))]
rd.SetFileNames(names)
vol = rd.Execute()
k = [os.path.normcase(os.path.abspath(n)) for n in names].index(os.path.normcase(os.path.abspath(chosen)))
import nibabel as nib
hu = sitk.GetArrayFromImage(vol)[k].astype(np.float32)
segv = None
for f in sorted(glob.glob(os.path.join(OUT_DIR, "ts_out", "p1", "vertebrae_*.nii.gz"))):
    a = np.asanyarray(nib.load(f).dataobj) > 0
    segv = a if segv is None else (segv | a)
best = (-1e9, None)
nz = segv.shape[2]
for zi in (k, nz - 1 - k):
    sl = segv[:, :, zi]
    if not sl.any():
        continue
    for tr in (False, True):
        for fu in (False, True):
            for fl in (False, True):
                m = sl.T if tr else sl
                m = m[::-1] if fu else m
                m = m[:, ::-1] if fl else m
                sc = hu[m].mean() if m.any() else -1e9
                if sc > best[0]:
                    best = (sc, m, zi)
ts = best[1]
print("alignment: mean HU in TS mask", round(float(best[0]), 1), "nifti z", best[2])
S0 = hu.shape[0]
yy, xx = np.mgrid[0:S0, 0:S0]
win = (yy >= 0.2 * S0) & (yy <= 0.8 * S0) & (xx >= 0.25 * S0) & (xx <= 0.75 * S0)
body = hu > -300
rule = ndi.binary_opening(hu > 150, np.ones((3, 3)))
t = threshold_otsu(hu[body & (hu > 0)])
otsu = ndi.binary_opening(hu > t, np.ones((3, 3)))
vals = hu[win].reshape(-1, 1)
km = KMeans(3, n_init=5, random_state=0).fit(vals)
bone_c = int(np.argmax(km.cluster_centers_.ravel()))
kmeans = np.zeros_like(win)
kmeans[win] = km.labels_ == bone_c
kmeans = ndi.binary_opening(kmeans, np.ones((3, 3)))
grad = sobel(ndi.gaussian_filter(hu, 1))
mk = np.zeros(hu.shape, int)
mk[hu < 0] = 1
mk[ndi.binary_erosion(hu > 250, np.ones((3, 3)))] = 2
ws = watershed(grad, mk) == 2
core = ndi.binary_erosion(ts, iterations=5)
M = {"HU > 150 (current)": rule, f"Otsu (T={t:.0f} HU)": otsu, "k-means (3 classes)": kmeans, "watershed": ws,
     "TotalSegmentator vertebra": ts, "TotalSegmentator\ntrabecular core": core}
Mw = {n: (m & win) for n, m in M.items()}
names_ = list(M)
dice = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
D = np.array([[dice(Mw[a], Mw[b]) for b in names_] for a in names_])
lo, hi = -160, 240
img = np.clip((hu - lo) / (hi - lo), 0, 1)
ys, xs = np.where(win)
y0, y1, x0, x1 = ys.min() - 20, ys.max() + 20, xs.min() - 20, xs.max() + 20
fig = plt.figure(figsize=(24, 12.5))
axs = [fig.add_subplot(2, 4, i + 1) for i in range(8)]
axs[0].imshow(img[y0:y1, x0:x1], cmap="gray")
axs[0].set_title("axial CT slice\n(LUMOS patient 1)", fontsize=15, fontweight="bold")
for ax, n in zip(axs[1:7], names_):
    ax.imshow(img[y0:y1, x0:x1], cmap="gray")
    ov = np.zeros(hu.shape + (4,))
    ov[Mw[n]] = [1, 0.1, 0.1, 0.55]
    ax.imshow(ov[y0:y1, x0:x1])
    ax.set_title(n, fontsize=15, fontweight="bold")
im = axs[7].imshow(D, cmap="viridis", vmin=0, vmax=1)
axs[7].set_xticks(range(len(names_)))
axs[7].set_yticks(range(len(names_)))
lab = ["HU>150", "Otsu", "k-means", "watershed", "TS vert.", "TS core"]
axs[7].set_xticklabels(lab, rotation=45, ha="right", fontsize=11)
axs[7].set_yticklabels(lab, fontsize=11)
for i in range(len(names_)):
    for j in range(len(names_)):
        axs[7].text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", color="white" if D[i, j] < 0.6 else "black", fontsize=11)
axs[7].set_title("Dice between masks\n(inside central window)", fontsize=15, fontweight="bold")
for ax in axs[:7]:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Same slice, different segmentation-mask generation methods", fontsize=22, fontweight="bold", y=0.975)
fig.text(0.5, 0.015, "TotalSegmentator is a pretrained published model (no training here). No mask is expert ground truth; Dice shows agreement between methods only.", ha="center", fontsize=13, color="#444444")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "mask_methods_compare.png"), dpi=105, facecolor="white")
print(np.round(D, 2)); print("slice idx", k, "of", len(names), "TS px", int(ts.sum()))
