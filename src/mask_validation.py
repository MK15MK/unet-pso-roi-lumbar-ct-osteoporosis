import os
DATA_DIR = os.environ.get("LUMOS_DIR", "data/LUMOS")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
import importlib.util, os, subprocess, sys, csv, time
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from sklearn.cluster import KMeans

spec = importlib.util.spec_from_file_location("rc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_compare.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
DATA = DATA_DIR
OUT = os.path.join(OUT_DIR, "mask_validation")
os.makedirs(OUT, exist_ok=True)
df, labels = rc.read_labels(DATA)
ct = rc.find_ct_patients(DATA)
per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
picked = {}
for p in sorted(set(labels) & set(ct)):
    picked.setdefault(labels[p], [])
    if len(picked[labels[p]]) < per:
        picked[labels[p]].append(p)
pids = [p for c in (0, 1, 2) for p in picked.get(c, [])]
if len(sys.argv) > 2:
    pids = [int(x) for x in sys.argv[2].split(",")]
print("patients", pids, flush=True)
cname = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}


def hd95(a, b, sp):
    if not a.any() or not b.any():
        return np.nan
    da = ndi.distance_transform_edt(~a, sampling=sp)
    db = ndi.distance_transform_edt(~b, sampling=sp)
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    d = np.concatenate([db[sa], da[sb]])
    return float(np.percentile(d, 95))


def metrics(m, ref, core, sp):
    i = (m & ref).sum()
    return dict(dice=2 * i / max(m.sum() + ref.sum(), 1), iou=i / max((m | ref).sum(), 1),
                precision=i / max(m.sum(), 1), recall=i / max(ref.sum(), 1),
                core_recall=(m & core).sum() / max(core.sum(), 1), hd95_mm=hd95(m, ref, sp))


csvp = os.path.join(OUT, os.environ.get("MV_CSV", "slice_metrics.csv"))
cols = ["pid", "cls", "slice", "method", "dice", "iou", "precision", "recall", "core_recall", "hd95_mm", "centroid_shift_mm", "align_meanHU"]
done = set()
if os.path.exists(csvp):
    for r in csv.DictReader(open(csvp)):
        done.add(r["pid"])
else:
    csv.writer(open(csvp, "w", newline="")).writerow(cols)
for pid in pids:
    if str(pid) in done:
        continue
    t0 = time.time()
    allf = sorted(ct[pid])
    d = os.path.dirname(allf[0])
    segdir = os.path.join(OUT, f"ts_{pid}")
    if not os.path.exists(os.path.join(segdir, "vertebrae_L3.nii.gz")):
        subprocess.run(["TotalSegmentator", "-i", d, "-o", segdir, "--task", "total", "--fast", "--roi_subset",
                        "vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "--device", "gpu"],
                       capture_output=True)
    zs = [float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in allf]
    names = [f for _, f in sorted(zip(zs, allf))]
    rd = sitk.ImageSeriesReader()
    rd.SetFileNames(names)
    vol = rd.Execute()
    V = sitk.GetArrayFromImage(vol).astype(np.float32)
    sp = vol.GetSpacing()[:2]
    segv = None
    for f in sorted(os.listdir(segdir)):
        if f.startswith("vertebrae_"):
            a = np.asanyarray(nib.load(os.path.join(segdir, f)).dataobj) > 0
            segv = a if segv is None else (segv | a)
    chosen = rc.pick_central(ct[pid])
    kk = [names.index(f) for f in chosen if f in names]
    rows = []
    for k in kk[len(kk) // 2 - 1: len(kk) // 2 + 2]:
        hu = V[k]
        S0 = hu.shape[0]
        yy, xx = np.mgrid[0:S0, 0:S0]
        win = (yy >= 0.2 * S0) & (yy <= 0.8 * S0) & (xx >= 0.25 * S0) & (xx <= 0.75 * S0)
        nz = segv.shape[2]
        best = (-1e9, None)
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
                            best = (sc, m)
        if best[1] is None:
            continue
        ts = best[1]
        ref = ts & win
        if ref.sum() < 300 or best[0] < 60:
            continue
        core = ndi.binary_erosion(ts, iterations=5) & win
        body = hu > -300
        rule = ndi.binary_opening(hu > 150, np.ones((3, 3)))
        t = threshold_otsu(hu[body & (hu > 0)])
        otsu = ndi.binary_opening(hu > t, np.ones((3, 3)))
        km = KMeans(3, n_init=3, random_state=0).fit(hu[win].reshape(-1, 1))
        kmask = np.zeros_like(win)
        kmask[win] = km.labels_ == int(np.argmax(km.cluster_centers_.ravel()))
        kmask = ndi.binary_opening(kmask, np.ones((3, 3)))
        mk = np.zeros(hu.shape, int)
        mk[hu < 0] = 1
        mk[ndi.binary_erosion(hu > 250, np.ones((3, 3)))] = 2
        ws = watershed(sobel(ndi.gaussian_filter(hu, 1)), mk) == 2
        rc_ts = np.array(ndi.center_of_mass(ref))
        for name, m in (("HU>150", rule), ("Otsu", otsu), ("k-means", kmask), ("watershed", ws)):
            mw = m & win
            r = metrics(mw, ref, core, sp)
            cen = np.array(ndi.center_of_mass(mw)) if mw.any() else np.array([np.nan, np.nan])
            shift = float(np.linalg.norm((cen - rc_ts) * np.array([sp[1], sp[0]])))
            rows.append([pid, cname[labels[pid]], k, name, r["dice"], r["iou"], r["precision"], r["recall"], r["core_recall"], r["hd95_mm"], shift, best[0]])
    csv.writer(open(csvp, "a", newline="")).writerows(rows)
    print(f"patient {pid} ({cname[labels[pid]]}): {len(rows)//4} slices, {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
