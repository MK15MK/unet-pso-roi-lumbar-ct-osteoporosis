# U-Net and PSO-Guided ROI Selection for Lumbar CT Osteoporosis Classification

**Abstract.** This repository contains research code for three-class osteoporosis classification
(Normal, Osteopenia, Osteoporosis) from lumbar-spine CT and structured clinical data on the public
LUMOS dataset. A U-Net trained on automatically generated bone masks supplies a bone-probability
prior; a metaheuristic (PSO, WOA or MPA) then searches a four-parameter crop policy scored by the
validation macro-F1 of a hybrid ResNet-18 + transformer classifier. Evaluation is patient-level and
label-stratified. The repository also holds scripts that visualise and compare ROI strategies and
segmentation-mask generation methods on a single slice, without training.

**Index Terms**: osteoporosis, lumbar CT, region of interest, U-Net, particle swarm optimization,
whale optimization, marine predators, vision transformer, opportunistic screening.

## I. Scope and status

- This is research code, not a medical device, and it is not validated for clinical use.
- No trained weights, patient data or dataset images are included.
- Results are not reported here. Any numbers must be regenerated from your own runs of the scripts.
- The segmentation targets are rule-based pseudo-labels (HU threshold plus morphological opening),
  not expert annotations. See Section VII.

## II. Method

1. **Pseudo-mask.** Bone mask = HU > 150 followed by a 3x3 morphological opening, computed on the
   DICOM-rescaled HU values before intensity scaling.
2. **Bone prior.** A four-level U-Net is trained with binary cross-entropy plus soft Dice loss
   (equal weights) and outputs a per-slice bone-probability map.
3. **ROI policy.** The crop centre is the centre of mass of pixels above a threshold inside a fixed
   central window; the policy vector (threshold, crop size, two shifts) is decoded to bounded ranges.
4. **Search.** PSO, WOA and MPA tune the policy with an equal evaluation budget (default 10 particles
   x 5 iterations, i.e. n(T+1) proxy trainings per fold). Fitness is 1 - patient-level validation
   macro-F1 of a quick proxy classifier trained on a slice subsample.
5. **Classifier.** ResNet-18 features form a token grid read by a two-layer transformer; clinical
   variables (age, height, weight, BMI, gender, ethnicity) are fused through a small MLP. Numeric
   variables are imputed and standardised, and categories are one-hot encoded using training
   patients only; unseen or rare categories encode as all zeros. Slice probabilities are averaged per patient.
6. **Protocol.** Patient-level, label-stratified five-fold cross-validation; every patient is in the
   test fold exactly once and the per-patient test predictions are pooled over the five folds; BMD, T-score
   and Z-score columns are refused as input features to prevent label leakage.

## III. Repository layout

| Path | Purpose |
|---|---|
| `src/roi_compare.py` | Main pipeline: data loading, U-Net, ROI optimizers, ViT-CNN, cross-validation |
| `src/pool_results.py` | Pools per-fold predictions: pooled metrics with patient-level bootstrap CIs, per-fold metrics, paired differences, permutation p-values (Holm-adjusted), confusion matrices |
| `src/roi_explain.py` | Figures explaining the masking and ROI parameters on axial CT slices |
| `src/roi_single.py` | One axial slice, all ROI strategies (no training) |
| `src/roi_axial.py`, `src/roi_ap.py` | Same-slice ROI comparison panels for axial CT and AP radiographs |
| `src/mask_compare.py` | Compare mask-generation methods on one slice |
| `src/mask_validation.py` | Agreement of mask methods with a TotalSegmentator reference over a patient sample |

## IV. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is required for the training pipeline.

## V. Usage

Obtain LUMOS from its authors' release [1] and point the scripts at it:

```bash
export LUMOS_DIR=/path/to/LUMOS     # folder with lumos_ct_*_dcm, lumos_x_*_dcm and the clinical xlsx
export OUT_DIR=outputs
python src/roi_compare.py           # all five folds, then pooled metrics (tag p10i5_clin)
python src/roi_compare.py --fold 0  # a single fold
python src/pool_results.py --tag p10i5_clin   # re-pool saved predictions
python src/roi_single.py            # one-slice ROI comparison, no training
python src/roi_axial.py             # axial CT panel figure
```

`mask_compare.py` and `mask_validation.py` read TotalSegmentator [8] output under
`$OUT_DIR`; run the TotalSegmentator CLI on a patient's DICOM folder first
(for example `TotalSegmentator -i <dicom_dir> -o $OUT_DIR/ts_out/p1 --task total --fast`).

## Outputs

For each method and fold, `pred_<tag>_<method>_fold<k>.csv` holds one row per test patient
(`patient_id, fold, true_label, p_normal, p_osteopenia, p_osteoporosis`). `pool_results.py` writes
`pooled_metrics_<tag>.csv`, `per_fold_metrics_<tag>.csv`, `class_counts_per_fold_<tag>.csv`,
`confusion_matrices_<tag>.json` and `paired_vs_unsegmented_<tag>.csv`. The tag encodes the swarm
size, iterations and whether clinical inputs were used, so runs with different settings never reuse
each other's cached searches.

## VI. Data availability and ethics

LUMOS is a third-party dataset [1]. This repository does not redistribute it or any image derived
from it. Follow the dataset's own data-access and ethics terms when using it.

## VII. Known limitations

- The HU threshold excludes low-density trabecular bone, so the pseudo-mask is biased toward dense
  bone and misses much of the vertebral body.
- Pseudo-mask quality has been checked only against a model-predicted reference (TotalSegmentator),
  not against expert annotation, and only on a small pilot sample.
- The ROI search is noisy and its selected policy can differ between folds.
- The scripts were developed on a single dataset; no external validation has been performed.

## VIII. Reproducibility

Seeds, fold assignments and selected ROI policies are controlled by command-line arguments
(`--seed`, `--folds`, `--fold`). Fix and record them for every run. Software versions are listed in
`requirements.txt` (unpinned); record your environment with `pip freeze`.

## IX. Citation

If you use this code, cite the repository (see `CITATION.cff`) and the works below.

## References

[1] K. Shi *et al.*, "LUMOS: A lumbar multimodal osteoporosis screening dataset with X-ray and CT images," in *Proc. ACM Int. Conf. Multimedia (MM)*, 2025, doi: 10.1145/3746027.3758282.

[2] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015, pp. 234-241.

[3] J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proc. IEEE Int. Conf. Neural Netw.*, 1995, pp. 1942-1948.

[4] S. Mirjalili and A. Lewis, "The whale optimization algorithm," *Adv. Eng. Softw.*, vol. 95, pp. 51-67, 2016.

[5] A. Faramarzi, M. Heidarinejad, S. Mirjalili, and A. H. Gandomi, "Marine predators algorithm: A nature-inspired metaheuristic," *Expert Syst. Appl.*, vol. 152, Art. no. 113377, 2020.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770-778.

[7] A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[8] J. Wasserthal *et al.*, "TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images," *Radiol. Artif. Intell.*, vol. 5, no. 5, 2023, doi: 10.1148/ryai.230024.

## License

MIT, see `LICENSE`.
