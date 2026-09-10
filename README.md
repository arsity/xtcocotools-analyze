# xtcocotools-analyze

Keypoint error diagnostics from [matteorr/coco-analyze](https://github.com/matteorr/coco-analyze), running on the native [xtcocotools](https://github.com/jin-s13/xtcocoapi) evaluator.

The package adds `xtcocotools.cocoanalyze.COCOanalyze` and a command-line runner. It reports jitter, left/right inversions, swaps between people, missed keypoints, score diagnostics, and unmatched detections and ground truths. The original `COCO` and `COCOeval` implementations remain unchanged.

## Install this fork

Use Python 3.10 or newer and an isolated environment:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -e . pytest
```

This checkout supplies the `xtcocotools` package, version `1.14.3+analyze.2`. Installing the upstream PyPI release alone does not provide the analysis module.

## Python API

```python
from xtcocotools.coco import COCO
from xtcocotools.cocoanalyze import COCOanalyze

coco_gt = COCO("ground_truth.json")
coco_dt = coco_gt.loadRes("predictions.json")
analysis = COCOanalyze(coco_gt, coco_dt, use_area=False)
analysis.params.maxDets = [20]
analysis.analyze()
analysis.summarize()
print(analysis.baseline_summary)
print(analysis.stats)
```

**Set `use_area=False` when using bbox-based normalization.** Both the OKS calculations and GT size groups then use `bbox[2] * bbox[3] * 0.53`, even if an `area` field is present. `use_area=True`, the default, requires the annotated GT area. Prediction areas follow `COCO.loadRes` and are not multiplied by 0.53.

## Command line

```bash
.venv/bin/python -m xtcocotools.analyze \
  ground_truth.json predictions.json results/my_baseline \
  --no-use-area --plots
```

The runner writes `analysis.json` with parameters, a `baseline_summary` for original predictions, diagnostic statistics, error counts, and FP/FN IDs; `corrected_detections.json` contains per-detection diagnostics. `--plots` adds PR plots as PDFs. Use `--image-ids` to select a condition subset and `--no-score-errors` to omit score correction.

For `keypoints_crowd`, the baseline summary contains the native nine metrics, including AP for the easy, medium, and hard `crowdIndex` groups. Selected GT images must provide `crowdIndex`; grouping respects `--image-ids`.

A demo using the public COCO examples included in the repository:

```bash
.venv/bin/python -m xtcocotools.analyze \
  annotations/example_coco_val.json annotations/example_coco_preds.json \
  results/coco_demo --plots
```

See [the analysis guide](docs/analysis.md) for custom skeletons, area rules, correction semantics, and differences from the original implementation. The analysis API supports `keypoints` and `keypoints_crowd`; WholeBody and separate body-part analysis are not implemented. The native evaluator retains its existing support for those modes.

## Tests and provenance

```bash
.venv/bin/python -m pytest -q
```

Tests compare the analysis baseline with native xtcocotools, cover missing-area inputs, and check a frozen reference generated from the original coco-analyze implementation. The reference includes its source commit and file hashes.

The port retains the original MIT notice in [LICENSE.coco-analyze](LICENSE.coco-analyze). The xtcocotools license is in [LICENSE](LICENSE). [Upstream documentation](docs/upstream-readme.md) is preserved for reference.
