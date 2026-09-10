# Analysis API and evaluation protocol

## Inputs and supported skeletons

`COCOanalyze(cocoGt, cocoDt, iouType="keypoints", sigmas=None, use_area=True, keypoint_names=None, inverse_indices=None)` accepts `xtcocotools.coco.COCO` objects. Load predictions through `COCO.loadRes` so their IDs, bbox, and area have the native result semantics. The analyzer works on private copies and does not modify the caller's annotations or predictions.

The default skeleton is COCO-17. A custom skeleton needs one positive sigma per keypoint and an explicit, zero-based left/right permutation. Applying that permutation twice must recover the original ordering; self-symmetric keypoints map to themselves. `keypoint_names` is optional. For CrowdPose-14, following the upstream demo's joint order:

```python
import numpy as np

analysis = COCOanalyze(
    coco_gt, coco_dt,
    iouType="keypoints_crowd",
    sigmas=np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07,
                     .87, .87, .89, .89, .79, .79]) / 10,
    inverse_indices=[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13],
    use_area=False,
)
```

Analyze one category and one `maxDets` limit per run. Select them through `analysis.params.catIds` and `analysis.params.maxDets` before calling `analyze()`. `analysis.params.imgIds` selects an image subset, for example a visibility condition. Multiple area ranges and OKS thresholds are supported. WholeBody and separate body-part analysis require additional geometry and score adapters and are rejected explicitly.

## Area rules

The port follows xtcocotools at commit `17252e742ed4306a6323d74055f4c15c75cba6ad`.

| Operation | `use_area=True` | `use_area=False` |
|---|---|---|
| Detection-to-GT OKS | GT `area` | GT bbox width × height × 0.53 |
| Per-keypoint similarity, including candidate people for swaps | Each candidate GT's `area` | Each candidate GT's bbox width × height × 0.53 |
| Corrected-keypoint distances | Matched GT's `area` | Matched GT's bbox width × height × 0.53 |
| GT area-range filtering | GT `area` | GT bbox width × height × 0.53 |
| Unmatched detection area filtering | Detection `area` | Detection `area` |

`use_area=False` ignores a supplied GT `area`; it is not merely a fallback for a missing field. With `use_area=True`, the analyzer reports a missing GT area as an error and tells the caller to choose the bbox policy explicitly. It does not invent segmentation areas.

For predictions, `COCO.loadRes` derives area from the provided bbox when one is present; for keypoint-only results it uses the rectangle spanning the predicted keypoints. These areas remain unscaled. The legacy score-correction Soft-NMS uses the average of the two predicted keypoint-envelope areas for its prediction-to-prediction similarity. This separate heuristic does not use GT areas or the 0.53 factor.

The default area ranges are `all=[0, 1e10]`, `medium=[32², 96²]`, and `large=[96², 1e10]`, with native inclusive boundaries. Original coco-analyze used `all=[32², 1e10]`; reproduce that exclusion only by setting the range explicitly.

## Evaluation and diagnostic outputs

- `evaluate(verbose=False, makeplots=False, savedir=None, team_name=None)` evaluates the original predictions with native xtcocotools matching and accumulation. For `keypoints`, `stats` contains ten values: AP, AP50, AP75, AP-medium, AP-large, AR, AR50, AR75, AR-medium, AR-large. For `keypoints_crowd`, it contains the native nine values: AP, AP50, AP75, AR, AR50, AR75, AP-easy, AP-medium, AP-hard. The last three refer to crowd groups, not object sizes. Summaries use the selected detection limit and thresholds. If there is no range named `all`, the first range is the overall range. Unsupported or absent slices are `-1`.
- `analyze(check_kpts=True, check_scores=True, check_bckgd=True)` computes diagnostics. `corrected_dts[area]` holds original predictions plus error masks, `opt_keypoints`, `max_oks`, and `opt_score` when the corresponding stages are enabled. Predictions filtered by the native all-zero visibility rule are omitted.
- `summarize(makeplots=False, savedir=None, team_name=None)` sets `stats` to records with `err`, `oks`, `areaRngLbl`, `maxDets`, `auc`, and `recall`. Here `auc` is mean interpolated precision over the evaluator's recall grid at that OKS threshold. Values remain on a 0–1 scale; undefined slices are `-1`.
- `false_pos_dts[(area, str(oks))]` and `false_neg_gts[(area, str(oks))]` contain unmatched, non-ignored IDs at the selected detection limit, after whichever correction stages were enabled. If keypoint and score corrections were enabled, these are residual errors after those corrections. They are not counts from the original predictions. Disable both stages to inspect original FP/FN matches.

Both `evaluate()` and `summarize()` populate `baseline_summary`, a mapping from metric names to values for the original predictions. The CLI includes this mapping in `analysis.json`, alongside the error-diagnostic `stats` records.

CrowdPose analysis uses native CrowdPose ignore, matching, and `crowdIndex` grouping rules:

| Group | Image `crowdIndex` |
|---|---|
| easy | `< 0.2` |
| medium | `>= 0.2` and `< 0.8` |
| hard | `>= 0.8` |

Each selected GT image needs a finite numeric `crowdIndex` for the nine-metric summary; missing metadata raises an error. The wrapper reads the metadata from the in-memory COCO object and evaluates each group with a private native evaluator. Groups include only `params.imgIds`; summary calculation preserves the caller's image selection and the full baseline evaluation records. Empty groups return `-1`.

The group AP reduction follows native `get_type_result()` exactly: mean precision over the first configured area range, including undefined (`-1`) cells, rounded to four decimal places. With the default parameters and full image set, all nine values match native `COCOeval.summarize()`. Custom detection limits and thresholds remain supported. This avoids the native report's file-path dependency and evaluator-state changes while retaining its metric definitions.

Change parameters before `analyze()`. Each run synchronizes `params.iouType` with the native evaluator before preparing annotations, so switching between `keypoints` and `keypoints_crowd` also switches the native ignore rules. Changing parameters afterward, including a subset, sigmas, area policy, or error order, makes `summarize()` reject the stale diagnostics until `analyze()` is run again. A failed analysis invalidates the previous result. Calling `evaluate()` with changed parameters also invalidates it, even if the parameters are later restored. Calling `evaluate()` between analysis and summarization is supported when parameters are unchanged.

## Interpretation and preserved correction behavior

Localization analysis first matches people at OKS 0.1, then classifies labeled keypoints using similarities to their matched person, its left/right counterpart, and other people. Default keypoint thresholds are 0.5 and 0.85. Ignored GT matches are excluded from localization diagnostics; unlabeled GT keypoints are excluded from error masks.

The original error order is `miss`, `swap`, `inversion`, `jitter`. Summary curves apply these corrections cumulatively. The score stage then uses the maximum OKS to an eligible GT followed by the original Soft-NMS heuristic. `opt_score` is a GT-assisted diagnostic score, not calibrated model confidence or a guaranteed AP-optimal ranking. Corrected curves are diagnostic counterfactuals and must be kept separate from baseline AP.

The port preserves the original coordinate-correction strength. Its displacement formula uses `sqrt(-log(t) * 2 * area * sigma²)`, whereas the OKS kernel uses `(2*sigma)²`. A correction targeted at threshold `t` therefore achieves keypoint similarity `t**0.25` under the matching kernel. Both formulas use the area policy selected above. Changing this correction rule would change the original analysis method and is outside this migration.

## Compatibility fixes and provenance

The source for the port is `matteorr/coco-analyze`, release commit `9eb8a0a9e57ad1e592661efc2b8964864c0e6f28`. Only the core `COCOanalyze` API and its PR plotting are migrated. The separate `analysisAPI` report suite, image-rendering utilities, and LaTeX report generator are not included.

The analysis adapter derives matched OKS and maximum-OKS diagnostics from native evaluation records. It preserves the native `-1` unmatched sentinel, so annotation ID zero remains valid. It also preserves prediction visibility filtering and CrowdPose's `num_keypoints` ignore convention.

The port fixes NumPy compatibility, floating-point coordinate correction for integer inputs, the exact 0.85 classification boundary, area-order contamination in score correction, false-negative threshold ordering, and threshold-specific GT ignore restoration. FP/FN sets respect evaluated IDs, ignored records, image subsets, and `maxDets`. Inputs are copied, and repeated calls restore original prediction geometry and scores.

`tests/fixtures/legacy_analysis.json` freezes original keypoint masks, corrected coordinates, score diagnostics, and summary values. Its test uses a single common area range to avoid the original area-order bug and excludes the original false-negative summary, whose threshold order was incorrect. That summary is tested separately against independently evaluated native slices.

To regenerate the legacy reference from the exact source checkout:

```bash
uv pip install --python .venv/bin/python scipy
.venv/bin/python scripts/generate_legacy_reference.py /path/to/coco-analyze
```

The generator records source hashes and patches only obsolete Python/NumPy usage and unused plotting imports in memory. Normal tests do not need the original checkout or SciPy.
