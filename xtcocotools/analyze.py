"""Run keypoint error diagnostics: python -m xtcocotools.analyze --help."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coco import COCO
from .cocoanalyze import COCOanalyze


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--use-area",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GT area (default); --no-use-area uses bbox width * height * 0.53",
    )
    parser.add_argument("--iou-type", choices=["keypoints", "keypoints_crowd"], default="keypoints")
    parser.add_argument("--sigmas", type=float, nargs="+", help="One OKS sigma per joint")
    parser.add_argument(
        "--inverse-indices", type=int, nargs="+", help="Zero-based left/right permutation"
    )
    parser.add_argument("--image-ids", type=int, nargs="+", help="Analyze only these image IDs")
    parser.add_argument("--category-id", type=int)
    parser.add_argument("--max-dets", type=int, default=20)
    parser.add_argument("--oks-thresholds", type=float, nargs="+")
    parser.add_argument("--no-keypoint-errors", action="store_true")
    parser.add_argument("--no-score-errors", action="store_true")
    parser.add_argument("--no-background-errors", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Also save error PR plots as PDF")
    args = parser.parse_args(argv)

    gt = COCO(str(args.ground_truth))
    dt = gt.loadRes(str(args.predictions))
    a = COCOanalyze(
        gt,
        dt,
        iouType=args.iou_type,
        sigmas=args.sigmas,
        use_area=args.use_area,
        inverse_indices=args.inverse_indices,
    )
    if args.image_ids is not None:
        unknown = set(args.image_ids) - set(gt.getImgIds())
        if unknown:
            parser.error(f"Unknown image IDs: {sorted(unknown)}")
        a.params.imgIds = args.image_ids
    if args.category_id is not None:
        if args.category_id not in gt.getCatIds():
            parser.error(f"Unknown category ID: {args.category_id}")
        a.params.catIds = [args.category_id]
    a.params.maxDets = [args.max_dets]
    if args.oks_thresholds is not None:
        a.params.oksThrs = np.array(sorted(set(args.oks_thresholds)))
    a.analyze(
        check_kpts=not args.no_keypoint_errors,
        check_scores=not args.no_score_errors,
        check_bckgd=not args.no_background_errors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a.summarize(makeplots=args.plots, savedir=str(args.output_dir), team_name=args.predictions.stem)
    counts = {}
    for area, records in a.corrected_dts.items():
        counts[area] = {
            err: sum(sum(d.get(err, [])) for d in records)
            for err in ["good", "jitter", "inversion", "swap", "miss"]
        }
    report = {
        "protocol": {
            "evaluator": "xtcocotools",
            "iou_type": args.iou_type,
            "use_area": args.use_area,
            "gt_area_rule": "annotation area" if args.use_area else "bbox width * height * 0.53",
            "dt_area_rule": "COCO.loadRes result area (unscaled)",
            "sigmas": a.params.sigmas,
            "inverse_indices": a.params.inv_idx,
            "image_ids": a.params.imgIds,
            "category_ids": a.params.catIds,
            "max_dets": a.params.maxDets,
            "oks_thresholds": a.params.oksThrs,
            "localization_threshold": a.params.oksLocThrs,
            "jitter_thresholds": a.params.jitterKsThrs,
            "area_ranges": dict(zip(a.params.areaRngLbl, a.params.areaRng)),
            "correction_order": a.params.err_types,
            "stages": {
                "keypoints": a.params.check_kpts,
                "scores": a.params.check_scores,
                "background": a.params.check_bckgd,
            },
            "gt_annotations": len(a._gts),
            "eligible_predictions": len(a._dts),
        },
        "baseline_summary": a.baseline_summary,
        "stats": a.stats,
        "keypoint_counts": counts,
        "background_errors": [
            {
                "area": area,
                "oks": float(oks),
                "false_positive_ids": sorted(fp),
                "false_negative_ids": sorted(a.false_neg_gts[area, oks]),
            }
            for (area, oks), fp in a.false_pos_dts.items()
        ],
    }
    output = args.output_dir / "analysis.json"
    output.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    (args.output_dir / "corrected_detections.json").write_text(
        json.dumps(a.corrected_dts, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved diagnostics to {output}")


if __name__ == "__main__":
    main()
