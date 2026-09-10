import copy

import numpy as np
import pytest

from xtcocotools.coco import COCO
from xtcocotools.cocoanalyze import COCOanalyze
from xtcocotools.cocoeval import COCOeval

SIGMAS = np.full(17, 0.1)
INVERSE = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def fixture(area=True, zero_id=False):
    # Joint spacing avoids accidental inversion/swap except where specified.
    kpts = np.array([[10 + (i % 5) * 35, 20 + (i // 5) * 45, 2] for i in range(17)], float)
    gt = COCO()
    anns = []
    for i, (im, shift) in enumerate([(1, 0), (1, 300), (2, 0)]):
        kp = kpts.copy()
        kp[:, 0] += shift
        a = {
            "id": i if zero_id else i + 1,
            "image_id": im,
            "category_id": 1,
            "keypoints": kp.ravel().tolist(),
            "num_keypoints": 17,
            "iscrowd": 0,
            "bbox": [shift, 0, 200, 200],
        }
        if area:
            a["area"] = 21200.0
        anns.append(a)
    gt.dataset = {
        "images": [
            {"id": 1, "crowdIndex": 0.1},
            {"id": 2, "crowdIndex": 0.2},
            {"id": 3, "crowdIndex": 0.8},
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": anns,
    }
    gt.createIndex()
    pred = copy.deepcopy(kpts)
    pred[0, 0] += 30  # jitter (KS ~0.588)
    pred[1, :2] = kpts[2, :2]  # inversion
    pred[5, :2] = kpts[5, :2] + [300, 0]  # swap
    pred[7, :2] += [1200, 1200]  # miss
    detections = [
        {"image_id": 1, "category_id": 1, "keypoints": pred.ravel().tolist(), "score": 0.6},
        {
            "image_id": 1,
            "category_id": 1,
            "keypoints": (kpts + [300, 0, 0]).ravel().tolist(),
            "score": 0.5,
        },
        {
            "image_id": 1,
            "category_id": 1,
            "keypoints": (kpts + [2500, 0, 0]).ravel().tolist(),
            "score": 0.9,
        },
        {"image_id": 3, "category_id": 1, "keypoints": kpts.ravel().tolist(), "score": 0.8},
    ]
    return gt, gt.loadRes(detections)


def analyzer(gt, dt, **kwargs):
    return COCOanalyze(gt, dt, sigmas=SIGMAS, **kwargs)


def summary(a):
    a.analyze()
    a.summarize()
    return {
        (s["err"], s["oks"], s["areaRngLbl"], s["maxDets"]): (s["auc"], s["recall"])
        for s in a.stats
    }


def native(gt, dt, use_area, **kwargs):
    ev = COCOeval(copy.deepcopy(gt), copy.deepcopy(dt), sigmas=SIGMAS, use_area=use_area)
    for key, val in kwargs.items():
        setattr(ev.params, key, val)
    ev.evaluate()
    ev.accumulate()
    return ev


@pytest.mark.parametrize("use_area", [True, False])
def test_baseline_and_matches_equal_native(use_area):
    gt, dt = fixture(area=use_area, zero_id=True)
    a = analyzer(gt, dt, use_area=use_area)
    ev = native(gt, dt, use_area)
    a.evaluate()
    np.testing.assert_allclose(a.cocoEval.eval["precision"], ev.eval["precision"])
    np.testing.assert_allclose(a.cocoEval.eval["recall"], ev.eval["recall"])
    for actual, expected in zip(a.cocoEval.evalImgs, ev.evalImgs):
        if expected is None:
            assert actual is None
            continue
        for field in ["dtMatches", "gtMatches", "dtIgnore", "gtIgnore", "dtScores"]:
            np.testing.assert_array_equal(actual[field], expected[field])
    stats = summary(a)
    for ti, oks in enumerate(a.params.oksThrs):
        for ai, label in enumerate(a.params.areaRngLbl):
            p = ev.eval["precision"][ti, :, :, ai, 0]
            r = ev.eval["recall"][ti, :, ai, 0]
            expected = (
                -1 if not (p >= 0).any() else p[p >= 0].mean(),
                -1 if not (r >= 0).any() else r[r >= 0].mean(),
            )
            np.testing.assert_allclose(stats["baseline", oks, label, 20], expected)


def test_no_area_equals_explicit_effective_area_through_full_pipeline():
    gt, dt = fixture(area=False)
    explicit = copy.deepcopy(gt)
    for g in explicit.anns.values():
        g["area"] = g["bbox"][2] * g["bbox"][3] * 0.53
    left = analyzer(gt, dt, use_area=False)
    right = analyzer(explicit, dt, use_area=True)
    sl, sr = summary(left), summary(right)
    assert sl.keys() == sr.keys()
    np.testing.assert_allclose(list(sl.values()), list(sr.values()))
    for label in left.params.areaRngLbl:
        for ld, rd in zip(left.corrected_dts[label], right.corrected_dts[label]):
            for key in [
                "good",
                "miss",
                "swap",
                "inversion",
                "jitter",
                "opt_keypoints",
                "opt_score",
                "max_oks",
            ]:
                if key in ld:
                    np.testing.assert_allclose(ld[key], rd[key])
    assert left.false_pos_dts == right.false_pos_dts
    assert left.false_neg_gts == right.false_neg_gts


def test_false_mode_ignores_existing_area():
    gt, dt = fixture(area=False)
    false_a = analyzer(gt, dt, use_area=False)
    for ann in gt.anns.values():
        ann["area"] = 1e8
    false_b = analyzer(gt, dt, use_area=False)
    np.testing.assert_allclose(list(summary(false_a).values()), list(summary(false_b).values()))
    true_a = analyzer(gt, dt, use_area=True)
    true_a.evaluate()
    false_a.evaluate()
    assert not np.allclose(true_a.cocoEval.eval["precision"], false_a.cocoEval.eval["precision"])


def test_area_missing_is_explicit_error():
    gt, dt = fixture(area=False)
    with pytest.raises(ValueError, match="use_area=False"):
        analyzer(gt, dt)


def test_error_labels_and_correction_strength():
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.analyze(check_scores=False, check_bckgd=False)
    d = a.corrected_dts["all"][0]
    assert np.flatnonzero(d["jitter"]).tolist() == [0]
    assert np.flatnonzero(d["inversion"]).tolist() == [1]
    assert np.flatnonzero(d["swap"]).tolist() == [5]
    assert np.flatnonzero(d["miss"]).tolist() == [7]
    assert sum(map(sum, [d[k] for k in ["good", "jitter", "inversion", "swap", "miss"]])) == 17
    gxy = np.array(gt.anns[1]["keypoints"]).reshape(-1, 3)[:, :2]
    oxy = np.array(d["opt_keypoints"]).reshape(-1, 3)[:, :2]
    ks = np.exp(-((gxy - oxy) ** 2).sum(1) / ((2 * SIGMAS) ** 2) / 21200 / 2)
    # Retain upstream's conservative correction placement (KS = threshold**.25).
    assert ks[0] == pytest.approx(0.85**0.25)
    assert ks[7] == pytest.approx(0.5**0.25)


def test_inputs_unchanged_and_repeatable():
    gt, dt = fixture(area=False)
    before_gt, before_dt = copy.deepcopy(gt.dataset), copy.deepcopy(dt.dataset)
    a = analyzer(gt, dt, use_area=False)
    first, second = summary(a), summary(a)
    assert first == second
    assert gt.dataset == before_gt
    assert dt.dataset == before_dt
    for d in a._dts:
        assert d["keypoints"] == dt.anns[d["id"]]["keypoints"]
        assert d["score"] == dt.anns[d["id"]]["score"]


@pytest.mark.parametrize("empty", ["detections", "ground_truth", "both", "zero_visibility"])
def test_empty_and_zero_predictions(empty):
    gt, dt = fixture(area=False)
    if empty in ("ground_truth", "both"):
        gt.dataset["annotations"] = []
        gt.createIndex()
    if empty in ("detections", "both"):
        dt = gt.loadRes([])
    if empty == "zero_visibility":
        for d in dt.anns.values():
            d["keypoints"][2::3] = [0] * 17
    a = analyzer(gt, dt, use_area=False)
    summary(a)
    if empty != "ground_truth":
        assert a.corrected_dts["all"] == []
    if empty != "ground_truth":
        assert a.false_neg_gts["all", "0.5"] == set(gt.anns)


def test_subset_and_maxdets_honored():
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.params.imgIds = [2]
    a.params.maxDets = [1]
    summary(a)
    assert not a._dts
    assert a.false_neg_gts["all", "0.5"] == {3}
    a.params.imgIds = [1]
    a.analyze(check_kpts=False, check_scores=False)
    assert a.false_pos_dts["all", "0.5"] == {3}
    assert a.false_neg_gts["all", "0.5"] == {1, 2}


def test_area_ranges_use_effective_gt_but_unscaled_dt_area():
    gt, dt = fixture(area=False)
    for g in gt.anns.values():
        g["bbox"] = [0, 0, 100, 100]  # effective GT area 5300 -> medium
    for d in dt.anns.values():
        d["bbox"] = [0, 0, 100, 100]
        d["area"] = 10000  # unmatched DT -> large
    a = analyzer(gt, dt, use_area=False)
    a.analyze(check_kpts=False, check_scores=False)
    assert a.false_pos_dts["medium", "0.5"] == set()
    assert 3 in a.false_pos_dts["large", "0.5"]
    assert a.false_neg_gts["large", "0.5"] == set()
    assert 3 in a.false_neg_gts["medium", "0.5"]


def test_crowd_and_zero_visible_gt_are_ignored():
    gt, dt = fixture(area=False)
    gt.anns[1]["iscrowd"] = 1
    gt.anns[2]["keypoints"][2::3] = [0] * 17
    a = analyzer(gt, dt, use_area=False)
    a.analyze(check_kpts=False, check_scores=False)
    assert a.false_neg_gts["all", "0.5"] == {3}


@pytest.mark.parametrize(
    "flags",
    [
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
    ],
)
def test_optional_stages(flags):
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.analyze(*flags)
    a.summarize()
    errors = {s["err"] for s in a.stats}
    assert ("score" in errors) == flags[1]
    assert ("false_neg" in errors) == flags[2]
    assert ("jitter" in errors) == flags[0]


def test_custom_skeleton_and_crowdpose():
    gt, dt = fixture(area=False)
    for a in list(gt.anns.values()) + list(dt.anns.values()):
        a["keypoints"] = a["keypoints"][:42]
        if "num_keypoints" in a:
            a["num_keypoints"] = 14
    kwargs = {"sigmas": np.full(14, 0.1), "inverse_indices": list(range(14)), "use_area": False}
    a = COCOanalyze(gt, dt, iouType="keypoints_crowd", **kwargs)
    summary(a)
    assert len(a.corrected_dts["all"][0]["good"]) == 14
    a._summarize_baseline()
    ev = COCOeval(
        copy.deepcopy(gt),
        copy.deepcopy(dt),
        "keypoints_crowd",
        sigmas=kwargs["sigmas"],
        use_area=False,
    )
    ev.evaluate()
    ev.accumulate()
    np.testing.assert_allclose(a.cocoEval.eval["precision"], ev.eval["precision"])
    with pytest.raises(ValueError, match="inverse_indices"):
        COCOanalyze(gt, dt, sigmas=kwargs["sigmas"], use_area=False)


def test_plot_smoke(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.params.oksThrs = [0.5]
    a.params.areaRng, a.params.areaRngLbl = [[0, 1e10]], ["all"]
    a.analyze()
    a.summarize(makeplots=True, savedir=str(tmp_path), team_name="test")
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_detection_id_zero_is_a_match():
    gt, dt = fixture(area=False, zero_id=True)
    dt.dataset["annotations"][0]["id"] = 0
    dt.createIndex()
    a = analyzer(gt, dt, use_area=False)
    a.analyze(check_scores=False, check_bckgd=False)
    assert a.localization_matches["all", "0.1", "dts"][0][0]["gtId"] == 0
    assert "jitter" in a.corrected_dts["all"][0]


def test_fn_summary_threshold_labels_match_independent_native_slices():
    gt, dt = fixture(area=False)
    # One prediction with OKS 0.7; unmatched at 0.95. Another GT has no prediction.
    gt.dataset["annotations"] = [gt.anns[1], gt.anns[3]]
    gt.createIndex()
    kp = np.array(gt.anns[1]["keypoints"]).reshape(-1, 3)
    kp[:, 0] += np.sqrt(-np.log(0.7) * 2 * 21200 * (0.2**2))
    dt = gt.loadRes(
        [{"image_id": 1, "category_id": 1, "keypoints": kp.ravel().tolist(), "score": 0.8}]
    )
    a = analyzer(gt, dt, use_area=False)
    a.params.oksThrs = [0.5, 0.95]
    a.params.areaRng, a.params.areaRngLbl = [[0, 1e10]], ["all"]
    a.analyze(check_kpts=False, check_scores=False)
    a.summarize()
    for s in a.stats:
        if s["err"] != "false_neg":
            continue
        ev = native(
            gt, dt, False, iouThrs=np.array([s["oks"]]), areaRng=[[0, 1e10]], areaRngLbl=["all"]
        )
        for e in ev.evalImgs:
            if e is None:
                continue
            e["dtIgnore"][0] |= e["dtMatches"][0] < 0
            e["gtIgnore"] = np.logical_or(e["gtIgnore"], e["gtMatches"][0] < 0)
        ev.accumulate()
        ps = ev.eval["precision"]
        expected = ps[ps >= 0].mean() if (ps >= 0).any() else -1
        assert s["auc"] == pytest.approx(expected)
    values = {s["oks"]: s["auc"] for s in a.stats if s["err"] == "false_neg"}
    assert values[0.5] == pytest.approx(1)
    assert values[0.95] == -1


def test_cli_writes_reproducible_protocol_and_diagnostics(tmp_path):
    import json

    from xtcocotools.analyze import main

    gt, dt = fixture(area=False)
    gt_path, dt_path = tmp_path / "gt.json", tmp_path / "dt.json"
    gt_path.write_text(json.dumps(gt.dataset))
    dt_path.write_text(json.dumps(dt.dataset["annotations"]))
    out = tmp_path / "output"
    main([str(gt_path), str(dt_path), str(out), "--no-use-area", "--no-score-errors"])
    report = json.loads((out / "analysis.json").read_text())
    assert report["protocol"]["use_area"] is False
    assert report["protocol"]["max_dets"] == [20]
    assert report["protocol"]["stages"]["scores"] is False
    assert report["stats"]
    assert (out / "corrected_detections.json").exists()


def test_score_diagnostics_do_not_depend_on_area_order():
    gt, dt = fixture(area=False)
    duplicate = copy.deepcopy(dt.dataset["annotations"][0])
    duplicate.update(id=5, score=0.55)
    dt.dataset["annotations"].append(duplicate)
    dt.createIndex()
    a = analyzer(gt, dt, use_area=False)
    b = analyzer(gt, dt, use_area=False)
    b.params.areaRng = [b.params.areaRng[i] for i in [0, 2, 1]]
    b.params.areaRngLbl = [b.params.areaRngLbl[i] for i in [0, 2, 1]]
    sa, sb = summary(a), summary(b)
    assert sa.keys() == sb.keys()
    for key in sa:
        np.testing.assert_allclose(sa[key], sb[key])
    for area in a.params.areaRngLbl:
        for ad, bd in zip(a.corrected_dts[area], b.corrected_dts[area]):
            assert ad["opt_score"] == pytest.approx(bd["opt_score"])


@pytest.mark.parametrize("iou_type", ["keypoints", "keypoints_crowd"])
def test_evaluate_summary_uses_actual_parameters_without_files(iou_type):
    gt, dt = fixture(area=False)
    a = COCOanalyze(gt, dt, iouType=iou_type, sigmas=SIGMAS, use_area=False)
    a.params.maxDets = [1]
    a.evaluate(verbose=True)
    assert a.params.imgIds == [1, 2, 3]
    assert a.cocoEval.params.imgIds == [1, 2, 3]
    p = a.cocoEval.eval["precision"][:, :, :, 0, 0]
    assert a.stats[0] == pytest.approx(p[p >= 0].mean())
    assert a.stats[1] == 0.0


def test_multicategory_input_can_select_person_before_running():
    gt, dt = fixture(area=False)
    gt.dataset["categories"].append({"id": 2, "name": "other"})
    gt.createIndex()
    a = analyzer(gt, dt, use_area=False)
    with pytest.raises(ValueError, match="one category"):
        a.analyze()
    a.params.catIds = [1]
    summary(a)


@pytest.mark.parametrize("change", ["images", "sigmas", "area", "max_dets", "errors"])
def test_summary_rejects_stale_analysis_parameters(change):
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.analyze()
    if change == "images":
        a.params.imgIds = [2]
        a.evaluate()
    elif change == "sigmas":
        a.params.sigmas *= 2
    elif change == "area":
        a.cocoEval.use_area = True
    elif change == "max_dets":
        a.params.maxDets = [1]
    else:
        a.params.err_types = ["miss"]
    with pytest.raises(RuntimeError, match="analyze"):
        a.summarize()


def test_failed_rerun_cannot_summarize_previous_results():
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.analyze()
    a.params.maxDets = [0]
    with pytest.raises(ValueError):
        a.analyze()
    a.params.maxDets = [20]
    with pytest.raises(RuntimeError, match="successfully"):
        a.summarize()


def test_evaluate_between_analyze_and_summary_is_safe():
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    expected = summary(a)
    a.evaluate()
    a.summarize()
    assert {
        (s["err"], s["oks"], s["areaRngLbl"], s["maxDets"]): (s["auc"], s["recall"])
        for s in a.stats
    } == expected


def test_port_matches_frozen_upstream_keypoint_and_score_diagnostics():
    import json
    from pathlib import Path

    frozen = json.loads((Path(__file__).parent / "fixtures/legacy_analysis.json").read_text())
    gt, dt = fixture(area=True)
    a = COCOanalyze(gt, dt)  # original COCO sigmas, not the synthetic test sigmas
    a.params.areaRng, a.params.areaRngLbl = [[0, 1e10]], ["all"]
    a.analyze()
    a.summarize()
    for actual, expected in zip(a.corrected_dts["all"], frozen["corrected_dts"]["all"]):
        for field in [
            "good",
            "jitter",
            "inversion",
            "swap",
            "miss",
            "opt_keypoints",
            "max_oks",
            "opt_score",
        ]:
            if field in expected:
                np.testing.assert_allclose(actual[field], expected[field], atol=1e-12)
    actual_stats = [s for s in a.stats if s["err"] != "false_neg"]
    for actual, expected in zip(actual_stats, frozen["stats"]):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], str):
                assert actual[key] == expected[key]
            else:
                assert actual[key] == pytest.approx(expected[key], abs=1e-12)


def test_temporary_parameter_change_via_evaluate_invalidates_analysis():
    gt, dt = fixture(area=False)
    a = analyzer(gt, dt, use_area=False)
    a.analyze()
    original = a.params.imgIds
    a.params.imgIds = [1]
    a.evaluate()
    a.params.imgIds = original
    with pytest.raises(RuntimeError, match="analyze"):
        a.summarize()


@pytest.mark.parametrize("iou_type", ["keypoints", "keypoints_crowd"])
@pytest.mark.parametrize("entrypoint", ["evaluate", "analyze"])
@pytest.mark.parametrize("warmup", [False, True])
def test_changed_iou_type_uses_native_ignore_rules(iou_type, entrypoint, warmup):
    gt, dt = fixture(area=False)
    # CrowdPose ignores a person with no visible joints even when v=1 joints
    # remain labeled; ordinary keypoints evaluation counts these labels.
    gt.anns[3]["keypoints"][2::3] = [1] * 17
    gt.anns[3]["num_keypoints"] = 0
    initial = "keypoints" if iou_type == "keypoints_crowd" else "keypoints_crowd"
    a = analyzer(gt, dt, iouType=initial, use_area=False)
    if warmup:
        a.analyze(check_kpts=False, check_scores=False)
    a.params.iouType = iou_type
    expected = analyzer(gt, dt, iouType=iou_type, use_area=False)
    if entrypoint == "evaluate":
        a.evaluate()
        ev = native(gt, dt, False, iouType=iou_type)
        np.testing.assert_array_equal(a.cocoEval.eval["precision"], ev.eval["precision"])
        np.testing.assert_array_equal(a.cocoEval.eval["recall"], ev.eval["recall"])
        expected.evaluate()
        np.testing.assert_array_equal(a.stats, expected.stats)
    else:
        for instance in (a, expected):
            instance.analyze(check_kpts=False, check_scores=False)
            instance.summarize()
        assert a.stats == expected.stats
        assert a.false_neg_gts == expected.false_neg_gts
        assert (3 in a.false_neg_gts["all", "0.5"]) == (iou_type == "keypoints")
    assert a.cocoEval.params.iouType == iou_type


def crowd_fixture(area):
    gt, dt = fixture(area=area)
    for im, index in zip(gt.dataset["images"], [0.1999, 0.2, 0.7999]):
        im["crowdIndex"] = index
    gt.dataset["images"].append({"id": 4, "crowdIndex": 0.8})
    person = copy.deepcopy(gt.anns[3])
    person.update(id=4, image_id=4)
    gt.dataset["annotations"].append(person)
    detection = copy.deepcopy(dt.anns[1])
    detection.update(id=5, image_id=4, keypoints=person["keypoints"].copy(), score=0.99)
    dt.dataset["annotations"].append(detection)
    gt.createIndex()
    dt.createIndex()
    return gt, dt


def native_crowd_summary(gt, dt, tmp_path, use_area, **params):
    import json

    gt_path, dt_path = tmp_path / "crowd_gt.json", tmp_path / "crowd_dt.json"
    gt_path.write_text(json.dumps(gt.dataset))
    dt_path.write_text(json.dumps(dt.dataset["annotations"]))
    reference_gt = COCO(str(gt_path))
    reference_dt = reference_gt.loadRes(str(dt_path))
    ev = COCOeval(reference_gt, reference_dt, "keypoints_crowd", SIGMAS, use_area)
    for key, val in params.items():
        setattr(ev.params, key, val)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats


@pytest.mark.parametrize("area_policy", ["provided", "missing", "ignored"])
@pytest.mark.parametrize("medium_first", [False, True])
def test_crowd_summary_matches_all_nine_native_metrics(tmp_path, area_policy, medium_first):
    gt, dt = crowd_fixture(area=area_policy != "missing")
    use_area = area_policy == "provided"
    if area_policy == "ignored":
        for ann in gt.anns.values():
            ann["area"] = 1.0  # Must not replace the bbox * 0.53 policy.
    before = copy.deepcopy((gt.dataset, dt.dataset))
    a = analyzer(gt, dt, iouType="keypoints_crowd", use_area=use_area)
    if medium_first:
        a.params.areaRng = [a.params.areaRng[i] for i in [1, 2, 0]]
        a.params.areaRngLbl = [a.params.areaRngLbl[i] for i in [1, 2, 0]]
    expected = native_crowd_summary(
        gt,
        dt,
        tmp_path,
        use_area,
        areaRng=a.params.areaRng,
        areaRngLbl=a.params.areaRngLbl,
    )
    a.evaluate()
    assert list(a.baseline_summary) == [
        "AP",
        "AP50",
        "AP75",
        "AR",
        "AR50",
        "AR75",
        "AP_easy",
        "AP_medium",
        "AP_hard",
    ]
    np.testing.assert_array_equal(a.stats, expected)
    np.testing.assert_array_equal(a.cocoEval.stats, expected)
    ev = native(
        gt,
        dt,
        use_area,
        iouType="keypoints_crowd",
        areaRng=a.params.areaRng,
        areaRngLbl=a.params.areaRngLbl,
    )
    np.testing.assert_array_equal(a.cocoEval.eval["precision"], ev.eval["precision"])
    assert list(a.cocoEval.params.imgIds) == [1, 2, 3, 4]
    assert list(a.cocoEval._paramsEval.imgIds) == [1, 2, 3, 4]
    # The ordinary diagnostic pipeline must retain the same original baseline.
    summary(a)
    np.testing.assert_array_equal(list(a.baseline_summary.values()), expected)
    a.evaluate()
    np.testing.assert_array_equal(a.stats, expected)
    assert (gt.dataset, dt.dataset) == before
    if medium_first:
        np.testing.assert_array_equal(a.stats[6:], [-1.0, -1.0, -1.0])
    else:
        assert a.baseline_summary["AP_medium"] == 0.0
        assert a.baseline_summary["AP_hard"] == 1.0


def test_crowd_summary_uses_only_selected_images_and_handles_empty_groups():
    gt, dt = crowd_fixture(area=False)
    del gt.imgs[1]["crowdIndex"]  # Unselected metadata is irrelevant.
    a = analyzer(gt, dt, iouType="keypoints_crowd", use_area=False)
    a.params.imgIds = [4, 3]
    a.evaluate()
    assert a.baseline_summary["AP_easy"] == -1.0  # No selected easy images.
    assert a.baseline_summary["AP_medium"] == -1.0  # Selected image has no GT.
    assert a.baseline_summary["AP_hard"] == 1.0
    ev = native(gt, dt, False, iouType="keypoints_crowd", imgIds=[3, 4])
    np.testing.assert_array_equal(a.cocoEval.eval["precision"], ev.eval["precision"])
    assert a.params.imgIds == [4, 3]
    assert list(a.cocoEval.params.imgIds) == [3, 4]


@pytest.mark.parametrize("index", [None, "0.2", float("nan"), float("inf")])
def test_crowd_summary_requires_real_crowd_index(index):
    gt, dt = fixture(area=False)
    gt.imgs[1].pop("crowdIndex")
    if index is not None:
        gt.imgs[1]["crowdIndex"] = index
    a = analyzer(gt, dt, iouType="keypoints_crowd", use_area=False)
    with pytest.raises(ValueError, match="crowdIndex for image 1"):
        a.evaluate()
    # The metadata requirement belongs to the grouped summary, not matching.
    a.analyze(check_kpts=False, check_scores=False)
    with pytest.raises(ValueError, match="crowdIndex for image 1"):
        a.summarize()


def test_crowd_cli_includes_native_baseline_groups(tmp_path):
    import json

    from xtcocotools.analyze import main

    gt, dt = crowd_fixture(area=False)
    expected = native_crowd_summary(gt, dt, tmp_path, use_area=False)
    out = tmp_path / "output"
    main(
        [
            str(tmp_path / "crowd_gt.json"),
            str(tmp_path / "crowd_dt.json"),
            str(out),
            "--no-use-area",
            "--iou-type",
            "keypoints_crowd",
            "--sigmas",
            *map(str, SIGMAS),
        ]
    )
    report = json.loads((out / "analysis.json").read_text())
    assert report["protocol"]["iou_type"] == "keypoints_crowd"
    np.testing.assert_array_equal(list(report["baseline_summary"].values()), expected)
    assert report["stats"]
