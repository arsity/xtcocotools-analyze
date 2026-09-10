"""Keypoint error analysis ported from matteorr/coco-analyze.

Copyright (c) 2017 matteorr. MIT license: see LICENSE.coco-analyze.
Uses the native xtcocotools evaluator; see docs/analysis.md for protocol details.
"""

__author__ = "mrr"
__version__ = "2.0"

import copy
import json
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from ._analyze_eval import AnalyzeEval
from .cocoeval import COCOeval


class COCOanalyze:
    # Interface for analyzing the keypoints detections on the Microsoft COCO dataset.
    def __init__(
        self,
        cocoGt,
        cocoDt,
        iouType="keypoints",
        sigmas=None,
        use_area=True,
        keypoint_names=None,
        inverse_indices=None,
    ):
        """
        Initialize COCOanalyze using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        # ground truth COCO API
        self.cocoGt = copy.deepcopy(cocoGt)
        # detections COCO API
        self.cocoDt = copy.deepcopy(cocoDt)
        # evaluation COCOeval API
        self.cocoEval = AnalyzeEval(self.cocoGt, self.cocoDt, iouType, sigmas, use_area)
        # gt for analysis
        self._gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds())
        # dt for analysis
        self._dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds())
        # store the original detections without any modification
        self._original_dts = {d["id"]: d for d in copy.deepcopy(self._dts)}
        # dt with corrections
        self.corrected_dts = {}
        # false positive dts
        self.false_pos_dts = {}
        # ground truths with info about false negatives
        self.false_neg_gts = {}
        # dt-gt matches
        self.localization_matches = {}
        self.bckgd_err_matches = {}
        # evaluation parameters
        self.params = {}
        self.params = Params(iouType=iouType)
        self.params.sigmas = np.array(self.cocoEval.sigmas, dtype=float, copy=True)
        self._configure_skeleton(keypoint_names, inverse_indices)
        self.params.imgIds = sorted(cocoGt.getImgIds())
        self.params.catIds = sorted(cocoGt.getCatIds())
        # get the max number of detections each team has per image
        if use_area and any("area" not in g for g in self._gts):
            raise ValueError("GT area is missing; pass use_area=False to use bbox area * 0.53")
        # result summarization
        self.stats = []
        self.baseline_summary = {}
        self._analysis_signature = None

    def _configure_skeleton(self, names, inverse_indices):
        """Require explicit metadata for non-COCO skeletons."""
        k = len(self.params.sigmas)
        if names is not None:
            if len(names) != k:
                raise ValueError("keypoint_names must match sigmas")
            self.params.kpts_name = list(names)
        elif k != 17:
            self.params.kpts_name = [str(i) for i in range(k)]
        if inverse_indices is None:
            if k != 17 or names is not None:
                raise ValueError("Provide inverse_indices for a custom skeleton")
        else:
            indices = np.asarray(inverse_indices)
            if (
                indices.shape != (k,)
                or not np.issubdtype(indices.dtype, np.integer)
                or sorted(indices.tolist()) != list(range(k))
                or not np.array_equal(indices[indices], np.arange(k))
            ):
                raise ValueError("inverse_indices must be an involutive permutation")
            self.params.inv_idx = indices.tolist()
        self.params.num_kpts = k
        self.params.inv_kpts_name = [self.params.kpts_name[i] for i in self.params.inv_idx]

    def _gt_area(self, gt):
        """Area for OKS and legacy correction distances, exactly as xtcocotools."""
        if self.cocoEval.use_area:
            return gt["area"]
        return gt["bbox"][2] * gt["bbox"][3] * 0.53

    def _prepare_analysis(self):
        self._cleanup()
        p = self.params
        if p.iouType not in ("keypoints", "keypoints_crowd"):
            raise ValueError("Analysis supports keypoints and keypoints_crowd only")
        if len(p.catIds) != 1:
            raise ValueError("Analyze one category at a time via params.catIds")
        if len(p.areaRng) != len(p.areaRngLbl) or len(set(p.areaRngLbl)) != len(p.areaRngLbl):
            raise ValueError("areaRng and unique areaRngLbl must have matching lengths")
        if not p.maxDets or any(int(m) != m or m <= 0 for m in p.maxDets):
            raise ValueError("maxDets must contain positive integers")
        if len(p.maxDets) != 1:
            raise ValueError("Analyze one maxDets limit at a time")
        if len(p.oksThrs) == 0 or not np.all(
            (np.asarray(p.oksThrs) > 0) & (np.asarray(p.oksThrs) <= 1)
        ):
            raise ValueError("oksThrs must be in (0, 1]")
        if not 0 < p.oksLocThrs <= 1 or not 0 < p.jitterKsThrs[0] < p.jitterKsThrs[1] <= 1:
            raise ValueError("Invalid localization or jitter thresholds")
        if not set(p.err_types) <= {"miss", "swap", "inversion", "jitter"}:
            raise ValueError("Unknown keypoint error type")
        sigmas = np.asarray(p.sigmas, dtype=float)
        if sigmas.shape != (p.num_kpts,) or not np.all(np.isfinite(sigmas) & (sigmas > 0)):
            raise ValueError("sigmas must contain one positive finite value per keypoint")
        p.sigmas = sigmas.copy()
        self.cocoEval.sigmas = sigmas.copy()
        self.cocoEval.params.iouType = p.iouType
        self.cocoEval.params.imgIds = sorted(set(p.imgIds))
        self.cocoEval.params.catIds = list(p.catIds)
        self.cocoEval._prepare()
        self._gts = [g for gs in self.cocoEval._gts.values() for g in gs]
        self._dts = [d for ds in self.cocoEval._dts.values() for d in ds]
        for g in self._gts:
            if self.cocoEval.use_area and "area" not in g:
                raise ValueError("GT area is missing; pass use_area=False to use bbox area * 0.53")
        for ann in self._gts + self._dts:
            if len(ann["keypoints"]) != 3 * p.num_kpts:
                raise ValueError("keypoint vector length does not match sigmas")
        p.teamMaxDets = [max([len(ds) for ds in self.cocoEval._dts.values()] + [1])]

    def evaluate(self, verbose=False, makeplots=False, savedir=None, team_name=None):
        if self._analysis_signature is not None and self._analysis_signature != self._signature():
            self._analysis_signature = None
        # at any point the evaluate function is called it will run the COCOeval
        # API on the current detections
        self._prepare_analysis()

        # set the cocoEval params based on the params from COCOanalyze
        self.cocoEval.params.areaRng = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = sorted(self.params.oksThrs)
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()

        # for all areaRngLbl and maxDets plot pr curves at all iouThrs values
        recalls = self.cocoEval.params.recThrs[:]
        # dimension of precision: [TxRxKxAxM]
        ps_mat = self.cocoEval.eval["precision"][::-1, :, :, :, :]

        # stats are the same returned from cocoEval.stats
        self.stats = self._evaluation_stats(verbose)
        self.cocoEval.stats = self.stats.copy()

        if makeplots:
            self._plot(
                recalls=recalls,
                ps_mat=ps_mat,
                params=self.params,
                savedir=savedir,
                team_name=team_name,
            )

    def _evaluation_stats(self, verbose):
        """Native metric ordering using the selected thresholds and detection limit."""
        self.baseline_summary = {}
        p = self.cocoEval.params
        overall = "all" if "all" in p.areaRngLbl else p.areaRngLbl[0]

        def mean(ap, threshold=None, area=None):
            area = overall if area is None else area
            if area not in p.areaRngLbl:
                return -1.0
            ai = p.areaRngLbl.index(area)
            values = self.cocoEval.eval["precision" if ap else "recall"]
            if threshold is not None:
                values = values[np.isclose(p.iouThrs, threshold)]
            values = values[..., ai, 0]
            valid = values[values >= 0]
            return float(valid.mean()) if valid.size else -1.0

        metrics = {"AP": mean(True), "AP50": mean(True, 0.5), "AP75": mean(True, 0.75)}
        if p.iouType == "keypoints":
            metrics.update(AP_medium=mean(True, area="medium"), AP_large=mean(True, area="large"))
        metrics.update(AR=mean(False), AR50=mean(False, 0.5), AR75=mean(False, 0.75))
        if p.iouType == "keypoints_crowd":
            metrics.update(self._crowd_stats())
        else:
            metrics.update(AR_medium=mean(False, area="medium"), AR_large=mean(False, area="large"))
        self.baseline_summary = metrics
        if verbose:
            for label, value in metrics.items():
                print(f"{label}: {value:.3f} (maxDets={p.maxDets[0]})")
        return np.array(list(metrics.values()))

    def _crowd_stats(self):
        """CrowdPose groups with native reduction, without file or evaluator side effects."""
        groups = {"AP_easy": [], "AP_medium": [], "AP_hard": []}
        for image_id in self.cocoEval.params.imgIds:
            index = self.cocoGt.imgs[image_id].get("crowdIndex")
            if not isinstance(index, (int, float, np.number)) or not np.isfinite(index):
                raise ValueError(
                    f"CrowdPose summary requires a finite numeric crowdIndex for image {image_id}"
                )
            label = "AP_easy" if index < 0.2 else "AP_medium" if index < 0.8 else "AP_hard"
            groups[label].append(image_id)

        # Native get_type_result() re-evaluates each group, reads a GT file, and
        # leaves its evaluator on the hard subset. Use private COCO objects here.
        evaluator = COCOeval(
            copy.deepcopy(self.cocoGt),
            copy.deepcopy(self.cocoDt),
            iouType="keypoints_crowd",
            sigmas=self.cocoEval.sigmas.copy(),
            use_area=self.cocoEval.use_area,
        )
        results = {}
        for label, image_ids in groups.items():
            if not image_ids:
                results[label] = -1.0
                continue
            evaluator.params = copy.deepcopy(self.cocoEval.params)
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            # Match native get_type_result(): first area, unfiltered precision,
            # four decimal places. In particular, do not drop undefined (-1) cells.
            score = evaluator.eval["precision"][:, :, :, 0, :]
            results[label] = float(round(np.mean(score), 4))
        return results

    def _signature(self):
        return json.dumps(
            {"params": self.params.__dict__, "use_area": self.cocoEval.use_area},
            sort_keys=True,
            default=lambda value: value.tolist(),
        )

    def analyze(self, check_kpts=True, check_scores=True, check_bckgd=True):
        self._analysis_signature = None
        self._prepare_analysis()
        self.corrected_dts = {}
        self.localization_matches = {}
        self.bckgd_err_matches = {}

        for areaRngLbl in self.params.areaRngLbl:
            self.corrected_dts[areaRngLbl] = copy.deepcopy(self._dts)

        self.false_neg_gts = {}
        self.false_pos_dts = {}

        # find keypoint errors in detections that are matched to ground truths
        self.params.check_kpts = check_kpts
        if check_kpts:
            self.find_keypoint_errors()

        # find scoring errors in all detections
        self.params.check_scores = check_scores
        if check_scores:
            self.find_score_errors()

        # find background false positive errors and false negatives
        self.params.check_bckgd = check_bckgd
        if check_bckgd:
            self.find_bckgd_errors()
        self._cleanup()
        self._analysis_signature = self._signature()

    def find_keypoint_errors(self):
        tic = time.time()
        print("Analyzing keypoint errors...")
        # find all matches between dts and gts at the lowest iou thresh
        # allowed for localization. Matches with lower oks are not valid
        oksLocThrs = [self.params.oksLocThrs]
        areaRng = self.params.areaRng
        areaRngLbl = self.params.areaRngLbl
        dtMatches, gtMatches = self._find_dt_gt_matches(oksLocThrs, areaRng, areaRngLbl)

        for aind, arearnglbl in enumerate(areaRngLbl):
            self.localization_matches[arearnglbl, str(self.params.oksLocThrs), "dts"] = dtMatches[
                arearnglbl, str(self.params.oksLocThrs)
            ]
            self.localization_matches[arearnglbl, str(self.params.oksLocThrs), "gts"] = gtMatches[
                arearnglbl, str(self.params.oksLocThrs)
            ]

        # find which errors affect the oks of detections that are matched
        corrected_dts = self._find_kpt_errors()

        for areaRngLbl in self.params.areaRngLbl:
            corrected_dts_dict = {}

            for cdt in corrected_dts[areaRngLbl]:
                corrected_dts_dict[cdt["id"]] = cdt
            assert len(corrected_dts[areaRngLbl]) == len(corrected_dts_dict)

            for cdt in self.corrected_dts[areaRngLbl]:
                if cdt["id"] in corrected_dts_dict:
                    cdt["opt_keypoints"] = corrected_dts_dict[cdt["id"]]["keypoints"]
                    cdt["inversion"] = corrected_dts_dict[cdt["id"]]["inversion"]
                    cdt["good"] = corrected_dts_dict[cdt["id"]]["good"]
                    cdt["jitter"] = corrected_dts_dict[cdt["id"]]["jitter"]
                    cdt["miss"] = corrected_dts_dict[cdt["id"]]["miss"]
                    cdt["swap"] = corrected_dts_dict[cdt["id"]]["swap"]

        toc = time.time()
        print(f"DONE (t={toc - tic:0.2f}s).")

    def _find_dt_gt_matches(self, oksThrs, areaRng, areaRngLbl):
        self.cocoEval.params.areaRng = areaRng
        self.cocoEval.params.areaRngLbl = areaRngLbl
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        self.cocoEval.evaluate()

        dtMatches = {}
        gtMatches = {}

        for aind, arearnglbl in enumerate(areaRngLbl):
            # evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]
            evalImgs = [
                e for e in filter(None, self.cocoEval.evalImgs) if e["aRng"] == areaRng[aind]
            ]

            for oind, oks in enumerate(oksThrs):
                dtMatchesAreaOks = {}
                gtMatchesAreaOks = {}

                for i, e in enumerate(evalImgs):
                    # add all matches to the dtMatches dictionary
                    for dind, did in enumerate(e["dtIds"]):
                        gtMatch = int(e["dtMatches"][oind][dind])

                        if gtMatch >= 0:
                            # check that a detection is not already matched
                            assert did not in dtMatchesAreaOks
                            dtMatchesAreaOks[did] = [
                                {
                                    "gtId": gtMatch,
                                    "dtId": did,
                                    "oks": e["dtIous"][oind][dind],
                                    "score": e["dtScores"][dind],
                                    "ignore": int(e["dtIgnore"][oind][dind]),
                                    "image_id": e["image_id"],
                                }
                            ]
                            # add the gt match as well since multiple dts can have same gt
                            entry = {
                                "dtId": did,
                                "gtId": gtMatch,
                                "oks": e["dtIous"][oind][dind],
                                "ignore": int(e["dtIgnore"][oind][dind]),
                                "image_id": e["image_id"],
                            }
                            gtMatchesAreaOks.setdefault(gtMatch, []).append(entry)

                    # add matches to the gtMatches dictionary
                    for gind, gid in enumerate(e["gtIds"]):
                        dtMatch = int(e["gtMatches"][oind][gind])

                        if dtMatch >= 0:
                            entry = {
                                "dtId": dtMatch,
                                "gtId": gid,
                                "oks": e["gtIous"][oind][gind],
                                "ignore": int(e["gtIgnore"][gind]),
                                "image_id": e["image_id"],
                            }
                            if gid in gtMatchesAreaOks:
                                if entry not in gtMatchesAreaOks[gid]:
                                    gtMatchesAreaOks[gid].append(entry)
                            else:
                                gtMatchesAreaOks[gid] = [entry]

                dtMatches[arearnglbl, str(oks)] = dtMatchesAreaOks
                gtMatches[arearnglbl, str(oks)] = gtMatchesAreaOks
        return dtMatches, gtMatches

    def _find_kpt_errors(self):
        zero_kpt_gts = 0
        corrected_dts = {}

        oksLocThrs = self.params.oksLocThrs
        areaRngLbls = self.params.areaRngLbl

        for aind, areaRngLbl in enumerate(areaRngLbls):
            localization_matches_dts = self.localization_matches[areaRngLbl, str(oksLocThrs), "dts"]
            corrected_dts[areaRngLbl] = []
            # this contains all the detections that have been matched with a gt
            for did in localization_matches_dts:
                # get the info on the [dt,gt] match
                # load the detection and ground truth annotations
                dtm = localization_matches_dts[did][0]
                if dtm["ignore"]:
                    continue
                image_id = dtm["image_id"]

                dt = self.cocoDt.loadAnns(did)[0]
                dt_kpt_x = np.array(dt["keypoints"][0::3])
                dt_kpt_y = np.array(dt["keypoints"][1::3])
                dt_kpt_v = np.array(dt["keypoints"][2::3])
                dt_kpt_arr = np.delete(np.array(dt["keypoints"]), slice(2, None, 3))

                gt = self.cocoGt.loadAnns(dtm["gtId"])[0]
                gt_kpt_x = np.array(gt["keypoints"][0::3])
                gt_kpt_y = np.array(gt["keypoints"][1::3])
                gt_kpt_v = np.array(gt["keypoints"][2::3])

                # if the gt match has no keypoint annotations the analysis
                # cannot be carried out.
                if not np.any(gt_kpt_v > 0):
                    zero_kpt_gts += 1
                    continue

                # for every detection match return a dictionary with the following info:
                #  - image_id
                #  - detection_id
                #  - 'corrected_keypoints': list containing good value for each keypoint
                #  - 'jitt': binary list identifying jitter errors
                #  - 'inv':  binary list identifying inversion errors
                #  - 'miss': binary list identifying miss errors
                #  - 'swap': binary list identifying swap errors

                # load all annotations for the image being analyzed
                image_anns = self.cocoGt.loadAnns(
                    self.cocoGt.getAnnIds(imgIds=image_id, catIds=self.params.catIds)
                )
                num_anns = len(image_anns)

                # create a matrix containing 2 * n rows where n is the number of
                # annotations in the image and 2 * n keypoints for x and y coords:
                # - n for all the ground truth with original keypoints
                # - n for all the ground truth with inverted keypoints
                gts_kpt_mat = np.zeros((2 * num_anns, 2 * self.params.num_kpts))
                # keep track of the visibility flags
                vflags = np.zeros((2 * num_anns, self.params.num_kpts))
                # keep track of the area of all annotations
                areas = np.zeros(2 * num_anns)

                # start filling the matrix from row 1 cause row 0 is reserved for
                # the ground truth that is matched with the current detection being
                # analyzed
                indx = 1
                for a in image_anns:
                    # get the keypoint vector and its inverted version
                    xs = np.array(a["keypoints"][0::3])
                    ys = np.array(a["keypoints"][1::3])
                    vs = np.array(a["keypoints"][2::3])
                    inv_vs = vs[self.params.inv_idx]

                    keypoints = np.insert(
                        ys.astype(float), np.arange(self.params.num_kpts), xs.astype(float)
                    )
                    inv_keypoints = np.insert(
                        ys[self.params.inv_idx].astype(float),
                        np.arange(self.params.num_kpts),
                        xs[self.params.inv_idx].astype(float),
                    )

                    if a["id"] == gt["id"]:
                        # if current annotation is the ground truth match
                        # fill in all the above matrices at index 0 and num_anns
                        areas[0] = self._gt_area(a)
                        areas[num_anns] = self._gt_area(a)

                        gts_kpt_mat[0, :] = keypoints
                        gts_kpt_mat[num_anns, :] = inv_keypoints

                        vflags[0, :] = vs
                        vflags[num_anns, :] = inv_vs

                    else:
                        # if current annotation is NOT the ground truth match
                        # fill in all the above matrices at index "indx" and indx + num_anns
                        areas[indx] = self._gt_area(a)
                        areas[indx + num_anns] = self._gt_area(a)

                        gts_kpt_mat[indx, :] = keypoints
                        gts_kpt_mat[indx + num_anns, :] = inv_keypoints

                        vflags[indx, :] = vs
                        vflags[indx + num_anns, :] = inv_vs

                        # increase the index for storing the next annotation info
                        indx += 1

                # compute OKS of every individual dt keypoint with corresponding gt
                dist = gts_kpt_mat - dt_kpt_arr
                sqrd_dist = np.add.reduceat(
                    np.square(dist), range(0, 2 * self.params.num_kpts, 2), axis=1
                )

                # get the keypoint similarity between every individual keypoint
                # detection and its corresponding ground truth location
                kpts_oks_mat = np.exp(
                    -sqrd_dist
                    / (self.params.sigmas * 2) ** 2
                    / (areas[:, np.newaxis] + np.spacing(1))
                    / 2
                ) * (vflags > 0) - 1 * (vflags == 0)
                div = np.sum(vflags > 0, axis=1)
                div[div == 0] = self.params.num_kpts
                oks_mat = (np.sum(kpts_oks_mat * (vflags > 0), axis=1) / div) * (
                    np.sum(vflags > 0, axis=1) > 0
                ) - 1 * (np.sum(vflags > 0, axis=1) == 0)
                assert np.isclose(oks_mat[0], dtm["oks"], atol=1e-08)

                # NOTE: if a 0 or a -1 appear in the oks_max array it doesn't matter
                # since that will automatically become a miss
                oks_max = np.amax(kpts_oks_mat, axis=0)
                assert np.all(vflags[:, np.where(oks_max < 0)] == 0)
                oks_max[np.where(oks_max < 0)] = 0
                oks_argmax = np.argmax(kpts_oks_mat, axis=0)

                # good keypoints are those that have oks max > 0.85 and argmax 0
                good_kpts = (
                    np.logical_and.reduce(
                        (oks_max >= self.params.jitterKsThrs[1], oks_argmax == 0, gt_kpt_v != 0)
                    )
                    * 1
                )

                # jitter keypoints have  0.5 <= oksm < 0.85 and oks_argmax == 0
                jitt_kpts = np.logical_and.reduce(
                    (
                        oks_max >= self.params.jitterKsThrs[0],
                        oks_max < self.params.jitterKsThrs[1],
                        oks_argmax == 0,
                    )
                )
                jitt_kpts = np.logical_and(jitt_kpts, gt_kpt_v != 0) * 1

                # inverted keypoints are those that have oks => 0.5 but on the inverted keypoint entry
                inv_kpts = (
                    np.logical_and.reduce(
                        (
                            oks_max >= self.params.jitterKsThrs[0],
                            oks_argmax == num_anns,
                            gt_kpt_v != 0,
                        )
                    )
                    * 1
                )

                # swapped keypoints are those that have oks => 0.5 but on keypoint of other person
                swap_kpts = np.logical_and.reduce(
                    (
                        oks_max >= self.params.jitterKsThrs[0],
                        oks_argmax != 0,
                        oks_argmax != num_anns,
                    )
                )
                swap_kpts = np.logical_and(swap_kpts, gt_kpt_v != 0) * 1

                # missed keypoints are those that have oks max < 0.5
                miss_kpts = np.logical_and(oks_max < self.params.jitterKsThrs[0], gt_kpt_v != 0) * 1

                # compute what it means in terms of pixels to be at a certain oks score
                # for simplicity it's computed only along one dimension and added only to x
                dist_to_oks_low = np.sqrt(
                    -np.log(self.params.jitterKsThrs[0])
                    * 2
                    * self._gt_area(gt)
                    * (self.params.sigmas**2)
                )
                dist_to_oks_high = np.sqrt(
                    -np.log(self.params.jitterKsThrs[1])
                    * 2
                    * self._gt_area(gt)
                    * (self.params.sigmas**2)
                )
                # note that for swaps we use the current ground truth match area because we
                # have to translate the oks to the scale of correct ground truth
                # round oks values to deal with numerical instabilities
                round_oks_max = (
                    oks_max + np.spacing(1) * (oks_max == 0) - np.spacing(1) * (oks_max == 1)
                )

                dist_to_oks_max = np.sqrt(
                    -np.log(round_oks_max) * 2 * self._gt_area(gt) * (self.params.sigmas**2)
                )

                # correct keypoints vectors using info from all the flag vectors
                correct_kpts_x = (
                    dt_kpt_x * good_kpts
                    + (gt_kpt_x + dist_to_oks_high) * jitt_kpts
                    + (gt_kpt_x + dist_to_oks_low) * miss_kpts
                    + dt_kpt_x * (gt_kpt_v == 0)
                    + (gt_kpt_x + dist_to_oks_max) * inv_kpts
                    + (gt_kpt_x + dist_to_oks_max) * swap_kpts
                )

                correct_kpts_y = (
                    dt_kpt_y * good_kpts
                    + gt_kpt_y * jitt_kpts
                    + gt_kpt_y * miss_kpts
                    + dt_kpt_y * (gt_kpt_v == 0)
                    + gt_kpt_y * inv_kpts
                    + gt_kpt_y * swap_kpts
                )

                correct_kpts = np.zeros(self.params.num_kpts * 3).tolist()
                correct_kpts[0::3] = correct_kpts_x.tolist()
                correct_kpts[1::3] = correct_kpts_y.tolist()
                correct_kpts[2::3] = dt_kpt_v

                new_dt = {}
                new_dt["id"] = dt["id"]
                new_dt["image_id"] = int(dt["image_id"])
                new_dt["keypoints"] = correct_kpts
                new_dt["good"] = good_kpts.tolist()
                new_dt["jitter"] = jitt_kpts.tolist()
                new_dt["inversion"] = inv_kpts.tolist()
                new_dt["swap"] = swap_kpts.tolist()
                new_dt["miss"] = miss_kpts.tolist()

                corrected_dts[areaRngLbl].append(new_dt)
        return corrected_dts

    def _correct_dt_keypoints(self, areaRngLbl):
        # change the detections in the cocoEval object to the corrected kpts
        for cdt in self.corrected_dts[areaRngLbl]:
            if "opt_keypoints" not in cdt:
                continue
            dtid = cdt["id"]
            image_id = cdt["image_id"]

            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d["id"] == dtid:
                    err_kpts_mask = np.zeros(len(cdt["good"]))
                    if "miss" in self.params.err_types:
                        err_kpts_mask += np.array(cdt["miss"])

                    if "swap" in self.params.err_types:
                        err_kpts_mask += np.array(cdt["swap"])

                    if "inversion" in self.params.err_types:
                        err_kpts_mask += np.array(cdt["inversion"])

                    if "jitter" in self.params.err_types:
                        err_kpts_mask += np.array(cdt["jitter"])

                    d["keypoints"] = cdt["opt_keypoints"] * (
                        np.repeat(err_kpts_mask, 3) == 1
                    ) + cdt["keypoints"] * (np.repeat(err_kpts_mask, 3) == 0)
                    break

    def find_score_errors(self):
        tic = time.time()
        print("Analyzing detection scores...")
        # NOTE: optimal score is measures at the lowest oks evaluation thresh
        self.cocoEval.params.iouThrs = [min(self.params.oksThrs)]
        self.cocoEval.params.maxDets = self.params.teamMaxDets

        # if the keypoint analyisis is required then the keypoints must be
        # corrected at all area range values requested before running scoring analysis
        if self.params.check_kpts:
            evalImgs = []
            for aind, areaRngLbl in enumerate(self.params.areaRngLbl):
                # restore original dts and gt ignore flags for new area range
                self._cleanup()
                self._correct_dt_keypoints(areaRngLbl)

                # run the evaluation with check scores flag
                self.cocoEval.params.areaRng = [self.params.areaRng[aind]]
                self.cocoEval.params.areaRngLbl = [areaRngLbl]
                self.cocoEval.evaluate(check_scores=True)
                evalImgs.extend([e for e in filter(None, self.cocoEval.evalImgs)])
        else:
            # run the evaluation with check scores flag
            self.cocoEval.params.areaRng = self.params.areaRng
            self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
            self.cocoEval.evaluate(check_scores=True)
            evalImgs = [e for e in filter(None, self.cocoEval.evalImgs)]

        for aind, areaRngLbl in enumerate(self.params.areaRngLbl):
            evalImgsArea = [
                e for e in filter(None, evalImgs) if e["aRng"] == self.params.areaRng[aind]
            ]

            max_oks = {}
            for e in evalImgsArea:
                dtIds = e["dtIds"]
                dtScoresMax = e["dtIousMax"]
                for i, j in zip(dtIds, dtScoresMax):
                    max_oks[i] = j
            # if assertion fails not all the detections have been evaluated
            assert len(max_oks) == len(self._dts)

            # Soft-NMS must use the geometry corrected for this area group.
            self._cleanup()
            if self.params.check_kpts:
                self._correct_dt_keypoints(areaRngLbl)
            _soft_nms_dts = self._soft_nms(max_oks)
            for cdt in self.corrected_dts[areaRngLbl]:
                d = _soft_nms_dts[cdt["id"]]
                cdt["opt_score"] = d["opt_score"]
                cdt["max_oks"] = d["max_oks"]

        toc = time.time()
        print(f"DONE (t={toc - tic:0.2f}s).")

    def _soft_nms(self, max_oks):
        _soft_nms_dts = {}

        variances = (self.params.sigmas * 2) ** 2
        for imgId in self.params.imgIds:
            B = []
            D = []
            for d in self.cocoEval._dts[imgId, self.params.catIds[0]]:
                dt = {}
                dt["keypoints"] = d["keypoints"]
                dt["max_oks"] = max_oks[d["id"]]
                dt["opt_score"] = max_oks[d["id"]]
                _soft_nms_dts[d["id"]] = dt
                B.append(dt)
            if len(B) == 0:
                continue

            while len(B) > 0:
                B.sort(key=lambda k: -k["opt_score"])
                M = B[0]
                D.append(M)
                B.remove(M)
                m_kpts = np.array(M["keypoints"])
                m_xs = m_kpts[0::3]
                m_ys = m_kpts[1::3]
                x0, x1, y0, y1 = np.min(m_xs), np.max(m_xs), np.min(m_ys), np.max(m_ys)
                m_area = (x1 - x0) * (y1 - y0)

                for dt in B:
                    d_kpts = np.array(dt["keypoints"])
                    d_xs = d_kpts[0::3]
                    d_ys = d_kpts[1::3]
                    x0, x1, y0, y1 = np.min(d_xs), np.max(d_xs), np.min(d_ys), np.max(d_ys)
                    d_area = (x1 - x0) * (y1 - y0)

                    deltax = d_xs - m_xs
                    deltay = d_ys - m_ys
                    # using the average of both areas as area for oks computation
                    e = (
                        (deltax**2 + deltay**2)
                        / variances
                        / ((0.5 * (m_area + d_area)) + np.spacing(1))
                        / 2
                    )
                    oks = np.sum(np.exp(-e)) / e.shape[0]

                    old_score = dt["opt_score"]
                    e = (oks**2) / 0.5  # .5 is a hyperparameter from soft_nms paper
                    new_score = old_score * np.exp(-e)
                    dt["opt_score"] = new_score
                    # print(old_score, oks, new_score)
        return _soft_nms_dts

    def _correct_dt_scores(self, areaRngLbl):
        # change the detections in the cocoEval object to the corrected score
        for cdt in self.corrected_dts[areaRngLbl]:
            dtid = cdt["id"]
            image_id = cdt["image_id"]
            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d["id"] == dtid:
                    d["score"] = cdt["opt_score"]
                    break

    def find_bckgd_errors(self):
        tic = time.time()
        print("Analyzing background false positives and false negatives...")
        # compute matches with current value of detections to determine new matches
        oksThrs = sorted(self.params.oksThrs)

        for areaRng, areaRngLbl in zip(self.params.areaRng, self.params.areaRngLbl):
            self._cleanup()
            # correct keypoints and score if the analysis flags are True
            if self.params.check_kpts:
                self._correct_dt_keypoints(areaRngLbl)
            if self.params.check_scores:
                self._correct_dt_scores(areaRngLbl)
            # get the matches at all oks thresholds for every area range
            dtMatches, gtMatches = self._find_dt_gt_matches(oksThrs, [areaRng], [areaRngLbl])

            for oind, oks in enumerate(oksThrs):
                dtMatchesAreaOks = dtMatches[areaRngLbl, str(oks)]
                gtMatchesAreaOks = gtMatches[areaRngLbl, str(oks)]

                self.bckgd_err_matches[areaRngLbl, str(oks), "dts"] = dtMatches[
                    areaRngLbl, str(oks)
                ]
                self.bckgd_err_matches[areaRngLbl, str(oks), "gts"] = gtMatches[
                    areaRngLbl, str(oks)
                ]

                # assert that detection and ground truth matches are consistent
                for d in dtMatchesAreaOks:
                    # assert that every detection matched has a corresponding gt in the gt matches dictionary
                    assert dtMatchesAreaOks[d][0]["gtId"] in gtMatchesAreaOks
                    # assert that this detection is in the dt matches of the gt it is matched to
                    assert d in [
                        dt["dtId"] for dt in gtMatchesAreaOks[dtMatchesAreaOks[d][0]["gtId"]]
                    ]

                # assert that all ground truth with multiple detection matches should be ignored
                count = 0
                for g in gtMatchesAreaOks:
                    count += len(gtMatchesAreaOks[g])
                    if len(gtMatchesAreaOks[g]) > 1:
                        # if this gt already has multiple matches assert it is a crowd
                        # since crowd gt can be matched to multiple detections
                        assert self.cocoGt.anns[g]["iscrowd"] == 1
                    assert gtMatchesAreaOks[g][0]["dtId"] in dtMatchesAreaOks
                assert count == len(dtMatchesAreaOks)

                # Use evaluated records so subsets, maxDets and ignore flags agree
                # with the native evaluator (including ID zero).
                fp, fn = set(), set()
                for e in self.cocoEval.evalImgs:
                    if e is None:
                        continue
                    fp.update(
                        did
                        for j, did in enumerate(e["dtIds"])
                        if e["dtMatches"][oind, j] < 0 and not e["dtIgnore"][oind, j]
                    )
                    fn.update(
                        gid
                        for j, gid in enumerate(e["gtIds"])
                        if e["gtMatches"][oind, j] < 0 and not e["gtIgnore"][j]
                    )
                self.false_pos_dts[areaRngLbl, str(oks)] = fp
                self.false_neg_gts[areaRngLbl, str(oks)] = fn

        toc = time.time()
        print(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self, makeplots=False, savedir=None, team_name=None):
        """
        Run the evaluation on the original detections to get the baseline for
        algorithm performance and after correcting the detections.
        """
        if self._analysis_signature is None:
            raise RuntimeError("Please run analyze() successfully first")
        if self._analysis_signature != self._signature():
            raise RuntimeError(
                "Analysis parameters changed; run analyze() again before summarize()"
            )

        self.stats = []
        oksThrs = sorted(self.params.oksThrs)[::-1]
        areaRngLbl = self.params.areaRngLbl
        maxDets = sorted(self.params.maxDets)
        # compute all the precision recall curves and return precise breakdown of
        # all error type in terms of keypoint, scoring, false positives and negatives
        ps_mat, rs_mat = self._summarize_baseline()
        self._evaluation_stats(verbose=False)
        err_types = ["baseline"]
        stats = self._summarize(err_types, ps_mat, rs_mat, oksThrs, areaRngLbl, maxDets)
        self.stats.extend(stats)
        # summarize keypoint errors
        if self.params.check_kpts and self.params.err_types:
            ps_mat_kpt_errors, rs_mat_kpt_errors = self._summarize_kpt_errors()
            ps_mat = np.append(ps_mat, ps_mat_kpt_errors, axis=0)
            err_types = self.params.err_types
            stats = self._summarize(
                err_types, ps_mat_kpt_errors, rs_mat_kpt_errors, oksThrs, areaRngLbl, maxDets
            )
            self.stats.extend(stats)
        # summarize scoring errors
        if self.params.check_scores:
            ps_mat_score_errors, rs_mat_score_errors = self._summarize_score_errors()
            ps_mat = np.append(ps_mat, ps_mat_score_errors, axis=0)
            err_types = ["score"]
            stats = self._summarize(
                err_types, ps_mat_score_errors, rs_mat_score_errors, oksThrs, areaRngLbl, maxDets
            )
            self.stats.extend(stats)
        # summarize detections that are unmatched (hallucinated false positives)
        # and ground truths that are unmatched (false negatives)
        if self.params.check_bckgd:
            ps_mat_bckgd_errors, rs_mat_bckgd_errors = self._summarize_bckgd_errors()
            ps_mat = np.append(ps_mat, ps_mat_bckgd_errors, axis=0)
            err_types = ["bckgd_false_pos", "false_neg"]
            stats = self._summarize(
                err_types, ps_mat_bckgd_errors, rs_mat_bckgd_errors, oksThrs, areaRngLbl, maxDets
            )
            self.stats.extend(stats)

        err_labels = []
        colors_vec = []
        if self.params.check_kpts:
            for err in self.params.err_types:
                if err == "miss":
                    err_labels.append("w/o Miss")
                    colors_vec.append("#F2E394")
                if err == "swap":
                    err_labels.append("w/o Swap")
                    colors_vec.append("#F2AE72")
                if err == "inversion":
                    err_labels.append("w/o Inv.")
                    colors_vec.append("#D96459")
                if err == "jitter":
                    err_labels.append("w/o Jit.")
                    colors_vec.append("#8C4646")

        if self.params.check_scores:
            err_labels += ["Opt. Score"]
            colors_vec += ["#4F82BD"]

        if self.params.check_bckgd:
            err_labels += ["w/o Bkg. FP", "w/o FN"]
            colors_vec += ["#8063A3", "seagreen"]

        if makeplots:
            self._plot(
                self.cocoEval.params.recThrs[:],
                ps_mat,
                self.params,
                err_labels,
                colors_vec,
                savedir,
                team_name,
            )
        self._cleanup()

    def _summarize_baseline(self):
        self._cleanup()
        # set area range and the oks thresholds
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.areaRng = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        ps = self.cocoEval.eval["precision"][::-1, :, :, :, :]
        rs = self.cocoEval.eval["recall"][::-1, :, :, :]
        return ps, rs

    def _summarize_kpt_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        indx_list = [i for i in range(self.params.num_kpts * 3) if (i - 2) % 3 != 0]
        err_types = self.params.err_types
        assert len(err_types) > 0
        T = len(oksThrs)
        E = len(self.params.err_types)
        R = len(self.cocoEval.params.recThrs)
        K = 1
        A = len(self.params.areaRng)
        M = len(self.params.maxDets)
        ps_mat_kpts = np.zeros([T * E, R, K, A, M])
        rs_mat_kpts = np.zeros([T * E, K, A, M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print(f"Correcting area range [{arearnglbl}]:")
            self._cleanup()

            self.cocoEval.params.areaRng = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            corrected_dts = self.corrected_dts[arearnglbl]
            # compute performance after solving for each error type
            for eind, err in enumerate(err_types):
                print(f"Correcting error type [{err}]:")
                tind_start = T * eind
                tind_end = T * (eind + 1)

                for cdt in corrected_dts:
                    # this detection doesn't have keypoint errors (wasn't matched)
                    if err not in cdt:
                        continue
                    # check if detection has error of that type
                    if sum(cdt[err]) != 0:
                        dtid = cdt["id"]
                        image_id = cdt["image_id"]
                        corrected_kpts = np.array(cdt["opt_keypoints"])
                        # correct only those keypoints
                        for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                            if d["id"] == dtid:
                                oth_kpts_mask = np.repeat(np.logical_not(cdt[err]) * 1, 2)
                                err_kpts_mask = np.repeat(cdt[err], 2)
                                all_kpts = np.delete(np.array(d["keypoints"]), slice(2, None, 3))
                                opt_kpts = np.delete(np.array(corrected_kpts), slice(2, None, 3))

                                kpts = all_kpts * oth_kpts_mask + opt_kpts * err_kpts_mask

                                d["keypoints"] = np.array(d["keypoints"], dtype=float)
                                d["keypoints"][indx_list] = kpts
                                d["keypoints"] = d["keypoints"].tolist()
                                break
                self.cocoEval.evaluate()
                self.cocoEval.accumulate()

                ps_mat_kpts[tind_start:tind_end, :, :, aind, :] = self.cocoEval.eval["precision"][
                    ::-1, :, :, 0, :
                ]
                rs_mat_kpts[tind_start:tind_end, :, aind, :] = self.cocoEval.eval["recall"][
                    ::-1, :, 0, :
                ]
        return ps_mat_kpts, rs_mat_kpts

    def _summarize_score_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        T = len(oksThrs)
        R = len(self.cocoEval.params.recThrs)
        K = 1
        A = len(self.params.areaRng)
        M = len(self.params.maxDets)
        ps_mat_score = np.zeros([T, R, K, A, M])
        rs_mat_score = np.zeros([T, K, A, M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print(f"Correcting area range [{arearnglbl}]:")
            print("Correcting error type [{}]:".format("score"))
            self._cleanup()

            self.cocoEval.params.areaRng = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            if self.params.check_kpts:
                self._correct_dt_keypoints(arearnglbl)
            self._correct_dt_scores(arearnglbl)

            self.cocoEval.evaluate()
            self.cocoEval.accumulate()

            # insert results into the precision matrix
            ps_mat_score[:, :, :, aind, :] = self.cocoEval.eval["precision"][::-1, :, :, 0, :]
            rs_mat_score[:, :, aind, :] = self.cocoEval.eval["recall"][::-1, :, 0, :]
        return ps_mat_score, rs_mat_score

    def _summarize_bckgd_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        T = len(oksThrs)
        R = len(self.cocoEval.params.recThrs)
        K = 1
        A = len(self.params.areaRng)
        M = len(self.params.maxDets)
        ps_mat_false_pos = np.zeros([T, R, K, A, M])
        rs_mat_false_pos = np.zeros([T, K, A, M])
        ps_mat_false_neg = np.zeros([T, R, K, A, M])
        rs_mat_false_neg = np.zeros([T, K, A, M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print(f"Correcting area range [{arearnglbl}]:")
            print("Correcting error type [{}]:".format("bckgd. fp, fn"))
            self._cleanup()

            self.cocoEval.params.areaRng = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            if self.params.check_kpts:
                self._correct_dt_keypoints(arearnglbl)
            if self.params.check_scores:
                self._correct_dt_scores(arearnglbl)

            self.cocoEval.evaluate()
            self.cocoEval.accumulate()

            for oind, oks in enumerate(oksThrs):
                # set unmatched detections to ignore and remeasure performance
                for e in self.cocoEval.evalImgs:
                    if e is None:
                        continue
                    for dind, dtid in enumerate(e["dtIds"]):
                        # check if detection is a background false pos at this oks
                        if dtid in self.false_pos_dts[arearnglbl, str(oks)]:
                            e["dtIgnore"][oind][dind] = True
                # accumulate results after having set all this ignores
                self.cocoEval.accumulate()
            ps_mat_false_pos[:, :, :, aind, :] = self.cocoEval.eval["precision"][::-1, :, :, 0, :]
            rs_mat_false_pos[:, :, aind, :] = self.cocoEval.eval["recall"][::-1, :, 0, :]

            original_ignores = [
                None if e is None else e["gtIgnore"].copy() for e in self.cocoEval.evalImgs
            ]
            for oind, oks in enumerate(oksThrs):
                # GT ignores are threshold specific; restore before each slice.
                for e, original_ignore in zip(self.cocoEval.evalImgs, original_ignores):
                    if e is None:
                        continue
                    e["gtIgnore"] = original_ignore.copy()
                    for gind, gtid in enumerate(e["gtIds"]):
                        if gtid in self.false_neg_gts[arearnglbl, str(oks)]:
                            e["gtIgnore"][gind] = 1
                # accumulate results after having set all this ignores
                self.cocoEval.accumulate()
                ps_mat_false_neg[T - 1 - oind, :, :, aind, :] = self.cocoEval.eval["precision"][
                    oind, :, :, 0, :
                ]
                rs_mat_false_neg[T - 1 - oind, :, aind, :] = self.cocoEval.eval["recall"][
                    oind, :, 0, :
                ]

        ps = np.append(ps_mat_false_pos, ps_mat_false_neg, axis=0)
        rs = np.append(rs_mat_false_pos, rs_mat_false_neg, axis=0)
        return ps, rs

    @staticmethod
    def _summarize(err_types, ps_mat, rs_mat, oksThrs, areaRngLbl, maxDets):
        stats = []
        l = len(oksThrs)

        for eind, err in enumerate(err_types):
            ps_mat_err_slice = ps_mat[eind * l : (eind + 1) * l, :, :, :, :]
            rs_mat_err_slice = rs_mat[eind * l : (eind + 1) * l, :, :, :]

            for oind, oks in enumerate(oksThrs):
                for aind, arearng in enumerate(areaRngLbl):
                    for mind, maxdts in enumerate(maxDets):
                        stat = {}
                        stat["oks"] = oks
                        stat["areaRngLbl"] = arearng
                        stat["maxDets"] = maxdts
                        stat["err"] = err

                        p = ps_mat_err_slice[oind, :, :, aind, mind]
                        r = rs_mat_err_slice[oind, :, aind, mind]

                        stat["auc"] = -1 if len(p[p > -1]) == 0 else np.mean(p[p > -1])
                        stat["recall"] = -1 if len(r[r > -1]) == 0 else np.mean(r[r > -1])
                        stats.append(stat)
        return stats

    def _cleanup(self):
        # restore detections and gt ignores to their original value
        for d in self._dts:
            d["keypoints"] = copy.deepcopy(self._original_dts[d["id"]]["keypoints"])
            d["score"] = self._original_dts[d["id"]]["score"]
        for g in self._gts:
            g["_ignore"] = 0

    @staticmethod
    def _plot(
        recalls, ps_mat, params, err_labels=None, color_vec=None, savedir=None, team_name=None
    ):
        if color_vec is None:
            color_vec = []
        if err_labels is None:
            err_labels = []
        iouThrs = sorted(params.oksThrs, reverse=True)
        areaRngLbl = params.areaRngLbl
        maxDets = params.maxDets
        catId = 0

        if err_labels:
            labels = ["Orig. Dts."] + err_labels
            colors = plt.cm.Greens(np.linspace(0, 0.75, len(labels)))
            colors[-len(err_labels) :] = [to_rgba(c) for c in color_vec]
        else:
            labels = [f"Oks {o:.2f}" for o in iouThrs]
            colors = plt.cm.Greens(np.linspace(0, 0.75, len(labels)))

        for aind, a in enumerate(areaRngLbl):
            for mind, m in enumerate(maxDets):
                if not err_labels:
                    fig = plt.figure(figsize=(10, 8))
                    fig.add_axes([0.1, 0.15, 0.56, 0.7])
                    plt.title(f"areaRng:[{a}], maxDets:[{m}]", fontsize=18)
                    oks_ps_mat = ps_mat

                for tind, t in enumerate(iouThrs):
                    legend_patches = []
                    if err_labels:
                        fig = plt.figure(figsize=(10, 8))
                        fig.add_axes([0.1, 0.15, 0.56, 0.7])
                        plt.title(f"oksThrs:[{t}], areaRng:[{a}], maxDets:[{m}]", fontsize=18)
                        thresh_idx = [tind + i * len(iouThrs) for i in range(len(labels))]
                        oks_ps_mat = ps_mat[thresh_idx, :, :, :, :]

                    for lind, l in enumerate(labels):
                        precisions = oks_ps_mat[lind, :, catId, aind, mind]
                        plt.plot(recalls, precisions, c="k", ls="-", lw=2)

                        if lind > 0:
                            prev_precisions = oks_ps_mat[lind - 1, :, catId, aind, mind]
                            plt.fill_between(
                                recalls,
                                prev_precisions,
                                precisions,
                                where=precisions >= prev_precisions,
                                facecolor=colors[lind],
                                interpolate=True,
                            )

                        valid = precisions[precisions >= 0]
                        m_map = float(valid.mean()) if valid.size else 0.0
                        interm_m_map = f"{m_map:.3f}"
                        m_map_val_str = interm_m_map[
                            1 - int(interm_m_map[0]) : 5 - int(interm_m_map[0])
                        ]

                        if err_labels:
                            the_label = f"{l:<11}: {m_map_val_str}"
                        else:
                            the_label = f"{l:<7}: {m_map_val_str}"

                        patch = mpatches.Patch(
                            facecolor=colors[lind], edgecolor="k", linewidth=1.5, label=the_label
                        )
                        legend_patches.append(patch)

                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.grid()
                    plt.xlabel("recall", fontsize=18)
                    plt.ylabel("precision", fontsize=18)
                    plt.legend(
                        handles=legend_patches[::-1],
                        ncol=1,
                        bbox_to_anchor=(1, 1),
                        loc="upper left",
                        fancybox=True,
                        shadow=True,
                        fontsize=18,
                    )

                    if savedir == None:
                        plt.show()
                    else:
                        prefix = "error_prc" if err_labels else "prc"
                        oks_str = f"[{int(100 * t)}]" if err_labels else ""
                        savepath = f"{savedir}/{prefix}_[{team_name}]{oks_str}[{a}][{m}].pdf"
                        plt.savefig(savepath, bbox_inches="tight")
                        plt.close()

                    if not err_labels:
                        break

    def __str__(self):
        return str(self.stats)


class Params:
    # Params for coco analyze api
    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        self.kpts_name = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.inv_kpts_name = [
            "nose",
            "right_eye",
            "left_eye",
            "right_ear",
            "left_ear",
            "right_shoulder",
            "left_shoulder",
            "right_elbow",
            "left_elbow",
            "right_wrist",
            "left_wrist",
            "right_hip",
            "left_hip",
            "right_knee",
            "left_knee",
            "right_ankle",
            "left_ankle",
        ]
        self.num_kpts = len(self.kpts_name)
        self.inv_idx = [self.inv_kpts_name.index(self.kpts_name[i]) for i in range(self.num_kpts)]
        self.sigmas = np.array(
            [
                0.026,
                0.025,
                0.025,
                0.035,
                0.035,
                0.079,
                0.079,
                0.072,
                0.072,
                0.062,
                0.062,
                0.107,
                0.107,
                0.087,
                0.087,
                0.089,
                0.089,
            ]
        )
        self.oksThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        # the threshold that determines the limit for localization error
        self.oksLocThrs = 0.1
        # oks thresholds that define a jitter error
        self.jitterKsThrs = [0.5, 0.85]
        self.maxDets = [20]
        self.teamMaxDets = []
        self.areaRng = [[0, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.err_types = ["miss", "swap", "inversion", "jitter"]
        self.check_kpts = True
        self.check_scores = True
        self.check_bckgd = True

    def __init__(self, iouType="keypoints"):
        if iouType in ("keypoints", "keypoints_crowd"):
            self.setKpParams()
        else:
            raise ValueError(
                f"Analysis does not support iouType={iouType!r}; use keypoints or keypoints_crowd"
            )
        self.iouType = iouType
