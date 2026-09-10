"""Diagnostic fields over native xtcocotools matching, without replacing it."""

import numpy as np

from .cocoeval import COCOeval


class AnalyzeEval(COCOeval):
    def evaluate(self, check_scores=False):
        self.params.iouThrs = np.asarray(self.params.iouThrs, dtype=float)
        super().evaluate()
        for record in self.evalImgs:
            if record is None:
                continue
            key = (record["image_id"], record["category_id"])
            # computeOks rows are stably sorted by score; columns are in _gts
            # order. evaluateImg reorders columns by ignore, so map by ID.
            raw_gt_index = {g["id"]: i for i, g in enumerate(self._gts[key])}
            gt_index = {gid: i for i, gid in enumerate(record["gtIds"])}
            dt_index = {did: i for i, did in enumerate(record["dtIds"])}
            oks = np.asarray(self.ious[key])
            shape = (len(record["dtIds"]), len(record["gtIds"]))
            ordered = np.zeros(shape, dtype=float)
            if oks.size:
                ordered[:] = oks[: shape[0], [raw_gt_index[gid] for gid in record["gtIds"]]]
            dt_oks = np.zeros_like(record["dtMatches"], dtype=float)
            gt_oks = np.zeros_like(record["gtMatches"], dtype=float)
            for ti in range(len(self.params.iouThrs)):
                for di, gid in enumerate(record["dtMatches"][ti]):
                    if gid >= 0:
                        dt_oks[ti, di] = ordered[di, gt_index[gid]]
                for gi, did in enumerate(record["gtMatches"][ti]):
                    if did >= 0:
                        gt_oks[ti, gi] = ordered[dt_index[did], gi]
            record["dtIous"] = dt_oks
            record["gtIous"] = gt_oks
            if check_scores:
                valid = record["gtIgnore"] == 0
                record["dtIousMax"] = (
                    ordered[:, valid].max(axis=1).tolist() if valid.any() else [0.0] * shape[0]
                )
