"""Freeze a reference from an unmodified checkout of matteorr/coco-analyze.

Usage: .venv/bin/python scripts/generate_legacy_reference.py /path/to/coco-analyze
Only Python/NumPy compatibility and unused plotting imports are patched in memory.
No upstream files are edited. Requires the test environment (including pytest).
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from xtcocotools import mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = runpy.run_path(str(root / "tests/test_analyze.py"))["fixture"]
    package = types.ModuleType("legacy_coco")
    package.__path__ = []
    sys.modules["legacy_coco"] = package
    sys.modules["legacy_coco.mask"] = mask  # keypoint tests never call mask operations
    hashes = {}
    for name in ["cocoeval", "cocoanalyze"]:
        path = args.upstream / "pycocotools" / f"{name}.py"
        raw = path.read_text()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = raw.replace("np.float", "float")
        source = source.replace(
            "np.round((0.95 - .5) / .05) + 1", "int(np.round((0.95 - .5) / .05)) + 1"
        )
        source = source.replace(
            "np.round((1.00 - .0) / .01) + 1", "int(np.round((1.00 - .0) / .01)) + 1"
        )
        source = source.replace("from colour import Color\n", "")
        source = source.replace("import skimage.io as io\n", "")
        spec = importlib.util.spec_from_loader(f"legacy_coco.{name}", loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - explicit reference source
    cls = sys.modules["legacy_coco.cocoanalyze"].COCOanalyze
    with contextlib.redirect_stdout(io.StringIO()):
        gt, dt = fixture(area=True)
        a = cls(copy.deepcopy(gt), copy.deepcopy(dt))
        # One area isolates the preserved algorithm from upstream's area-order bug.
        a.params.areaRng, a.params.areaRngLbl = [[0, 1e10]], ["all"]
        a.analyze()
        a.summarize()
    frozen = {
        "source": "https://github.com/matteorr/coco-analyze",
        "commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sha256": hashes,
        "corrected_dts": a.corrected_dts,
        # The upstream FN tensor is indexed in the wrong threshold order. Test it
        # separately against independently evaluated native slices, not this bug.
        "stats": [s for s in a.stats if s["err"] != "false_neg"],
    }
    out = root / "tests/fixtures/legacy_analysis.json"
    out.write_text(json.dumps(frozen, indent=2, default=lambda v: np.asarray(v).tolist()) + "\n")
    print(out)


if __name__ == "__main__":
    main()
