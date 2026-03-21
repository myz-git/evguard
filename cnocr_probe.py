#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image, ImageGrab

try:
    from cnocr import CnOcr
except Exception as exc:  # pragma: no cover
    print(f"Failed to import cnocr: {exc}", file=sys.stderr)
    raise


KNOWN_MODELS = [
    "densenet_lite_136-gru",
    "densenet_lite_246-gru_base",
    "doc-densenet_lite_246-gru_base",
    "ppocr",
    "scene-densenet_lite_246-gru_base",
]


def _default_cache_root() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    return Path(appdata) / "cnocr"


def _find_model_file(model_dir: Path) -> Optional[Path]:
    for suffix in (".onnx", ".ckpt"):
        files = sorted(p for p in model_dir.glob(f"*{suffix}") if p.is_file())
        if files:
            return files[0]
    return None


def discover_candidates(project_dir: Path, cache_root: Path) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []

    project_version_root = project_dir / "2.3"
    if project_version_root.is_dir():
        candidates.append(
            {
                "id": "project-root:auto",
                "mode": "project-root",
                "model_name": "<default>",
                "path": str(project_version_root),
                "rec_root": str(project_dir),
                "det_root": str(project_dir),
            }
        )

    for model_name in KNOWN_MODELS:
        cache_dir = cache_root / "2.3" / model_name
        if cache_dir.is_dir():
            candidates.append(
                {
                    "id": f"cache:{model_name}",
                    "mode": "cache-root",
                    "model_name": model_name,
                    "path": str(cache_dir),
                    "rec_root": str(cache_root),
                }
            )

        local_dir = project_dir / model_name
        if local_dir.is_dir():
            model_fp = _find_model_file(local_dir)
            if model_fp is not None:
                candidates.append(
                    {
                        "id": f"local-file:{model_name}",
                        "mode": "direct-file",
                        "model_name": model_name,
                        "path": str(model_fp),
                        "rec_model_fp": str(model_fp),
                    }
                )

    return candidates


def build_ocr(candidate: Dict[str, str], det_model_name: str) -> CnOcr:
    if candidate["mode"] == "project-root":
        kwargs = {
            "rec_root": candidate["rec_root"],
            "det_root": candidate["det_root"],
        }
    elif candidate["mode"] == "cache-root":
        kwargs = {
            "rec_model_name": candidate["model_name"],
            "rec_root": candidate["rec_root"],
        }
    elif candidate["mode"] == "direct-file":
        rec_model_fp = candidate["rec_model_fp"]
        kwargs = {
            "rec_model_name": candidate["model_name"],
            "det_model_name": det_model_name,
            "rec_model_fp": rec_model_fp,
            "rec_model_backend": "onnx" if rec_model_fp.lower().endswith(".onnx") else "pytorch",
        }
    else:
        raise ValueError(f"Unsupported candidate mode: {candidate['mode']}")
    return CnOcr(**kwargs)


def load_image(image_path: Optional[str], screenshot: bool, region: Optional[Tuple[int, int, int, int]]) -> Union[str, Image.Image]:
    if image_path:
        return str(Path(image_path).resolve())
    if screenshot:
        bbox = None
        if region is not None:
            x, y, w, h = region
            bbox = (x, y, x + w, y + h)
        return ImageGrab.grab(bbox=bbox).convert("RGB")
    raise ValueError("Either --image or --screenshot is required")


def parse_region(raw: str) -> Tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("Region must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def ocr_to_texts(result: Iterable[Dict]) -> List[str]:
    return [item.get("text", "") for item in result if item.get("text")]


def print_candidate(candidate: Dict[str, str]) -> None:
    print(f"[{candidate['id']}]")
    print(f"  mode      : {candidate['mode']}")
    print(f"  model     : {candidate['model_name']}")
    print(f"  source    : {candidate['path']}")
    if candidate["mode"] == "project-root":
        print(f"  rec_root  : {candidate['rec_root']}")
        print(f"  det_root  : {candidate['det_root']}")


def run_candidate(candidate: Dict[str, str], image: Union[str, Image.Image], det_model_name: str, expected: Optional[str]) -> int:
    print_candidate(candidate)
    started = time.time()
    try:
        ocr = build_ocr(candidate, det_model_name=det_model_name)
        init_elapsed = time.time() - started
        print(f"  init_sec  : {init_elapsed:.3f}")
    except Exception as exc:
        print(f"  status    : INIT_FAILED")
        print(f"  error     : {exc}")
        print()
        return 2

    try:
        started = time.time()
        result = ocr.ocr(image)
        ocr_elapsed = time.time() - started
        texts = ocr_to_texts(result)
        joined = " | ".join(texts)
        print("  status    : OK")
        print(f"  ocr_sec   : {ocr_elapsed:.3f}")
        print(f"  texts     : {joined if joined else '<empty>'}")
        if expected:
            matched = any(expected in text for text in texts)
            print(f"  expected  : {expected}")
            print(f"  matched   : {matched}")
            print()
            return 0 if matched else 1
        print()
        return 0
    except Exception as exc:
        print(f"  status    : OCR_FAILED")
        print(f"  error     : {exc}")
        print()
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local/cache CnOcr models against an image or screenshot.")
    parser.add_argument("--image", help="Image path to OCR")
    parser.add_argument("--screenshot", action="store_true", help="Capture a screenshot instead of reading a file")
    parser.add_argument("--region", help="Screenshot region x,y,w,h")
    parser.add_argument("--expected", help="Expected substring to verify in OCR output")
    parser.add_argument("--det-model", default="naive_det", help="Detection model name, default: naive_det")
    parser.add_argument("--cache-root", default=str(_default_cache_root()), help="CnOcr cache root, default: %%APPDATA%%\\cnocr\\2.3")
    parser.add_argument("--project-dir", default=os.getcwd(), help="Project root to scan for local model dirs")
    parser.add_argument("--model", action="append", help="Restrict test to one or more model names")
    parser.add_argument("--list-only", action="store_true", help="Only list discovered candidates")
    args = parser.parse_args()

    region = parse_region(args.region) if args.region else None
    project_dir = Path(args.project_dir).resolve()
    cache_root = Path(args.cache_root).resolve()

    candidates = discover_candidates(project_dir=project_dir, cache_root=cache_root)
    if args.model:
        allowed = set(args.model)
        candidates = [candidate for candidate in candidates if candidate["model_name"] in allowed]

    if not candidates:
        print("No candidates discovered.")
        return 1

    print(f"Project dir : {project_dir}")
    print(f"Cache root  : {cache_root}")
    print(f"Det model   : {args.det_model}")
    print()

    for candidate in candidates:
        print_candidate(candidate)
        print()

    if args.list_only:
        return 0

    image = load_image(image_path=args.image, screenshot=args.screenshot, region=region)
    status_codes = [
        run_candidate(candidate, image=image, det_model_name=args.det_model, expected=args.expected)
        for candidate in candidates
    ]
    return 0 if any(code == 0 for code in status_codes) else max(status_codes)


if __name__ == "__main__":
    raise SystemExit(main())
