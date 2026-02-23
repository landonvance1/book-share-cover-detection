#!/usr/bin/env python
"""
Capture real Florence-2 ONNX OCR output for the 7 test images and save as
JSON fixtures used by the GLiNER integration tests.

Run this once to generate (or regenerate) the fixtures:
    python tests/integration/generate_ocr_fixtures.py

Requires the ONNX model to be present:
    huggingface-cli download onnx-community/Florence-2-base-ft --local-dir florence2-onnx
"""
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = Path(__file__).parent / "images"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
ONNX_MODEL_DIR = REPO_ROOT / "florence2-onnx"

IMAGES = [
    "a-restless-truth.jpg",
    "gardens-of-the-moon.jpg",
    "jade-city.jpg",
    "mistborn.jpg",
    "snow-crash.jpg",
    "to-kill-a-mockingbird.jpg",
    "under-the-whispering-door.jpg",
]


async def main() -> None:
    if not ONNX_MODEL_DIR.exists():
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_DIR}", file=sys.stderr)
        print("Run: huggingface-cli download onnx-community/Florence-2-base-ft --local-dir florence2-onnx")
        sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT))
    from app.engines.florence2_onnx_engine import Florence2OnnxEngine

    print("Loading Florence-2 ONNX engine...")
    engine = Florence2OnnxEngine(model_path=str(ONNX_MODEL_DIR))

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for image_name in IMAGES:
        image_bytes = (IMAGES_DIR / image_name).read_bytes()
        print(f"  {image_name} ...", end=" ", flush=True)
        result = await engine.extract_text(image_bytes)
        stem = Path(image_name).stem
        (FIXTURES_DIR / f"{stem}.json").write_text(
            result.model_dump_json(indent=2)
        )
        print(repr(result.text))

    print(f"\n{len(IMAGES)} fixtures written to {FIXTURES_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    asyncio.run(main())
