import asyncio
import io

import torch
import transformers.dynamic_module_utils as _dmu
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from app.interfaces.ocr import OcrEngine
from app.models import OcrBoundingBox, OcrResult

# Florence-2's modeling file unconditionally imports flash_attn, which is
# CUDA-only and cannot be installed on CPU. Patch get_imports so the
# transformers import-checker skips it; Florence-2 only *uses* flash_attn
# when _attn_implementation="flash_attention_2", which we never enable.
_orig_get_imports = _dmu.get_imports


def _get_imports_no_flash_attn(filename: str) -> list[str]:
    return [imp for imp in _orig_get_imports(filename) if imp != "flash_attn"]


_dmu.get_imports = _get_imports_no_flash_attn


class Florence2OcrEngine(OcrEngine):
    def __init__(self, model_name: str = "microsoft/Florence-2-base", gpu: bool = False, num_beams: int = 1) -> None:
        device = "cuda" if gpu else "cpu"
        dtype = torch.float16 if gpu else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._device = device
        self._dtype = dtype
        self._num_beams = num_beams

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._run_ocr(image))

    def _run_ocr(self, image: Image.Image) -> OcrResult:
        task = "<OCR_WITH_REGION>"
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        pixel_values = inputs["pixel_values"].to(self._device, self._dtype)
        generated_ids = self._model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            num_beams=self._num_beams,
            do_sample=False,
            early_stopping=self._num_beams > 1,
        )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )
        return _build_ocr_result(parsed[task])


def _build_ocr_result(ocr_data: dict) -> OcrResult:
    # quad is flat [x1,y1,x2,y2,x3,y3,x4,y4] in pixel-space
    regions = [
        OcrBoundingBox(
            text=label,
            confidence=1.0,  # Florence-2 has no per-region confidence
            coordinates=[[q[i], q[i + 1]] for i in range(0, 8, 2)],
        )
        for q, label in zip(
            ocr_data.get("quad_boxes", []),
            ocr_data.get("labels", []),
        )
    ]
    return OcrResult(text=" ".join(r.text for r in regions), regions=regions)
