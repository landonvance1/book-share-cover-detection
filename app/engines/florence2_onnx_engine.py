import asyncio
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from app.config import settings
from app.engines.florence2_engine import _build_ocr_result
from app.interfaces.ocr import OcrEngine
from app.models import OcrResult

_NUM_LAYERS = 6


class Florence2OnnxEngine(OcrEngine):
    """Florence-2 OCR engine using ONNX Runtime for inference.

    Loads pre-exported ONNX models from onnx-community/Florence-2-base-ft
    and runs inference without requiring trust_remote_code for the model
    forward pass. The AutoProcessor still requires trust_remote_code for
    Florence-2's custom post-processing logic.

    Uses the merged decoder model which combines prefill and decode-with-past
    into a single ONNX graph controlled by a ``use_cache_branch`` boolean.
    """

    def __init__(
        self,
        model_path: str,
        quantization: str = "q4",
        processor_name: str = "microsoft/Florence-2-base-ft",
        intra_op_num_threads: int | None = None,
    ) -> None:
        onnx_dir = Path(model_path) / "onnx"
        suffix = f"_{quantization}" if quantization else ""
        threads = intra_op_num_threads if intra_op_num_threads is not None else settings.onnx_num_threads

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

        self._vision_encoder = ort.InferenceSession(
            str(onnx_dir / f"vision_encoder{suffix}.onnx"), opts
        )
        self._embed_tokens = ort.InferenceSession(
            str(onnx_dir / f"embed_tokens{suffix}.onnx"), opts
        )
        self._encoder = ort.InferenceSession(
            str(onnx_dir / f"encoder_model{suffix}.onnx"), opts
        )
        self._decoder = ort.InferenceSession(
            str(onnx_dir / f"decoder_model_merged{suffix}.onnx"), opts
        )

        self._processor = AutoProcessor.from_pretrained(
            processor_name, trust_remote_code=True, local_files_only=True
        )
        self._eos_token_id = self._processor.tokenizer.eos_token_id

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._run_ocr(image))

    def _run_ocr(self, image: Image.Image) -> OcrResult:
        task = "<OCR_WITH_REGION>"
        inputs = self._processor(text=task, images=image, return_tensors="np")

        # Stage 1: Vision encoding
        pixel_values = inputs["pixel_values"].astype(np.float32)
        image_features = self._vision_encoder.run(
            None, {"pixel_values": pixel_values}
        )[0]

        # Stage 2: Text embedding + encoder
        input_ids = inputs["input_ids"].astype(np.int64)
        prompt_embeds = self._embed_tokens.run(
            None, {"input_ids": input_ids}
        )[0]
        combined_embeds = np.concatenate(
            [image_features, prompt_embeds], axis=1
        )
        combined_mask = np.ones(
            (1, combined_embeds.shape[1]), dtype=np.int64
        )
        encoder_hidden = self._encoder.run(None, {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_mask,
        })[0]

        # Stage 3: Greedy autoregressive decode
        generated_ids = self._greedy_decode(encoder_hidden, combined_mask)

        # Stage 4: Post-process
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )
        return _build_ocr_result(parsed[task])

    def _greedy_decode(self, encoder_hidden, attention_mask, max_tokens=1024):
        # Seed decoder with EOS token (BART convention: decoder_start = EOS)
        seed_ids = np.array([[self._eos_token_id]], dtype=np.int64)
        seed_embeds = self._embed_tokens.run(
            None, {"input_ids": seed_ids}
        )[0]

        # Prefill: use_cache_branch=False, pass empty KV cache tensors
        empty_kv = np.zeros((1, 12, 0, 64), dtype=np.float32)
        feed = {
            "inputs_embeds": seed_embeds,
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask,
            "use_cache_branch": np.array([False]),
        }
        for layer in range(_NUM_LAYERS):
            feed[f"past_key_values.{layer}.decoder.key"] = empty_kv
            feed[f"past_key_values.{layer}.decoder.value"] = empty_kv
            feed[f"past_key_values.{layer}.encoder.key"] = empty_kv
            feed[f"past_key_values.{layer}.encoder.value"] = empty_kv

        outs = self._decoder.run(None, feed)
        logits = outs[0]
        kv_out_names = [o.name for o in self._decoder.get_outputs()[1:]]
        kv_cache = dict(zip(kv_out_names, outs[1:]))

        # Encoder KV is computed once during prefill and reused for all
        # decode steps. The merged decoder's If node corrupts the encoder
        # KV pass-through on the use_cache_branch=True path, so we pin it.
        encoder_kv = {
            k: v for k, v in kv_cache.items() if ".encoder." in k
        }

        tokens = [self._eos_token_id]
        for _ in range(max_tokens):
            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            tokens.append(next_token)
            if next_token == self._eos_token_id:
                break

            # Embed next token
            next_embeds = self._embed_tokens.run(None, {
                "input_ids": np.array([[next_token]], dtype=np.int64),
            })[0]

            # Decode step: use_cache_branch=True, pass KV cache
            feed = {
                "inputs_embeds": next_embeds,
                "encoder_hidden_states": encoder_hidden,
                "encoder_attention_mask": attention_mask,
                "use_cache_branch": np.array([True]),
            }
            for out_name, kv_val in kv_cache.items():
                in_name = out_name.replace("present", "past_key_values")
                # Use pinned encoder KV; decoder KV comes from latest output
                if ".encoder." in out_name:
                    feed[in_name] = encoder_kv[out_name]
                else:
                    feed[in_name] = kv_val

            outs = self._decoder.run(None, feed)
            logits = outs[0]
            kv_cache = dict(zip(kv_out_names, outs[1:]))

        return [tokens]
