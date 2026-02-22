import asyncio
import io
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

from app.config import settings
from app.engines.florence2_engine import _build_ocr_result
from app.interfaces.ocr import OcrEngine
from app.models import OcrResult

_NUM_LAYERS = 6
_VOCAB_SIZE = 51289
_EMBED_DIM = 768
_EMBED_EXTRACT_CHUNK = 1024


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

        # Extract embedding weight matrix once at init for fast numpy indexing.
        # embed_tokens is a simple lookup table; extracting it avoids ~100-200
        # session.run() calls per image (one per generated token).
        self._embedding_weights = self._extract_embedding_weights()

        # Pre-compute decode loop constants to avoid per-token allocations.
        self._use_cache_false = np.array([False])
        self._use_cache_true = np.array([True])

        # Decoder KV output names (excluding logits at index 0).
        kv_out_names = [o.name for o in self._decoder.get_outputs()[1:]]
        self._kv_out_names = kv_out_names
        # Map output KV names -> input names (present.* -> past_key_values.*).
        self._kv_out_to_in = {n: n.replace("present", "past_key_values") for n in kv_out_names}
        # Partition output indices into encoder vs decoder KV slots.
        self._enc_kv_indices = [i for i, n in enumerate(kv_out_names) if ".encoder." in n]
        self._dec_kv_indices = [i for i, n in enumerate(kv_out_names) if ".encoder." not in n]

    def _extract_embedding_weights(self) -> np.ndarray:
        """Run embed_tokens in chunks to build a (vocab_size, embed_dim) weight matrix."""
        weights = np.empty((_VOCAB_SIZE, _EMBED_DIM), dtype=np.float32)
        for start in range(0, _VOCAB_SIZE, _EMBED_EXTRACT_CHUNK):
            end = min(start + _EMBED_EXTRACT_CHUNK, _VOCAB_SIZE)
            ids = np.arange(start, end, dtype=np.int64).reshape(1, -1)
            chunk_embeds = self._embed_tokens.run(None, {"input_ids": ids})[0]
            weights[start:end] = chunk_embeds[0]
        return weights

    async def extract_text(self, image_bytes: bytes) -> OcrResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._run_ocr(image))

    def _run_ocr(self, image: Image.Image) -> OcrResult:
        timing = settings.onnx_log_timing
        t0 = time.perf_counter() if timing else None
        task = "<OCR_WITH_REGION>"

        inputs = self._processor(text=task, images=image, return_tensors="np")
        t_processor = time.perf_counter() if timing else None

        # Stage 1: Vision encoding
        pixel_values = inputs["pixel_values"].astype(np.float32)
        image_features = self._vision_encoder.run(
            None, {"pixel_values": pixel_values}
        )[0]
        t_vision = time.perf_counter() if timing else None

        # Stage 2: Text embedding (numpy indexing) + encoder
        input_ids = inputs["input_ids"].astype(np.int64)
        prompt_embeds = self._embedding_weights[input_ids[0]][np.newaxis]  # (1, seq, dim)
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
        t_encoder = time.perf_counter() if timing else None

        # Stage 3: Greedy autoregressive decode
        generated_ids = self._greedy_decode(encoder_hidden, combined_mask)
        t_decode = time.perf_counter() if timing else None

        # Stage 4: Post-process
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self._processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )
        result = _build_ocr_result(parsed[task])

        if timing:
            t_end = time.perf_counter()
            num_tokens = len(generated_ids[0]) if generated_ids else 0
            logger.info(
                "ONNX timing — processor: %.1fms, vision_enc: %.1fms, "
                "text_enc: %.1fms, decode: %.1fms (%d tokens), "
                "postprocess: %.1fms, total: %.1fms",
                (t_processor - t0) * 1000,
                (t_vision - t_processor) * 1000,
                (t_encoder - t_vision) * 1000,
                (t_decode - t_encoder) * 1000,
                num_tokens,
                (t_end - t_decode) * 1000,
                (t_end - t0) * 1000,
            )

        return result

    def _greedy_decode(self, encoder_hidden, attention_mask, max_tokens=1024):
        # Seed decoder with EOS token (BART convention: decoder_start = EOS)
        seed_embeds = self._embedding_weights[[self._eos_token_id]][np.newaxis]  # (1,1,dim)

        # Prefill: use_cache_branch=False, pass empty KV cache tensors
        empty_kv = np.zeros((1, 12, 0, 64), dtype=np.float32)
        feed = {
            "inputs_embeds": seed_embeds,
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask,
            "use_cache_branch": self._use_cache_false,
        }
        for layer in range(_NUM_LAYERS):
            feed[f"past_key_values.{layer}.decoder.key"] = empty_kv
            feed[f"past_key_values.{layer}.decoder.value"] = empty_kv
            feed[f"past_key_values.{layer}.encoder.key"] = empty_kv
            feed[f"past_key_values.{layer}.encoder.value"] = empty_kv

        outs = self._decoder.run(None, feed)
        logits = outs[0]
        # Store KV cache as flat list parallel to self._kv_out_names
        kv_cache = list(outs[1:])

        # Encoder KV is computed once during prefill and reused for all
        # decode steps. The merged decoder's If node corrupts the encoder
        # KV pass-through on the use_cache_branch=True path, so we pin it.
        encoder_kv_snap = [kv_cache[i] for i in self._enc_kv_indices]

        tokens = [self._eos_token_id]
        # Reuse feed dict across iterations — mutate values in-place
        decode_feed = {
            "inputs_embeds": None,
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask,
            "use_cache_branch": self._use_cache_true,
        }

        for _ in range(max_tokens):
            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            tokens.append(next_token)
            if next_token == self._eos_token_id:
                break

            # Embed next token via numpy indexing (no session.run overhead)
            decode_feed["inputs_embeds"] = self._embedding_weights[[next_token]][np.newaxis]  # (1,1,dim)

            # Build KV cache inputs from flat list
            for i, out_name in enumerate(self._kv_out_names):
                in_name = self._kv_out_to_in[out_name]
                if i in self._enc_kv_indices:
                    decode_feed[in_name] = encoder_kv_snap[self._enc_kv_indices.index(i)]
                else:
                    decode_feed[in_name] = kv_cache[i]

            outs = self._decoder.run(None, decode_feed)
            logits = outs[0]
            kv_cache = list(outs[1:])

        return [tokens]
