from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models import OcrResult

FAKE_BYTES = b"fake image bytes"
TASK = "<OCR_WITH_REGION>"
MODULE = "app.engines.florence2_onnx_engine"
NUM_LAYERS = 6


def _make_mock_session(output_names=None):
    session = MagicMock()
    if output_names:
        outputs = []
        for name in output_names:
            out = MagicMock()
            out.name = name
            outputs.append(out)
        session.get_outputs.return_value = outputs
    return session


def _decoder_output_names():
    """KV cache output names for a 6-layer merged decoder."""
    names = ["logits"]
    for layer in range(NUM_LAYERS):
        names.extend([
            f"present.{layer}.decoder.key",
            f"present.{layer}.decoder.value",
            f"present.{layer}.encoder.key",
            f"present.{layer}.encoder.value",
        ])
    return names


def _make_sessions():
    decoder_out_names = _decoder_output_names()
    return {
        "vision_encoder": _make_mock_session(["image_features"]),
        "embed_tokens": _make_mock_session(["embeddings"]),
        "encoder": _make_mock_session(["encoder_hidden_states"]),
        "decoder": _make_mock_session(decoder_out_names),
    }


@pytest.fixture
def mock_onnx_deps():
    with patch(f"{MODULE}.ort") as mock_ort, \
         patch(f"{MODULE}.AutoProcessor") as mock_proc_cls:
        mock_ort.SessionOptions.return_value = MagicMock()

        sessions = _make_sessions()
        mock_ort.InferenceSession.side_effect = list(sessions.values())

        processor_instance = MagicMock()
        processor_instance.tokenizer.eos_token_id = 2
        mock_proc_cls.from_pretrained.return_value = processor_instance

        yield sessions, processor_instance


def _configure_embed_run(sessions):
    """Set embed_tokens.run side_effect to handle chunked weight extraction at init.

    _extract_embedding_weights calls embed_tokens.run in chunks of 1024 tokens,
    passing input_ids of shape (1, chunk_size). This helper makes the mock return
    the correctly shaped zeros so __init__ completes without error.
    """
    def _embed_side_effect(_, input_dict):
        seq_len = input_dict["input_ids"].shape[1]
        return [np.zeros((1, seq_len, 768), dtype=np.float32)]
    sessions["embed_tokens"].run.side_effect = _embed_side_effect


def _build_engine(sessions, processor_instance):
    """Build engine with fresh mocks for ort and AutoProcessor."""
    _configure_embed_run(sessions)
    with patch(f"{MODULE}.ort") as mock_ort, \
         patch(f"{MODULE}.AutoProcessor") as mock_proc_cls:
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.side_effect = list(sessions.values())
        mock_proc_cls.from_pretrained.return_value = processor_instance

        from app.engines.florence2_onnx_engine import Florence2OnnxEngine
        return Florence2OnnxEngine(model_path="/fake/path"), mock_ort


class TestFlorence2OnnxEngineInit:
    def test_creates_four_sessions(self, mock_onnx_deps):
        sessions, proc = mock_onnx_deps
        _, mock_ort = _build_engine(_make_sessions(), proc)
        assert mock_ort.InferenceSession.call_count == 4

    def test_session_file_paths_use_quantization_suffix(self, mock_onnx_deps):
        _, proc = mock_onnx_deps
        sessions = _make_sessions()
        _configure_embed_run(sessions)
        with patch(f"{MODULE}.ort") as mock_ort, \
             patch(f"{MODULE}.AutoProcessor") as mock_proc_cls:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.InferenceSession.side_effect = list(sessions.values())
            mock_proc_cls.from_pretrained.return_value = proc

            from app.engines.florence2_onnx_engine import Florence2OnnxEngine
            Florence2OnnxEngine(model_path="/fake/path", quantization="q4")

            paths = [str(c[0][0]) for c in mock_ort.InferenceSession.call_args_list]
            assert any("vision_encoder_q4.onnx" in p for p in paths)
            assert any("embed_tokens_q4.onnx" in p for p in paths)
            assert any("encoder_model_q4.onnx" in p for p in paths)
            assert any("decoder_model_merged_q4.onnx" in p for p in paths)

    def test_no_quantization_suffix(self, mock_onnx_deps):
        _, proc = mock_onnx_deps
        sessions = _make_sessions()
        _configure_embed_run(sessions)
        with patch(f"{MODULE}.ort") as mock_ort, \
             patch(f"{MODULE}.AutoProcessor") as mock_proc_cls:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.InferenceSession.side_effect = list(sessions.values())
            mock_proc_cls.from_pretrained.return_value = proc

            from app.engines.florence2_onnx_engine import Florence2OnnxEngine
            Florence2OnnxEngine(model_path="/fake/path", quantization="")

            paths = [str(c[0][0]) for c in mock_ort.InferenceSession.call_args_list]
            assert any("vision_encoder.onnx" in p for p in paths)
            assert any("decoder_model_merged.onnx" in p for p in paths)

    def test_processor_loaded_with_trust_remote_code(self, mock_onnx_deps):
        _, proc = mock_onnx_deps
        sessions = _make_sessions()
        _configure_embed_run(sessions)
        with patch(f"{MODULE}.ort") as mock_ort, \
             patch(f"{MODULE}.AutoProcessor") as mock_proc_cls:
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.InferenceSession.side_effect = list(sessions.values())
            mock_proc_cls.from_pretrained.return_value = proc

            from app.engines.florence2_onnx_engine import Florence2OnnxEngine
            Florence2OnnxEngine(model_path="/fake/path")

            mock_proc_cls.from_pretrained.assert_called_once_with(
                "microsoft/Florence-2-base-ft", trust_remote_code=True, local_files_only=True
            )


class TestFlorence2OnnxEngineExtractText:
    def _setup_mocks(self, sessions, processor_instance):
        # Vision encoder returns image features
        sessions["vision_encoder"].run.return_value = [
            np.zeros((1, 577, 768), dtype=np.float32)
        ]
        # Encoder returns hidden states
        sessions["encoder"].run.return_value = [
            np.zeros((1, 578, 768), dtype=np.float32)
        ]
        # Decoder: logits with EOS as argmax so decode stops immediately
        decoder_logits = np.zeros((1, 1, 51289), dtype=np.float32)
        decoder_logits[0, 0, 2] = 10.0
        kv_tensors = [np.zeros((1, 12, 1, 64), dtype=np.float32)] * (NUM_LAYERS * 4)
        sessions["decoder"].run.return_value = [decoder_logits] + kv_tensors

        # Processor: return correct shapes for pixel_values and input_ids.
        # input_ids must be 2D (1, seq_len) so that input_ids[0] is a 1D
        # index array suitable for numpy fancy indexing into _embedding_weights.
        def _inputs_getitem(self_mock, key):
            if key == "pixel_values":
                return np.zeros((1, 3, 768, 768), dtype=np.float32)
            return np.zeros((1, 9), dtype=np.int64)  # input_ids

        inputs_mock = MagicMock()
        inputs_mock.__getitem__ = _inputs_getitem
        processor_instance.return_value = inputs_mock
        processor_instance.batch_decode.return_value = ["<fake text>"]
        processor_instance.post_process_generation.return_value = {
            TASK: {
                "labels": ["The Great Gatsby", "F Scott Fitzgerald"],
                "quad_boxes": [
                    [0, 0, 50, 0, 50, 10, 0, 10],
                    [0, 15, 60, 15, 60, 25, 0, 25],
                ],
            }
        }

    @pytest.mark.asyncio
    async def test_returns_ocr_result(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        self._setup_mocks(sessions, processor_instance)

        engine, _ = _build_engine(_make_sessions_with(sessions), processor_instance)

        with patch(f"{MODULE}.Image") as mock_image:
            img_mock = MagicMock()
            img_mock.width = 100
            img_mock.height = 200
            mock_image.open.return_value.convert.return_value = img_mock

            result = await engine.extract_text(FAKE_BYTES)

        assert isinstance(result, OcrResult)
        assert "The Great Gatsby" in result.text
        assert "F Scott Fitzgerald" in result.text

    @pytest.mark.asyncio
    async def test_calls_vision_encoder(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        self._setup_mocks(sessions, processor_instance)

        engine, _ = _build_engine(_make_sessions_with(sessions), processor_instance)

        with patch(f"{MODULE}.Image") as mock_image:
            img_mock = MagicMock()
            img_mock.width = 100
            img_mock.height = 200
            mock_image.open.return_value.convert.return_value = img_mock

            await engine.extract_text(FAKE_BYTES)

        sessions["vision_encoder"].run.assert_called_once()


def _make_sessions_with(real_sessions):
    """Return session dict that reuses pre-configured mock sessions."""
    return {
        "vision_encoder": real_sessions["vision_encoder"],
        "embed_tokens": real_sessions["embed_tokens"],
        "encoder": real_sessions["encoder"],
        "decoder": real_sessions["decoder"],
    }


class TestGreedyDecode:
    def _make_engine(self, sessions, processor_instance):
        engine = _build_engine(
            _make_sessions_with(sessions), processor_instance
        )[0]
        # _embedding_weights is already populated by __init__ (all zeros from mock).
        # Explicitly set here to ensure a known state independent of mock details.
        engine._embedding_weights = np.zeros((51289, 768), dtype=np.float32)
        return engine

    def test_eos_terminates_loop(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        engine = self._make_engine(sessions, processor_instance)

        kv_tensors = [np.zeros((1, 12, 1, 64), dtype=np.float32)] * (NUM_LAYERS * 4)

        # Prefill returns token 100
        logits_first = np.zeros((1, 1, 51289), dtype=np.float32)
        logits_first[0, 0, 100] = 10.0

        # Second step returns EOS
        logits_eos = np.zeros((1, 1, 51289), dtype=np.float32)
        logits_eos[0, 0, 2] = 10.0

        sessions["decoder"].run.side_effect = [
            [logits_first] + kv_tensors,
            [logits_eos] + kv_tensors,
        ]

        encoder_hidden = np.zeros((1, 578, 768), dtype=np.float32)
        attention_mask = np.ones((1, 578), dtype=np.int64)
        result = engine._greedy_decode(encoder_hidden, attention_mask)

        # Should have: [EOS_start, 100, EOS_end]
        assert result[0] == [2, 100, 2]

    def test_max_tokens_limit(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        engine = self._make_engine(sessions, processor_instance)

        # Always return token 100 (never EOS)
        logits = np.zeros((1, 1, 51289), dtype=np.float32)
        logits[0, 0, 100] = 10.0
        kv_tensors = [np.zeros((1, 12, 1, 64), dtype=np.float32)] * (NUM_LAYERS * 4)
        sessions["decoder"].run.return_value = [logits] + kv_tensors

        encoder_hidden = np.zeros((1, 578, 768), dtype=np.float32)
        attention_mask = np.ones((1, 578), dtype=np.int64)
        result = engine._greedy_decode(encoder_hidden, attention_mask, max_tokens=5)

        # Should have: [EOS_start, 100, 100, 100, 100, 100] (1 seed + 5 generated)
        assert len(result[0]) == 6

    def test_kv_cache_passed_correctly(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        engine = self._make_engine(sessions, processor_instance)

        kv_tensors = [np.ones((1, 12, 1, 64), dtype=np.float32) * 42] * (NUM_LAYERS * 4)

        # Prefill returns non-EOS token
        logits_first = np.zeros((1, 1, 51289), dtype=np.float32)
        logits_first[0, 0, 100] = 10.0

        # Second step returns EOS
        logits_eos = np.zeros((1, 1, 51289), dtype=np.float32)
        logits_eos[0, 0, 2] = 10.0

        sessions["decoder"].run.side_effect = [
            [logits_first] + kv_tensors,
            [logits_eos] + kv_tensors,
        ]

        encoder_hidden = np.zeros((1, 578, 768), dtype=np.float32)
        attention_mask = np.ones((1, 578), dtype=np.int64)
        engine._greedy_decode(encoder_hidden, attention_mask)

        # Verify second call (decode step) has KV cache and use_cache_branch=True
        assert sessions["decoder"].run.call_count == 2
        call_args = sessions["decoder"].run.call_args_list[1]
        feed_dict = call_args[0][1]
        assert "past_key_values.0.decoder.key" in feed_dict
        assert feed_dict["past_key_values.0.decoder.key"][0, 0, 0, 0] == 42.0
        assert feed_dict["use_cache_branch"][0] is np.bool_(True)

    def test_embed_tokens_not_called_during_decode(self, mock_onnx_deps):
        sessions, processor_instance = mock_onnx_deps
        engine = self._make_engine(sessions, processor_instance)

        # Capture call count after __init__ (chunked weight extraction already done)
        init_call_count = sessions["embed_tokens"].run.call_count

        kv_tensors = [np.zeros((1, 12, 1, 64), dtype=np.float32)] * (NUM_LAYERS * 4)
        logits_eos = np.zeros((1, 1, 51289), dtype=np.float32)
        logits_eos[0, 0, 2] = 10.0
        sessions["decoder"].run.return_value = [logits_eos] + kv_tensors

        encoder_hidden = np.zeros((1, 578, 768), dtype=np.float32)
        attention_mask = np.ones((1, 578), dtype=np.int64)
        engine._greedy_decode(encoder_hidden, attention_mask)

        # embed_tokens.run must NOT be called during decode (numpy indexing is used instead)
        assert sessions["embed_tokens"].run.call_count == init_call_count


class TestFlorence2OnnxEngineInitEmbedding:
    def test_embedding_weights_shape(self, mock_onnx_deps):
        sessions, proc = mock_onnx_deps
        engine, _ = _build_engine(_make_sessions_with(sessions), proc)
        assert engine._embedding_weights.shape == (51289, 768)

    def test_embed_tokens_called_at_init(self, mock_onnx_deps):
        sessions, proc = mock_onnx_deps
        _build_engine(_make_sessions_with(sessions), proc)
        # Chunked extraction: ceil(51289 / 1024) = 51 chunks
        assert sessions["embed_tokens"].run.call_count == 51
