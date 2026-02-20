from unittest.mock import MagicMock, patch

import pytest
import torch

from app.engines.florence2_engine import Florence2OcrEngine, _build_ocr_result
from app.models import OcrResult

FAKE_BYTES = b"fake image bytes"
TASK = "<OCR_WITH_REGION>"


def _make_mock_parsed(labels=None, quad_boxes=None):
    if labels is None:
        labels = ["The Great Gatsby", "F Scott Fitzgerald"]
    if quad_boxes is None:
        quad_boxes = [
            [0, 0, 50, 0, 50, 10, 0, 10],
            [0, 15, 60, 15, 60, 25, 0, 25],
        ]
    return {TASK: {"labels": labels, "quad_boxes": quad_boxes}}


@pytest.fixture
def mock_transformers():
    with patch("app.engines.florence2_engine.AutoModelForCausalLM") as mock_model_cls, \
         patch("app.engines.florence2_engine.AutoProcessor") as mock_processor_cls:
        model_instance = MagicMock()
        model_instance.to.return_value = model_instance
        mock_model_cls.from_pretrained.return_value = model_instance

        processor_instance = MagicMock()
        mock_processor_cls.from_pretrained.return_value = processor_instance

        yield mock_model_cls, mock_processor_cls, model_instance, processor_instance


class TestFlorence2OcrEngineInit:
    def test_default_model_name(self, mock_transformers):
        mock_model_cls, mock_processor_cls, _, _ = mock_transformers
        Florence2OcrEngine()
        mock_model_cls.from_pretrained.assert_called_once_with(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        mock_processor_cls.from_pretrained.assert_called_once_with(
            "microsoft/Florence-2-base", trust_remote_code=True
        )

    def test_custom_model_name(self, mock_transformers):
        mock_model_cls, mock_processor_cls, _, _ = mock_transformers
        Florence2OcrEngine(model_name="microsoft/Florence-2-large")
        mock_model_cls.from_pretrained.assert_called_once_with(
            "microsoft/Florence-2-large",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    def test_cpu_device_by_default(self, mock_transformers):
        _, _, model_instance, _ = mock_transformers
        engine = Florence2OcrEngine()
        model_instance.to.assert_called_once_with("cpu")
        assert engine._device == "cpu"

    def test_gpu_flag_uses_cuda(self, mock_transformers):
        _, _, model_instance, _ = mock_transformers
        engine = Florence2OcrEngine(gpu=True)
        model_instance.to.assert_called_once_with("cuda")
        assert engine._device == "cuda"


class TestFlorence2OcrEngineExtractText:
    def _setup_processor(self, processor_instance, parsed):
        inputs_mock = MagicMock()
        inputs_mock.__getitem__ = lambda self, key: MagicMock()
        processor_instance.return_value = inputs_mock
        processor_instance.batch_decode.return_value = ["<fake generated text>"]
        processor_instance.post_process_generation.return_value = parsed

    @pytest.mark.asyncio
    async def test_returns_ocr_result(self, mock_transformers):
        _, _, model_instance, processor_instance = mock_transformers
        parsed = _make_mock_parsed()
        self._setup_processor(processor_instance, parsed)
        model_instance.generate.return_value = MagicMock()

        with patch("app.engines.florence2_engine.Image") as mock_image:
            mock_image.open.return_value.__enter__ = MagicMock()
            img_mock = MagicMock()
            img_mock.width = 100
            img_mock.height = 200
            mock_image.open.return_value.convert.return_value = img_mock

            engine = Florence2OcrEngine()
            result = await engine.extract_text(FAKE_BYTES)

        assert isinstance(result, OcrResult)

    @pytest.mark.asyncio
    async def test_joined_text(self, mock_transformers):
        _, _, model_instance, processor_instance = mock_transformers
        parsed = _make_mock_parsed()
        self._setup_processor(processor_instance, parsed)
        model_instance.generate.return_value = MagicMock()

        with patch("app.engines.florence2_engine.Image") as mock_image:
            img_mock = MagicMock()
            img_mock.width = 100
            img_mock.height = 200
            mock_image.open.return_value.convert.return_value = img_mock

            engine = Florence2OcrEngine()
            result = await engine.extract_text(FAKE_BYTES)

        assert "The Great Gatsby" in result.text
        assert "F Scott Fitzgerald" in result.text


class TestBuildOcrResult:
    def test_quad_to_coordinate_pairs(self):
        ocr_data = {
            "labels": ["Hello"],
            "quad_boxes": [[10, 20, 30, 40, 50, 60, 70, 80]],
        }
        result = _build_ocr_result(ocr_data)
        assert result.regions[0].coordinates == [
            [10, 20],
            [30, 40],
            [50, 60],
            [70, 80],
        ]

    def test_confidence_is_one(self):
        ocr_data = {
            "labels": ["text"],
            "quad_boxes": [[0, 0, 10, 0, 10, 10, 0, 10]],
        }
        result = _build_ocr_result(ocr_data)
        assert result.regions[0].confidence == pytest.approx(1.0)

    def test_text_joined_with_spaces(self):
        ocr_data = {
            "labels": ["foo", "bar", "baz"],
            "quad_boxes": [
                [0, 0, 10, 0, 10, 10, 0, 10],
                [0, 0, 10, 0, 10, 10, 0, 10],
                [0, 0, 10, 0, 10, 10, 0, 10],
            ],
        }
        result = _build_ocr_result(ocr_data)
        assert result.text == "foo bar baz"

    def test_empty_quad_boxes_returns_empty(self):
        result = _build_ocr_result({"labels": [], "quad_boxes": []})
        assert result.text == ""
        assert result.regions == []

    def test_missing_keys_returns_empty(self):
        result = _build_ocr_result({})
        assert result.text == ""
        assert result.regions == []

    def test_region_count_matches_labels(self):
        ocr_data = {
            "labels": ["a", "b", "c"],
            "quad_boxes": [
                [0, 0, 1, 0, 1, 1, 0, 1],
                [2, 0, 3, 0, 3, 1, 2, 1],
                [4, 0, 5, 0, 5, 1, 4, 1],
            ],
        }
        result = _build_ocr_result(ocr_data)
        assert len(result.regions) == 3
