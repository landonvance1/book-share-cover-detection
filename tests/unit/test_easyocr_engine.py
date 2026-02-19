from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.engines.easyocr_engine import EasyOcrEngine
from app.models import OcrResult

FAKE_BYTES = b"fake image bytes"

EASYOCR_RAW_RESULT = [
    ([[0, 0], [200, 0], [200, 50], [0, 50]], "The Great Gatsby", 0.95),
    ([[0, 60], [200, 60], [200, 100], [0, 100]], "F Scott Fitzgerald", 0.88),
]


@pytest.fixture
def mock_reader():
    with patch("app.engines.easyocr_engine.easyocr.Reader") as MockReader:
        instance = MagicMock()
        MockReader.return_value = instance
        yield MockReader, instance


@pytest.fixture
def mock_image():
    with patch("app.engines.easyocr_engine.Image.open") as mock_open, \
         patch("app.engines.easyocr_engine.np.array") as mock_array:
        mock_open.return_value = MagicMock()
        mock_array.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        yield


class TestEasyOcrEngineInit:
    def test_default_params(self, mock_reader):
        MockReader, _ = mock_reader
        EasyOcrEngine()
        MockReader.assert_called_once_with(["en"], gpu=False)

    def test_custom_languages(self, mock_reader):
        MockReader, _ = mock_reader
        EasyOcrEngine(languages=["en", "fr"])
        MockReader.assert_called_once_with(["en", "fr"], gpu=False)

    def test_gpu_flag(self, mock_reader):
        MockReader, _ = mock_reader
        EasyOcrEngine(gpu=True)
        MockReader.assert_called_once_with(["en"], gpu=True)


class TestEasyOcrEngineExtractText:
    @pytest.mark.asyncio
    async def test_returns_ocr_result(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.return_value = EASYOCR_RAW_RESULT
        result = await EasyOcrEngine().extract_text(FAKE_BYTES)
        assert isinstance(result, OcrResult)

    @pytest.mark.asyncio
    async def test_joined_text(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.return_value = EASYOCR_RAW_RESULT
        result = await EasyOcrEngine().extract_text(FAKE_BYTES)
        assert "The Great Gatsby" in result.text
        assert "F Scott Fitzgerald" in result.text

    @pytest.mark.asyncio
    async def test_region_count(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.return_value = EASYOCR_RAW_RESULT
        result = await EasyOcrEngine().extract_text(FAKE_BYTES)
        assert len(result.regions) == 2

    @pytest.mark.asyncio
    async def test_region_fields(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.return_value = EASYOCR_RAW_RESULT
        result = await EasyOcrEngine().extract_text(FAKE_BYTES)
        first = result.regions[0]
        assert first.text == "The Great Gatsby"
        assert first.confidence == pytest.approx(0.95)
        assert first.coordinates == [[0, 0], [200, 0], [200, 50], [0, 50]]

    @pytest.mark.asyncio
    async def test_empty_result(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.return_value = []
        result = await EasyOcrEngine().extract_text(FAKE_BYTES)
        assert result.text == ""
        assert result.regions == []

    @pytest.mark.asyncio
    async def test_pil_error_propagates(self, mock_reader):
        with patch("app.engines.easyocr_engine.Image.open", side_effect=Exception("bad image")):
            with pytest.raises(Exception, match="bad image"):
                await EasyOcrEngine().extract_text(b"not valid")

    @pytest.mark.asyncio
    async def test_reader_error_propagates(self, mock_reader, mock_image):
        _, reader_instance = mock_reader
        reader_instance.readtext.side_effect = RuntimeError("model crashed")
        with pytest.raises(RuntimeError, match="model crashed"):
            await EasyOcrEngine().extract_text(FAKE_BYTES)
