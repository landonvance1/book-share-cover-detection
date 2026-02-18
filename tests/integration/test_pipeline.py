import pytest


@pytest.mark.skip(reason="Integration tests require real images and engine implementations")
class TestFullPipeline:
    async def test_analyze_real_book_cover(self):
        pass

    async def test_analyze_with_real_ocr(self):
        pass
