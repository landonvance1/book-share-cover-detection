import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import AnalysisStatus, CoverAnalysisResponse, NlpAnalysis


@pytest.fixture
def mock_analyzer():
    with patch("app.main.analyzer") as mock:
        mock.analyze = AsyncMock(
            return_value=CoverAnalysisResponse(
                analysis=AnalysisStatus(
                    is_success=True,
                    extracted_text="The Great Gatsby",
                ),
                matched_books=[],
                nlp_analysis=NlpAnalysis(
                    detected_title="The Great Gatsby",
                    title_confidence=0.92,
                ),
            )
        )
        yield mock


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"


class TestAnalyzeEndpoint:
    @pytest.mark.asyncio
    async def test_analyze_success(self, client, mock_analyzer):
        fake_image = io.BytesIO(b"fake image data")
        response = await client.post(
            "/analyze",
            files={"file": ("cover.jpg", fake_image, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["analysis"]["isSuccess"] is True
        mock_analyzer.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_content_type(self, client, mock_analyzer):
        fake_file = io.BytesIO(b"not an image")
        response = await client.post(
            "/analyze",
            files={"file": ("doc.pdf", fake_file, "application/pdf")},
        )
        assert response.status_code == 400
        assert "Invalid content type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_file_too_large(self, client, mock_analyzer):
        large_file = io.BytesIO(b"x" * (4 * 1024 * 1024 + 1))
        response = await client.post(
            "/analyze",
            files={"file": ("big.jpg", large_file, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_file(self, client, mock_analyzer):
        response = await client.post("/analyze")
        assert response.status_code == 422
