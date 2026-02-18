from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class OcrBoundingBox(CamelModel):
    text: str
    confidence: float
    coordinates: list[list[float]]


class OcrResult(CamelModel):
    text: str
    regions: list[OcrBoundingBox]


class NlpAnalysis(CamelModel):
    detected_title: str | None = None
    title_confidence: float = 0.0
    detected_author: str | None = None
    author_confidence: float = 0.0


class BookMatch(CamelModel):
    title: str
    author: str
    isbn: str | None = None
    thumbnail_url: str | None = None


class AnalysisStatus(CamelModel):
    is_success: bool
    error_message: str | None = None
    extracted_text: str | None = None


class CoverAnalysisResponse(CamelModel):
    analysis: AnalysisStatus
    matched_books: list[BookMatch] = []
    exact_match: BookMatch | None = None
    nlp_analysis: NlpAnalysis | None = None


class HealthResponse(CamelModel):
    status: str
    version: str
