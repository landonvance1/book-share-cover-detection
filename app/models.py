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
    potential_authors: list[str] = []
    potential_titles: list[str] = []


class BookMatch(CamelModel):
    title: str
    author: str
    isbn: str | None = None
    thumbnail_url: str | None = None


class AnalysisStatus(CamelModel):
    is_success: bool
    error_message: str | None = None

class CoverAnalysisResponse(CamelModel):
    analysisStatus: AnalysisStatus
    ocr_result: OcrResult | None = None
    nlp_analysis: NlpAnalysis | None = None

class HealthResponse(CamelModel):
    status: str
    version: str
