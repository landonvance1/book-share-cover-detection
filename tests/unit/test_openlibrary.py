import pytest

from app.models import BookMatch
from app.services.openlibrary import OpenLibraryClient


@pytest.fixture
def client() -> OpenLibraryClient:
    return OpenLibraryClient()


class TestBuildWordSet:
    def test_basic_text(self, client):
        result = client.build_word_set("The Great Gatsby")
        assert result == {"the", "great", "gatsby"}

    def test_strips_punctuation(self, client):
        result = client.build_word_set("Hello, World! It's a test.")
        assert "hello" in result
        assert "world" in result
        assert "test" in result
        assert "its" in result

    def test_removes_short_words(self, client):
        result = client.build_word_set("I am a hero of an era")
        assert "hero" in result
        assert "era" in result
        assert "am" not in result
        assert "a" not in result
        assert "an" not in result
        assert "of" not in result

    def test_empty_string(self, client):
        result = client.build_word_set("")
        assert result == set()

    def test_all_short_words(self, client):
        result = client.build_word_set("I am a he")
        assert result == set()


class TestScoreResults:
    def test_perfect_match(self, client):
        books = [BookMatch(title="The Great Gatsby", author="F Scott Fitzgerald")]
        ocr_words = {"the", "great", "gatsby", "scott", "fitzgerald"}
        scored = client.score_results(books, ocr_words)
        assert len(scored) == 1
        assert scored[0][1] == 1.0

    def test_partial_match_above_threshold(self, client):
        books = [BookMatch(title="The Great Gatsby", author="F Scott Fitzgerald")]
        ocr_words = {"the", "great", "gatsby"}
        scored = client.score_results(books, ocr_words)
        assert len(scored) == 1
        assert scored[0][1] >= 0.5

    def test_below_threshold_excluded(self, client):
        books = [BookMatch(title="The Great Gatsby", author="F Scott Fitzgerald")]
        ocr_words = {"random", "words", "here"}
        scored = client.score_results(books, ocr_words)
        assert len(scored) == 0

    def test_multiple_books_sorted_by_score(self, client):
        books = [
            BookMatch(title="The Great Gatsby", author="F Scott Fitzgerald"),
            BookMatch(title="Great Expectations", author="Charles Dickens"),
        ]
        ocr_words = {"the", "great", "gatsby", "scott", "fitzgerald"}
        scored = client.score_results(books, ocr_words)
        assert len(scored) >= 1
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self, client):
        scored = client.score_results([], {"test", "words"})
        assert scored == []


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_raises_not_implemented(self, client):
        with pytest.raises(NotImplementedError):
            await client.search("test query")
