import re
from abc import ABC, abstractmethod

from app.models import BookMatch


class BookSearchClient(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int = 11) -> list[BookMatch]:
        ...

    def score_results(
        self, results: list[BookMatch], ocr_words: set[str]
    ) -> list[tuple[BookMatch, float]]:
        scored: list[tuple[BookMatch, float]] = []
        for book in results:
            book_text = f"{book.title} {book.author}"
            book_words = self.build_word_set(book_text)
            if not book_words:
                continue
            match_count = len(book_words & ocr_words)
            score = match_count / len(book_words)
            if score >= 0.5:
                scored.append((book, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def build_word_set(text: str) -> set[str]:
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return {w for w in words if len(w) > 2}
