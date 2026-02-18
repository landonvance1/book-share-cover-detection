import httpx

from app.interfaces.book_search import BookSearchClient
from app.models import BookMatch


class OpenLibraryClient(BookSearchClient):
    BASE_URL = "https://openlibrary.org"
    COVER_URL = "https://covers.openlibrary.org/b/id"
    USER_AGENT = "Community Bookshare App (landonpvance@gmail.com)"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def search(self, query: str, limit: int = 11) -> list[BookMatch]:
        raise NotImplementedError("OpenLibrary search not yet implemented")
