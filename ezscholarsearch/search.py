from .utils import PubDownloader
from .datastructs import PubMeta

from scholarly import scholarly
from typing import Dict, List, Optional
from time import sleep

__all__ = [
    'ScholarSearch',
]


class ScholarSearch:
    def __init__(self, delay: float = 2.0):
        self.delay = delay

    def _delay(self) -> None:
        sleep(self.delay)

    def search_author(self, author_name: str, max_results: int = 1
                      ) -> dict | List[Dict]:
        query = scholarly.search_author(author_name)
        ret = []
        for _ in range(max_results):
            try:
                author = next(query)
                filled_author = scholarly.fill(author)
                ret.append(filled_author)
                self._delay()
            except StopIteration:
                break
        return ret[0] if len(ret) == 1 else ret

    def search_authors(self, *author_names: str) -> dict | List[Dict]:
        ret = []
        for author_name in author_names:
            author = next(scholarly.search_author(author_name))
            author_filled = scholarly.fill(author)
            ret.append(author_filled)
            self._delay()
        return ret[0] if len(ret) == 1 else ret

    def search_pubs(self, pubs: str,
                    max_results: int = 3
                    ) -> PubMeta | List[PubMeta]:
        search_query = scholarly.search_pubs(pubs)
        ret = []
        for _ in range(max_results):
            try:
                pub = next(search_query)
                filled_pub = scholarly.fill(pub)
                meta = PubMeta(**filled_pub)
                ret.append(meta)
                self._delay()
            except StopIteration:
                break
        return ret[0] if len(ret) == 1 else ret

    def advanced_search(self, keyword: Optional[str] = None,
                        author: Optional[str] = None,
                        year_min: Optional[int] = None,
                        year_max: Optional[int] = None,
                        max_results: int = 5) -> List[PubMeta]:
        results = []
        try:
            query = keyword if keyword else ''
            pub_iter = scholarly.search_pubs(query)
            for pub in pub_iter:
                pub_filled = scholarly.fill(pub)
                bib = pub_filled.get("bib", {})
                pub_year = int(bib.get("pub_year", "0"))
                if author and author.lower() not in \
                        bib.get("author", "").lower():
                    continue
                if year_min and pub_year < year_min:
                    continue
                if year_max and pub_year > year_max:
                    continue
                meta = PubMeta(**pub_filled)
                results.append(meta)
                if len(results) >= max_results:
                    break
                self._delay()
        except Exception as e:
            print(f"Error during advanced search: {e}")
        return results

    def download_pub(self, filled_pub: dict,
                     saving_dir: Optional[str] = None
                     ) -> str | None:
        return PubDownloader.download(filled_pub, saving_dir)
