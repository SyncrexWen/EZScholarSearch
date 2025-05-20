from .utils import PubDownloader
from .datastructs import PubMeta
from .AI import DataPacket

from scholarly import scholarly
from typing import Dict, List, Optional
from time import sleep
from warnings import warn
from functools import wraps

__all__ = [
    'ScholarSearch',
]


def _safe_search_input(func):
    @wraps(func)
    def wrapper(self, data: DataPacket | str, *args, **kwargs):
        if isinstance(data, DataPacket):
            input_data = data.content or "\n---\n".join([
                f"##{key}\n\n{value}"
                for key, value in data.metadata.items()
            ])
        else:
            input_data = data
        return func(self, input_data, *args, **kwargs)
    return wrapper


class ScholarSearch:
    def __init__(self, delay: float = 2.0):
        self.delay = delay

    def _delay(self) -> None:
        sleep(self.delay)

    @staticmethod
    def config_api_email(email: str):
        PubDownloader.config_api_email(email)

    @_safe_search_input
    def search_author(self, author_name: str, max_results: int = 1
                      ) -> List[Dict]:
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
        return ret

    @_safe_search_input
    def search_authors(self, *author_names: str) -> List[Dict]:
        ret = []
        for author_name in author_names:
            author = next(scholarly.search_author(author_name))
            author_filled = scholarly.fill(author)
            ret.append(author_filled)
            self._delay()
        return ret

    @_safe_search_input
    def search_pubs(self, pubs: str,
                    max_results: int = 3
                    ) -> List[PubMeta]:
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
        return ret

    @_safe_search_input
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

    def download_pubs(self, *filled_pubs: PubMeta,
                      saving_dir: Optional[str] = None
                      ) -> tuple[List[str], int]:
        paths = []
        cnt = fail_cnt = 0
        for pub in filled_pubs:
            try:
                cnt += 1
                path = PubDownloader.download(pub, saving_dir=saving_dir)
                paths.append(path)
            except Exception as e:
                warn(
                    f"\nException when download the pub {cnt}:"
                    f"\n\tTitle: {pub.bib.title}"
                    f"\n\tURL: {pub.eprint_url}"
                    f"\n\tException: {e}"
                    )
                fail_cnt += 1
        return paths, cnt - fail_cnt
