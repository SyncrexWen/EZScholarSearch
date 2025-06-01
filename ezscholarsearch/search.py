# flake8: noqa: E501
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from functools import wraps
from time import sleep
from scholarly import scholarly, MaxTriesExceededException
from .datastructs import PubMeta
from .AI import DataPacket

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'ScholarSearch',
]


def _safe_search_input(separator: str = "\n---\n"):
    """装饰器，安全处理 DataPacket 或字符串输入。

    Args:
        separator (str): 用于拼接 metadata 的分隔符，默认 "\n---\n"。

    Returns:
        Callable: 包装后的函数。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, data: DataPacket | str, *args, **kwargs):
            if isinstance(data, DataPacket):
                if data.content:
                    input_data = data.content
                elif data.metadata:
                    input_data = separator.join([f"##{key}\n\n{value}" for key, value in data.metadata.items()])
                else:
                    raise ValueError("DataPacket must have content or metadata")
            else:
                if not data or not isinstance(data, str):
                    raise ValueError("Input must be a non-empty string or DataPacket")
                input_data = data.strip()
            return func(self, input_data, *args, **kwargs)
        return wrapper
    return decorator


class ScholarSearch:
    """学术搜索工具，基于 Google Scholar API (scholarly) 提供出版物搜索功能。

    Args:
        delay (float): 每次 API 调用后的等待时间（秒），默认从环境变量 SCHOLAR_DELAY 获取，或 4.0 秒。
    """
    def __init__(self, delay: float = float(os.getenv("SCHOLAR_DELAY", "4.0"))):
        self.delay = delay
        self._last_error = None

    def _delay(self, increase_factor: float = 1.5) -> None:
        """根据上次错误动态调整延迟。

        Args:
            increase_factor (float): 延迟增加的倍数，默认 1.5。
        """
        sleep(self.delay)
        if self._last_error and isinstance(self._last_error, MaxTriesExceededException):
            self.delay *= increase_factor
            logger.info(f"延迟增加到 {self.delay} 秒，因 API 限制")

    @_safe_search_input()
    def search_pubs(self, pubs: str, max_results: int = 3) -> List[PubMeta]:
        """搜索学术出版物。

        Args:
            pubs (str): 出版物查询关键词。
            max_results (int): 最大返回结果数，默认 3。

        Returns:
            List[PubMeta]: 出版物元数据列表。
        """
        if not pubs.strip():
            raise ValueError("出版物查询不能为空")
        search_query = scholarly.search_pubs(pubs)
        pubs_list = []
        for _ in range(max_results):
            try:
                pubs_list.append(next(search_query))
            except StopIteration:
                break
            except MaxTriesExceededException as e:
                self._last_error = e
                logger.warning(f"出版物查询 API 限制: {pubs}")
                break
            except Exception as e:
                self._last_error = e
                logger.error(f"search_pubs 错误: {pubs}: {e}")
                continue
        with ThreadPoolExecutor(max_workers=3) as executor:
            filled_pubs = list(executor.map(scholarly.fill, pubs_list))
        ret = [PubMeta(**pub) for pub in filled_pubs]
        self._delay()
        return ret

    @_safe_search_input()
    def advanced_search(self, keyword: Optional[str] = None, year_min: Optional[int] = None,
                        year_max: Optional[int] = None, max_results: int = 5) -> List[PubMeta]:
        """高级出版物搜索，基于关键词和年份过滤，包含无年份的出版物。

        Args:
            keyword (Optional[str]): 搜索关键词。
            year_min (Optional[int]): 最早出版年份。
            year_max (Optional[int]): 最晚出版年份。
            max_results (int): 最大返回结果数，默认 5。

        Returns:
            List[PubMeta]: 符合条件的出版物元数据列表。
        """
        results = []
        query = keyword.strip() if keyword else ''
        try:
            pubs_list = self.search_pubs(query, max_results=max_results * 2)  # 获取更多结果以便过滤
            for pub in pubs_list:
                try:
                    bib = pub.bib
                    pub_year_str = bib.get("pub_year", "")
                    if pub_year_str:
                        try:
                            pub_year = int(pub_year_str)
                            if year_min and pub_year < year_min:
                                continue
                            if year_max and pub_year > year_max:
                                continue
                        except ValueError:
                            logger.warning(f"无效年份格式: {pub_year_str}, 出版物: {bib.get('title', '未知')}")
                    results.append(pub)
                    if len(results) >= max_results:
                        break
                except Exception as e:
                    self._last_error = e
                    logger.error(f"处理出版物错误: {e}")
                    continue
            self._delay()
        except Exception as e:
            self._last_error = e
            logger.error(f"高级搜索错误: {e}")
        return results[:max_results]
