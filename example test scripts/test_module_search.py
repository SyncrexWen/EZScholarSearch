# NOQA
import unittest
from ezscholarsearch.search import ScholarSearch, PubMeta
import logging
from scholarly import MaxTriesExceededException

# 配置日志以记录正常输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestScholarSearch(unittest.TestCase):
    def setUp(self):
        """初始化 ScholarSearch 实例，设置延迟以避免 API 限制。"""
        self.scholar = ScholarSearch(delay=10.0)  # 5 秒延迟减少速率限制

    def test_search_pubs(self):
        """测试 search_pubs 使用真实查询。"""
        try:
            results = self.scholar.search_pubs("yunfeng xiao", max_results=1)
            self.assertTrue(len(results) <= 1, "结果数量应不超过 1")
            for result in results:
                self.assertIsInstance(result, PubMeta, "结果应为 PubMeta 类型")
                self.assertIn('title', result.bib, "出版物应包含标题")
                logger.info(f"search_pubs 结果: {result.bib['title']}")
        except MaxTriesExceededException:
            self.skipTest("因 API 速率限制跳过测试")
        except Exception as e:
            self.fail(f"search_pubs 失败: {e}")


if __name__ == '__main__':
    unittest.main()
