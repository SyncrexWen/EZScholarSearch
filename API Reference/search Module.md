# ScholarSearch 库使用文档

## 概述

`ScholarSearch` 是一个基于 Python 的工具库，利用 Google Scholar API（通过 `scholarly` 包）提供学术出版物搜索功能。它支持通过关键词搜索出版物，并提供基于年份过滤的高级搜索功能。库通过动态调整延迟来处理 API 速率限制，并支持以 `DataPacket` 结构或普通字符串形式处理输入数据。

## 安装

要使用 `ScholarSearch` 库，请确保已安装所需依赖项：

```bash
pip install scholarly
```

此外，库依赖于项目中的自定义模块（`PubDownloader`、`PubMeta`、`DataPacket`）。库还使用环境变量进行配置，例如 `SCHOLAR_DELAY` 用于设置 API 调用之间的默认延迟。

## 配置

通过设置 `SCHOLAR_DELAY` 环境变量来控制 API 请求之间的延迟时间（单位：秒）。如果未设置，默认延迟为 4.0 秒。

```bash
export SCHOLAR_DELAY=4.0
```

## 主要类：`ScholarSearch`

`ScholarSearch` 类是库的核心组件，提供学术出版物搜索的方法。

### 构造函数

```python
ScholarSearch(delay: float = float(os.getenv("SCHOLAR_DELAY", "4.0")))
```

- 参数：
  - `delay`：API 调用之间的等待时间（秒）。默认从环境变量 `SCHOLAR_DELAY` 获取，若未设置则为 4.0 秒。

### 方法

##### `search_pubs(pubs: str | DataPacket, max_results: int = 3) -> List[PubMeta]`

根据查询字符串或 `DataPacket` 搜索学术出版物。

- 参数：
  - `pubs`：搜索查询字符串或包含查询内容/元数据的 `DataPacket` 对象。
  - `max_results`：返回的最大结果数（默认：3）。
- **返回值**：包含匹配出版物元数据的 `PubMeta` 对象列表。
- 异常：
  - `ValueError`：输入为空或无效时抛出。
  - `MaxTriesExceededException`：当超过 Google Scholar API 速率限制时抛出。

##### `advanced_search(keyword: Optional[str] = None, year_min: Optional[int] = None, year_max: Optional[int] = None, max_results: int = 5) -> List[PubMeta]`

执行高级搜索，支持关键词和出版年份范围的过滤。

- 参数：
  - `keyword`：搜索关键词（可选）。
  - `year_min`：最早出版年份（可选）。
  - `year_max`：最晚出版年份（可选）。
  - `max_results`：返回的最大结果数（默认：5）。
- **返回值**：符合条件的 `PubMeta` 对象列表。
- 异常：
  - `ValueError`：查询输入无效时抛出。
  - `MaxTriesExceededException`：当超过 Google Scholar API 速率限制时抛出。

## 使用示例（假设程序在EZScholarSearch-main文件夹下）

### 示例 1：基本出版物搜索（有可能会被谷歌拦截，拦截规律未知）

搜索与“机器学习”相关的出版物，最多返回 3 个结果。

```python
from ezscholarsearch.search import ScholarSearch

# 初始化 ScholarSearch 实例
searcher = ScholarSearch(delay=4.0)

# 执行基本搜索
results = searcher.search_pubs("机器学习", max_results=3)

# 打印出版物标题
for pub in results:
    print(pub.bib.get("title", "无标题可用"))
```

### 示例 2：高级搜索（带年份过滤）（受谷歌拦截政策影响极不稳定）

搜索包含“神经网络”的出版物，限定出版年份为 2020 至 2023 年。

```python
from ezscholarsearch.search import ScholarSearch

# 初始化 ScholarSearch 实例
searcher = ScholarSearch()

# 执行高级搜索
results = searcher.advanced_search(keyword="神经网络", year_min=2020, year_max=2023, max_results=5)

# 打印出版物标题和年份
for pub in results:
    print(f"标题: {pub.bib.get('title', '无标题可用')}, 年份: {pub.bib.get('pub_year', '未知')}")
```

## 注意事项

1. **API 速率限制**：库通过动态调整延迟（`delay`）来应对 Google Scholar API 的速率限制。如果遇到 `MaxTriesExceededException`，延迟会自动增加。
2. **输入格式**：支持字符串或 `DataPacket` 输入。`DataPacket` 需包含 `content` 或 `metadata`，否则会抛出 `ValueError`。
3. **日志**：库使用 Python 的 `logging` 模块记录信息、警告和错误，便于调试。
4. **多线程**：`search_pubs` 方法使用 `ThreadPoolExecutor` 并行填充出版物元数据，以提高效率。