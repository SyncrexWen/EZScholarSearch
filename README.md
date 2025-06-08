# EZScholarSearch

EZScholarSearch 是一个帮助科研工作者结合 AI 高效完成一些科研工作的工具，力求通过简洁高效的 API 辅助科研工作者的工作流。

---

## 项目简介

`EZScholarSearch` 是一个专为科研工作者设计的工具，基于`自定义工作流`和简单`Agent MCP`实现，具有以下特点：

- 简洁易用的 API
- 支持模块化扩展
- 高兼容性和灵活性

> 例如，本工具适用于需要结合 AI 实现高效文献检索、文献下载、文献pdf解析、文献AI总结的研究人员。
>
> 通过使用内置的文献检索、文献下载、pdf解析工具，可以高效搭建自定义的工作流

---

## 快速开始

本项目使用 [`uv`](https://github.com/astral-sh/uv) 进行依赖管理和构建，加速开发体验并支持现代 Python 项目结构。

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果你是 Mac OS 用户，推荐使用 [`HomeBrew`](https://brew.sh/)安装:

```bash
brew install astral-sh/tap/uv
```

> 更多安装细节或问题，请参见[`uv官方文档`](https://github.com/astral-sh/uv)

### 克隆源码并搭建环境

1. 克隆仓库

```bash
git clone https://github.com/SyncrexWen/EZScholarSearch.git
cd EZScholarSearch
```

2. 使用 uv 创建虚拟环境并安装依赖

```bash
uv venv       # 创建虚拟环境
# source .venv/bin/activate  # 激活虚拟环境（Linux/macOS）
# .venv\Scripts\activate    # Windows 下使用此命令激活
uv pip install -e .        # 开发模式安装本地源码
```

### 简单使用示例

1. 文献下载与解析

```python
from ezscholarsearch.search import ScholarSearch
from ezscholarsearch.utils import PubDownloader, BasePDFParser

# 可选设置 UNPAYWALL API 邮箱
# UNPAYWALL 是下载文献资料的途径之一，设置可以提高文献下载成功的可能性
UNPAYWALL_EMAIL = None

scholar = ScholarSearch(delay=10.0)  # 设置延迟减少速率限制

pubs = scholar.search_pubs(
    "transformer",  # 搜索'transformer'有关的论文
    max_results=1,  # 设置最大返回结果数
)

PubDownloader.config_api_email(None)
paper_paths = [PubDownloader.download(pub) for pub in pubs]  # 批量下载文献

if paper_paths:
  	# 以解析第一篇文献为例
    paper_path = paper_paths[0]

    abstract = BasePDFParser.extract_abstract(paper_path)  # 提取摘要内容
    conclusion = BasePDFParser.extract_conclusion(paper_path)  # 提取结论内容
    # 提取论文图片
    img_count = BasePDFParser.extract_images(
        paper_path,
        saving_dir='output images',  # 图片保存的路径
    )

    pub_bib = pubs[0].bib
    paper_info = {
        'title': pub_bib.title,
        'author': pub_bib.author,
        'abstract': abstract,
        'conclusion': conclusion,
    }
```

通过`ScholarSearch`搜索的文献自动以`PubMeta`数据元类的格式存储，`ScholarSearch.search_pubs`返回`PubMeta`的列表，更多字段信息详见[datastruct API参考]
