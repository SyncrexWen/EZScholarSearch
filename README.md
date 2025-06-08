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

通过`ScholarSearch`搜索的文献自动以`PubMeta`数据元类的格式存储，`ScholarSearch.search_pubs`返回`PubMeta`的列表，更多字段信息详见[datastruct API参考](https://github.com/SyncrexWen/EZScholarSearch/blob/main/API%20Reference/datastructs%20Module.md)

重要提示：`ezscholarsearch.utils`中内置了三种PDF解析器，请按照需求选择使用：

+ `BasePDFParser`: 基于`pymupdf`的PDF解析器
+ `AdvancedPDFParser`: 默认基于`pdfplumber`，同时兼容`pymupdf`的解析器
+ `GrobidPDFParser`(未完成开发测试)：基于使用`Docker`部署`Grobid`的PDF解析器

2. 自定义搭建工作流

简单的自定义工作流使用`ezscholarsearch.AI`中的API实现，`AI`模块受启发于`pytorch.nn`的相关设计

```python
import ezscholarsearch.AI as ai

# 设置 openai SDK 所需的参数
BASE_URL = "https://api.deepseek.com"
API_KEY = "<Your API Key>"
MODEL = "deepseek-chat"

# 使用 AIModelFactory 快速创建 AIModel 实例
fct = ai.AIModelFactory(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
)

# 自定义的工作流需要继承 WorkFlow 抽象类并完成 __init__ 和 forward 两个方法
# 需要说明的是，WorkFlow 定义了元类，使得其子类未经实例化即可被调用
class my_workflow(ai.WorkFlow):
    def __init__(self):
        # ParallelBlock 将数据信息按键值传入，如果未能匹配键值则传入全部信息
        self.conclude = ai.ParallelBlock(
            abstract=fct("这是一份文献的摘要，请梳理研究背景、研究方法和研究结果，并作总结"),
            conclusion=fct("这是一篇文献的结论，请作总结"),
        )
        self.summerize = fct(
            "我会给你两份总结，第一份是关于文章摘要的总结，第二份是关于结论的总结"
            "请据此提出这篇文章的创新点以及其他有价值的信息",
            output_type='str',  # 指定输出类型为字符形式
        )
        self.work = ai.SequentialBlock(
            self.conclude,
            self.summerize
        )
    
    def forward(self, data):
        return self.work(data)

'''
这里假设我们通过前面的方法获取了
paper_info = {
    'title': pub_bib.title,
    'author': pub_bib.author,
    'abstract': abstract,
    'conclusion': conclusion,
}
格式论文信息的列表parsed_papers
'''

processor = ai.SequenceProcessor(my_workflow) # 用于处理序列的工具
results = processor(parsed_papers)
```

重要说明：

+ 在整个工作流中，数据传输的格式依赖数据结构`DataPacket`，若在`summerize`定义时未指定`output_type='str'`则默认会输出`DataPacket`格式的信息
+ 工作流自定义的关键工具为`AIModelFactory`(用于快速创建AIModel实例), `SequentialBlock`(将数据连续传过传入的参数，可以参考`torch.nn.Sequential`的使用), `ParallelBlock`(按关键词平行传入数据)，而他们的基础都是`AIModel`
+ 除了传递数据给AI外，使用`AI.DataProcessor`还可以在任意时刻按照你想的方式处理数据，只需要定义`[DataPacket] -> DataPacket`的函数并传入
+ `SequenceProcessor`和`MultiThreadsSequenceProcessor`都可以用于处理序列信息，由于Python GIL 的存在，使用多线程处理序列效率有时并得不到提升

使用`AI.AIModel`你还可以进行`Tools call`, `few shot learning`的使用，更多详细信息请参考[AI模块API参考](https://github.com/SyncrexWen/EZScholarSearch/blob/main/API%20Reference/AI%20Module.md)

3. 简易的MCP实现

本工具还能够实现简易MCP的实现，主要依赖于`ezscholarsearch.mcps`中的`Agent`和`MCP`两个类，这里仅简单介绍两个类的机制（两个类的机制类似），更具体的运用请参考[MCP 模块 API 参考](https://github.com/SyncrexWen/EZScholarSearch/blob/main/API%20Reference/mcps%20Module.md)

+ Planner 与 Aggregator:

  + Planner: 在 Agent 和 MCP 中，接受传入的数据，并给出给出应该调用哪些工具，内置有`LLM Planner`的实现，通过Openai SDK 的`tools call`调用 LLM 判断使用的工具，可以传入`[DataPacket] -> List[str]`的可调用对象进行自定义，输出列表为调用工具注册名的列表
  + Aggregator: 决定如何综合调用各种工具得到的数据，默认以 MarkDown 格式输出，可以传入`[Dict[str, DataPacket]] -> Any`的可调用对象进行自定义

+ Register 机制：

  工具通过给定注册名和功能注册到 Agent 或 MCP 中

  ```python
  agent.register(
  		name='tokenizer',  # 工具注册名
    	role='将中文文段拆分成token', # 工具注册的功能，与工具调用的选择有关
    	tool=tokenizer,  # 传入可调用对象（工具）
  )
  ```

  只要是可调用对象，都可以注册到 Agent 或 MCP 的实例之中，数据传输格式可以是任何格式，但更推荐使用内置格式`DataPacket`

---

## API 参考

请参阅[完整 API 参考](https://github.com/SyncrexWen/EZScholarSearch/tree/main/API%20Reference)

---

## 开发&贡献

我们在`example test scripts`中内置了用于测试各个模块的脚本，你可以使用我们的脚本进行测试

开发环境建议：

+ Python版本：3.8+
+ 包管理工具：[uv](https://github.com/astral-sh/uv)
+ 测试框架：pytest 或 unittest
+ 代码规范：flake8

---

## LICENSE

本项目基于 MIT 开源协议，具体参阅 [LICENSE](https://github.com/SyncrexWen/EZScholarSearch/blob/main/LICENSE)文件 
