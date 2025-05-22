# EZScholarSearch API Reference

## AI Module

​模块`AI`用于设计与 AI 交互的工作流WorkFlow，这里支持 openai 库的接口。下面逐一展开介绍 API 信息。

### AI.DataPacket

用于传输信息的标准数据类

**属性**

*content*: 数据的内容

*metadata*: 保存字典数据或数据标签等

**方法**

*to_str*

将 DataPacket 以字符形式输出

*validate(expected_type: type = str)*

检测 DataPacket.content 是否属于给定的类型

### AI.AIModel

class AI.AIModel(base_url, api_key, model, system_prompt, temprature, tools, tool_choice, 
output_type, few_shot_messages)

**参数**

*base_url*: str, api_key: str: 调用 AI API 的 url 和 API Key

*model*: str: AI API 的模型信息

*system_prompt*: str: 用于进行提示词工程的系统级提示词

*temperature*: float = 0.8: 调整模型输出的多样性，数值越大越多样，应该在 0.0 至 2.0 之间

*tools*: List[Tool]: 可能对被大模型调用的工具的列表，可以得到格式化的输出。使用示例：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "introduce_law",
            "description": "用中文介绍公式的内容、发明者、发现的历史",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "公式的内容"
                    },
                    "discoverer": {
                        "type": "string",
                        "description": "公式的发现者"
                    },
                    "history": {
                        "type": "string",
                        "description": "公式发现的历史"
                    }
                },
                "required": ["content", "discoverer", "history"]
            }
        }
    }
]
```

这样传入后，调用模型就可以得到如下格式的输出（示例输入为'热力学三定律'）:

```python
output = {
    "content": '热力学三定律是热力学的基本定律，包括：\n1. 热力学第一定律（能量守恒定律）：能量不能被创造或消灭，只能从一种形式转化为另一种形式。\n2. 热力学第二定律（熵增定律）：在一个孤立系统中，熵总是趋向于增加，即系统的无序度会不断增加。\n3. 热力学第三定律：当温度趋近于绝对零度时，系统的熵趋近于一个极小值，通常为零。',
    ""
    "discoverer": "热力学第一定律由多位科学家（如焦耳、亥姆霍兹等）共同发展；热力学第二定律由克劳修斯和开尔文提出；热力学第三定律由能斯特提出。",
    "history": "热力学三定律的发展贯穿了19世纪和20世纪初。第一定律源于能量守恒的研究，第二定律揭示了热力学过程的不可逆性，第三定律则是对低温下系统行为的总结。"
}
```

你也可以使用 `utils.ToolBuilder` 方便地创建 Tool ，并使用 `utils.ToolRegistry` 记录创建过的 Tool 。

*tool_choice*: str = 'auto': AI 何时使用Tools中的工具，字面值可以为：

+ 'none': 无论如何都不使用tools
+ 'auto': 让 AI 自动判断是否使用 Tool 以及使用哪一个 Tool
+ 'required': 让 AI 必须使用 Tools 中的一个 Tool
+ Tools 中某个 Tool 的名称: 明确要求模型调用哪个 Tool

*output_type*: str = 'default': AI 模型输出使用哪种类型，若注明 'str' 则以字符串形式返回，若为其他值则用程序的数据交互协议 AI.DataPacket 返回

*few_shot_messages*: List[Dict]: 可选使用 few-shot 的方法调整 AI 的输出，使用实例:

```python
model = AI.AIModel(
    base_url='https://api.deepseek.com',
    api_key='<Your API Key>',
    model='deepseek-chat',
    system_prompt='请以一个历史专家的视角回答这个问题'
    few_shot_messages=[
        {"role": "user", "content": "你是一位历史学专家，用户将提供一系列问题，你的回答应当简明扼要，并以`Answer:`开头"},
        {"role": "assistant", "content": "Answer:公元前221年"},
        {"role": "user", "content": "请问汉朝的建立者是谁？"},
        {"role": "assistant", "content": "Answer:刘邦"},
        {"role": "user", "content": "请问唐朝最后一任皇帝是谁"},
        {"role": "assistant", "content": "Answer:李柷"},
        {"role": "user", "content": "请问明朝的开国皇帝是谁？"},
        {"role": "assistant", "content": "Answer:朱元璋"},
],
    output_type='str'
)

output = model("请问清朝的开国皇帝是谁？")

'''
实例输出：
Answer: 皇太极
'''
```

**方法**

AIModel 实现了 `__call__` 方法，这就意味着你可以直接调用 AIModel 的实例

`AIModel.__call__`

*data*: str | AI.DataPacket: 传入提问的信息

*output_type*: str = 'default': 输出的类型，如果未指定则为 AIModel 定义时的 output_type

### AI.AIModelFactory

创建 AIModel 实例时，反复输入重复的参数显得十分麻烦，创建`AIModelFactory`实例能够帮助你解决这个问题

**参数**

*base_url*, *api_key*, *model* 配置 AIModel 的参数

**方法**

AIModelFacotry 实现了 `__call__` 方法，调用 AIModelFactory 的实例可以方便地创建 AIModel 实例

`AIModelFactory.__call__`

*system_prompt: str, temperature, tools, tool_choice, output_type, few_shot_messages* 创建 AIModel 的剩余参数

### AI.DataProcessor

一个用于处理 DataPacket 的数据流处理类，支持函数链式组合、模式选择和命名缓存。

**参数**

*callback*: Callable | Sequence[Callable] | None: 单个或多个回调函数，每个函数接受并返回一个 DataPacket 实例。函数链按照顺序逐个执行。

*mode*: Literal['print', 'noop'] | None可选内建处理模式：
+ 'print': 将 DataPacket 的内容或元数据打印到控制台。
+ 'noop': 无操作处理，直接返回原始数据。
+ 默认为 None，表示不启用内建模式。

*name*: str | None: 命名当前处理器，支持通过名称复用或缓存回调组合。如果设置了 name 且未提供 callback，将尝试从缓存中加载同名处理器。

**方法**

DataPacket 同样实现了 `__call__` 方法，调用 DataProcessor 的实例，使用缓存的回调函数处理 DataPacket

DataPacket 还实现了 `__rshift__` 方法，使用右移运算符 `>>` 可以方便的将 Callable 加入到 DataProcessor 中

### AI.SequentialBlock

这是一个链接多个 Callable 的模块，按顺序传入多个 Callable (一般为 AI.AIModel, AI.DataProcessor 以及嵌套的 AI.SequentialBlock, AI.ParallelBlock)，随后调用 AI.SequentialBlock 的实例，会按顺序将数据传过这些模块，最后返回

### AI.ParallelBlock

这是一个接受多个 Callable 的平行处理模块，传入的 Callable 需要按关键词传入

调用 ParallelBlock 的实例，传入的数据 data（自动包装为 AI.DataPacket）会优先使用 data.metadata 的键值与 Callable 的键值进行匹配，如果匹配失败，则会将 data.content 的传给每一个关键词的 Callable

返回的数据类型为 AI.DataPacket, 其 content 的值为 None，metadata 为每个 Callable 的关键词与该 Callble 对应输出的字典

如果你需要将某些数据精确地传给某些关键词的 Callable，在前一层的输出使用 Tools Call是一个很好的选择，你也可以自定义 DataProcessor 对数据信息进行分割

### AI.SequenceProcessor 和 AI.MultiThreadsSequenceProcessor

AI.SequenceProcessor 和 AI.MultiThreadsSequenceProcessor 是一个批量处理序列的工具，他们的实例化可以传入一系列的 Callable，调用实例会对传入序列中的元素按顺序调用这一系列的 Callable，输出经过处理后结果的序列

AI.MultiThreadsSequenceProcessor 使用多线程处理序列，实例化时还需传入 max_worker: int(初始值为 5)

由于 Python 多线程存在的一系列问题，大部分轻度到中等强度的任务中 AI.MultiThreadsSequenceProcessor 的表现都不如 AI.SequenceProcessor，因此建议优先使用 AI.SequenceProcessor

### AI.WorkFlow

定义工作流的关键抽象类，继承 AI.WorkFlow 需要子类完成 `__init__` 方法和 `forward` 方法，定义完成的子类无需实例化即可直接调用

使用示例：

```python
fct = ai.AIModelFactory(
    base_url=BASE_URL,
    api_key=API_KEY,
    model='deepseek-chat'
)

saving_dir = 'outputs'

def save_datapacket_output(title: str):
    def wrapper(datapacket: ai.DataPacket):
        if datapacket.content:
            save_md(title, datapacket.content)
        elif datapacket.metadata:
            save_json(title, datapacket.metadata)
        return datapacket
    return wrapper

def save_md(title: str, content: str, saving_dir=saving_dir):
    path = os.path.join(saving_dir, title + ".md")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(f"# {title}\n---\n")
        file.write(content)
    return content

def save_json(title: str, content: dict, saving_dir=saving_dir):
    path = os.path.join(saving_dir, title + ".md")
    with open(path, 'w', encoding='utf-8') as file:
        file.write(f"# {title}\n\n")
        file.write("\n---\n".join([
            f"## {key}\n\n{value}"
            for key, value in content.items()
        ]))
    return content

class MyWorkFlow(ai.WorkFlow):
    def __init__(self):

        self.search = fct('为我寻找这个作者的诗，输出仅保留标题和内容，注意千万不要出现作者名称', temperature=1.4)

        self.save_search = ai.DataProcessor(save_datapacket_output("诗歌搜索"))

        self.feature = ai.ParallelBlock(
            summerize=fct("请简要总结这些诗歌的风格，注意不要保留任何作者和作品名称", temperature=1.4),
            translate=fct('请将这些诗歌有诗意地翻译成英文', temperature=1.8),
            content=fct("接下来我会给你一些诗歌，我需要你完成下面的任务：我们要进行一个游戏，描述一个诗人诗歌的内容、诗人的人生态度等，让别人猜诗人是谁。"
                        "现在请你进行出题，注意不能暴露与诗人直接关联的任何信息。"
                        "不能暴露的信息包括：具体诗句、诗句中的意向、诗人名称、诗人所处的时代、诗人与其他人的关系、诗人的字或别名等",
                        temperature=1.5)
        )

        self.save_feature = ai.DataProcessor(save_datapacket_output("诗歌特征"))

        self.guess = ai.SequentialBlock(
            fct("我会给你一些诗歌内容的总结，假如这些诗歌由我创作，我在你眼里是个什么样的人？请以'我'为主语进行回答", temperature=1.6),
            ai.DataProcessor(save_datapacket_output("诗人画像")),
            fct("我是一个诗人，接下来我会给你一些别人眼中我的样子，请你猜猜我是谁，并简要说明原因", temperature=1.6),
            ai.DataProcessor(save_datapacket_output("诗人猜测"))
        )

    def forward(self, data):
        data = self.search(data)
        data = self.save_search(data)
        data = self.feature(data)
        data = self.save_feature(data)
        data = ai.DataPacket(content=data.metadata['content'])
        assert data.validate(str)
        data = self.guess(data)
        return data

output = MyWorkFlow("苏轼")
```

于是我们就在`outputs`文件夹下得到了一系列输出文件
