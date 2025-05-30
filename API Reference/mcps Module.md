# EZScholarSearch API Reference

## mcps Module

模块`mcps`用于设计和自定义简单的`Agent`和`MCP`结构，AI API 调用使用 openai SDK的接口，可以与`AI`模块联合使用，并且强烈推荐联合

### mcps.DataPacket

与`AI`模块一样，`DataPacket`是数据传输的基本类别

**属性**

*content*: 数据的内容

*metadata*: 保存字典数据或数据标签等

**方法**

*to_str*

将 DataPacket 以字符形式输出

*validate(expected_type: type = str)*

检测 DataPacket.content 是否属于给定的类型

### mcps.Model

与`AI.AIModel`相似，调用AI API得到AI的回答，是一个简化的模型，仅支持system_prompt的传入

mcps.Model(name, role, model, base_url, api_key, system_prompt, temperature)

**参数**

*name*: str: Model实例的名称

*role*: str: Model实例的功能

*base_url*, *api_key*: str: openai SDK 的 base URL 和 API Key

*model*: str: AI API 使用的模型

*system_prompt*: str: 自定义系统提示词，若不传入则通过 `Model.role`自动生成

*temperature*: float = 0.8: 调整模型输出的多样性，数值越大越多样，应该在 0.0 至 2.0 之间

**方法**

*call*(data: mcps.DataPacket | str)

与直接调用: 传入DataPacket或问题的字符，调用AI API得到答案

*descriibe*()

展示模型名字和功能


### mcps.Skill(ABC)

定义具有某种能力的抽象基类

mcps.Skill(name, role)

**参数**

*name*: str: 实例的名字


*role*: str: 实例的功能

**方法**

需要继承并实现call方法，传入的首个参数要支持DataPacket的传入，并最终输出DataPacket

*describe*()

展示模型名字和功能

### mcps.FunctionSkill(Skill)

具有函数功能的`Skill`，最常用的一种Skill

mcps.FunctionSkill(name, role, func)

**参数**

*name*, *role*: str: 名称和功能

*func*: callable: 可调用的函数

**方法**

*call*与直接调用

传入数据，返回函数运行的结果

### mcps.Agent 与 mcps.MCP

Agent 与 MCP 的简单实现，两者的 API和使用方式相似

mcps.Agent(planner, planner_mode, aggregator, aggregator_mode, base_url, api_key, model)

mcps.MCP(planner, base_url, api_key, model)

**参数**

*planner*: Callable[[DataPacket], List[str]] = None: 工具选择器，定义传入的data如何选择合适的工具，若不传入，在`Agent`中由*planner_mode*决定，在`MCP`中则默认使用大模型判断

(Agent)*planner_mode*: Literal['llm', 'literal'] = 'literal': `llm`使用大模型判断使用的工具，`literal`用字符匹配判断

(Agent)*aggregator*: Callable[[Dict[str, DataPacket]], Any]: 定义Agent得到的数据如何整合，输入值是{注册名(str): 输出结果(DataPacket)}的字典

(Agent)*aggregator_mode*: Literal['md', 'default', 'noop'] = 'default': `md`将注册名和内容以markdown形式结合返回; `default`以"{注册名}:\n{内容}"的形式返回; `noop`则保留字典返回，注意这里的字典不会被DataPacket包装

*base_url*, *api_key*, *model*: str: openai SDK 的相关 API 参数

*tools*: 记录{注册名: 工具}的字典

*roles*: 记录{注册名: 功能}的字典

**方法**

*register*(name, role, tool)

将工具注册至 Agent / MCP

name: str: 工具的注册名（与决定是否调用改工具无关，只与输出有关）

role: str: 工具的功能，以此决定是否调用工具

tool: Callable: 工具本身，任何可调用对象、输入并输出DataPacket都可以传入

> 前面提到建议 mcps 模块与 AI 模块搭配使用
> 在这里，来自 AI 模块的 AIModel (更全面版本的 Model), SequentialBlok, ParallelBlock 甚至 WorkFlow 抽象类的子类都可以注册至 Agent 与 MCP，只要正确描述其功能，都可以正确被调用

*call*与直接调用

输入数据，使用相应的工具处理数据并输出返回，注意`MCP`的输出为字典形式

*roles*()

展示注册的工具名称及功能
