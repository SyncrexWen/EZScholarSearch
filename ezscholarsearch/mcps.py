from __future__ import annotations

from .utils import ToolBuilder
from .AI import _safe_datapacket, DataPacket

from openai import OpenAI
from abc import ABC, abstractmethod
from typing import (Dict, Callable, List, Literal, Any)
from functools import wraps
from collections import OrderedDict

import warnings
import json

__all__ = [
    "ComplexAgent",
    "Model",
    "Agent",
    "Skill",
    "FunctionSkill"
]


def _check_tools(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'tools'):
            raise KeyError(f"Object {self} doesn't have the attribute 'tools'")
        if not self.tools:
            raise ValueError("No tool registered")
        return func(self, *args, **kwargs)
    return wrapper


class ComplexAgent:
    '''复式Agent

    可直接相连的Agent对象

    Args:
        base_url: str: AI API 的 URL
        api_key: str: API Key
        model: str: 使用的模型
        system_prompt: str: 系统级提示词
        tool_choice: str = 'required': 是否启用tool calls选项
        functions: List[Callable] = None: Agent可调用的函数对象
        callback: Callable = None: 调用AI API后结果的处理
    '''
    def __init__(self, base_url: str, api_key: str,
                 model: str, name: str, role: str,
                 temperature: float = 0.8,
                 system_prompt: str = "",
                 tool_choice: str = 'required',
                 functions: List[Callable] = None,
                 callback: Callable = None
                 ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.name = name
        self.role = role
        if not (0.0 <= temperature <= 2.0):
            warnings.warn("Temperature limit exceed. Setting to default.")
            self.temperature = 0.8
        else:
            self.temperature = temperature
        self.tool = ToolBuilder(name, role)
        self.tool.add_param("Main_Question", description=system_prompt)
        self.tool_choice = tool_choice
        self.functions = functions or []
        self.memory = [
            {
                "role": "system",
                "content": (
                    system_prompt or
                    "你是一个乐于助人的人工智能助手，请帮助我解决这些问题"
                )
            }
        ]
        self.connections = {}
        self.callback = callback

    @_safe_datapacket
    def run(self, data: DataPacket, save_message: bool = True):
        '''调用AI API'''
        input_message = (
            data.content or
            "\n\n---\n\n".join(
                [
                    f"## {key}\n\n{value}"
                    for key, value in data.metadata.items()
                ]
            )
        )
        messages = self.memory + [{"role": "user", "content": input_message}]
        if save_message:
            self.memory.append({"role": "user", "content": input_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=[self.tool.build()],
            tool_choice=self.tool_choice,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        arguments = json.loads(tool_calls[0].function.arguments)
        answer = arguments.get("Main_Question", "")
        if save_message:
            self.memory.append({"role": "assistant", "content": answer})
        del arguments['Main_Question']
        for key_word, flag in arguments.items():
            if flag:
                self.connections[key_word].call(answer)

        return DataPacket(content=answer)

    def connect(self, agent: ComplexAgent) -> None:
        '''连接两个ComplexAgent对象'''
        if (
            agent.name in self.connections and
            self.connections[agent.name] is not agent
        ):
            raise KeyError(f"Key Word {agent.name} already occupied")
        self.connections[agent.name] = agent
        self.tool.add_param(
            name=agent.name,
            description=f"你有一个功能为{agent.role}的帮手，请决定是否使用这个它",
            param_type='boolean'
        )

    def connect_both(self, agent: ComplexAgent):
        '''同时连接两方的ComplexAgent对象'''
        self.connect(agent)
        agent.connect(self)

    def call(self, content: DataPacket):
        '''调用Agent网络'''
        datapacket = self.run(content)
        if not self.callback:
            warnings.warn("Your response won't be saved or demostrated "
                          "since callable parameter 'callback' is not passed")
        else:
            self.callback(datapacket)
        return datapacket

    def __call__(self, data: str | DataPacket, save_message: bool = False):
        return self.run(data, save_message)


class Model:
    '''用于执行特定功能的Model对象

    Args:
        name: 模型的名称
        role: 模型的功能
        model: 使用的模型
        base_url: base URL
        api_key: AI API Key
        system_prompt: 系统级提示词
        temperature: float = 0.8: AI生成文字的随机性，应在0.0-2.0之间，数值越大随机性越强
    '''
    def __init__(self, name: str, role: str, model: str, base_url: str,
                 api_key: str, system_prompt: str = None,
                 temperature: float = 0.8):
        self.name = name
        self.role = role
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.memory = [{"role": "system", "content": system_prompt or f"你是一个能够{self.role}的助手"}] # NOQA
        if not (0.0 <= temperature <= 2.0):
            warnings.warn("Temperature limit exceeded, setting to default.")
            self.temperature = 0.8
        else:
            self.temperature = temperature

    @_safe_datapacket
    def call(self, data: DataPacket, save_message: bool = True) -> DataPacket:
        '''调用AI API获取回答'''
        input_message = {"role": "user", "content": data.to_str()}

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.memory[:] + [input_message],
        )

        message = response.choices[0].message.content
        output_message = {"role": "assistant", "content": message}

        if save_message:
            self.memory.extend([input_message, output_message])

        return DataPacket(content=message)

    def describe(self):
        '''展示自身名字和功能'''
        return {
            "name": self.name,
            "role": self.role
        }

    def __str__(self) -> str:
        desc = "<Agent object>"
        desc += "\n    name: " + self.name
        desc += "\n    role: " + self.role
        return desc

    def __call__(self, data: DataPacket | str) -> DataPacket:
        return self.call(data)


class Skill(ABC):
    """
    抽象工具 Skill 类

    Args:
        name: str: 名称
        role: str: 功能
    """
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    def call(self, data: DataPacket, *args, **kwargs) -> DataPacket:
        """
        技能的主执行接口，子类需实现具体逻辑。
        """
        pass

    def describe(self):
        return {
            'name': self.name,
            'role': self.role
        }

    def __str__(self):
        desc = "<Skill object>"
        desc += f"\n    name: {self.name}"
        desc += f"\n    role: {self.role}"
        return desc

    @_safe_datapacket
    def __call__(self, data: DataPacket, *args, **kwargs) -> DataPacket:
        return self.call(data, *args, **kwargs)


class FunctionSkill(Skill):
    def __init__(self, name: str, role: str,
                 func: Callable[[DataPacket], DataPacket]):
        super().__init__(name, role)
        self.func = func

    @_safe_datapacket
    def call(self, data: DataPacket) -> DataPacket:
        return self.func(data)

    def __str__(self):
        desc = "<FunctionSkill object>"
        desc += f"\n    name: {self.name}"
        desc += f"\n    role: {self.role}"
        return desc


class Agent:
    '''智能体对象

    Args:
        planner: 定义如何选择工具
        planner_mode: ['llm', 'literal']: 内建选择工具
        aggregator: 如何整合回答
        aggregator_mode: ['md', 'default', 'noop']: 内建整合器工具
        base_url: AI API URl，如果使用 llm planner 则必须传入
        api_key: AI API Key，如果使用 llm planner 则必须传入
        model: 使用的模型，如果使用 llm planner 则必须传入
    '''
    def __init__(self, planner: Callable[[DataPacket], List[str]] = None,
                 planner_mode: Literal['llm', 'literal'] = 'literal',
                 aggregator: Callable[[Dict[str, DataPacket]], Any] = None, # NOQA
                 aggregator_mode: Literal['md', 'default', 'noop'] = 'default',
                 base_url: str = None, api_key: str = None, model: str = None):
        self.planner = planner or self._default_planner(planner_mode)
        self.aggregator = aggregator or self._default_aggregator(aggregator_mode) # NOQA
        self.tools: Dict[str, Callable] = {}
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.roles = {}
        self._build_tool()

    def register(self, name: str, role: str, tool: Callable) -> None:
        '''注册工具或模型至 Agent'''
        if name in self.tools and not self.tools[name] is tool:
            raise KeyError(f"Name {name} is already registered to {tool}")
        self.tools[name] = tool
        self.roles[name] = role
        self._tool.add_param(
            name=name,
            description=f"这是一个功能为{role}的工具",
            param_type='boolean'
        )

    @_safe_datapacket
    def call(self, data: DataPacket):
        '''选择输入、综合输出'''
        tools_name = self.planner(data)
        if len(tools_name) == 0:
            return None
        if 'no_tool_available' in tools_name:
            del tools_name['no_tool_available']
        results = {
            name: self.tools[name](data)
            for name in tools_name
        }
        aggregated_result = self.aggregator(results)
        if isinstance(aggregated_result, str):
            return DataPacket(content=aggregated_result)
        elif isinstance(aggregated_result, dict):
            return aggregated_result
        else:
            return DataPacket(content=aggregated_result)

    @_check_tools
    def role(self) -> str:
        return "+ " + "\n+ ".join([f"{name}: {role}"
                                   for name, role in self.roles.items()])

    def __str__(self) -> str:
        desc = '<Agent object>\n'
        return desc + ''.join([f"\n{name}:\n{str(tool)}"
                               for name, tool in self.tools.items()])

    def __call__(self, data: str | DataPacket):
        return self.call(data)

    @_check_tools
    def _planner_llm(self, data: DataPacket) -> List[str]:
        '''llm 工具选择器'''
        if self.client is None or self.model is None:
            raise KeyError("Params 'client' and 'model' should pass to use llm planner.") # NOQA
        data_str = data.to_str()
        messages = [
            {"role": "system", "content": "请根据给你的内容或任务选择工具"},
            {"role": "user", "content": data_str}
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=[self._tool.build()],
            tool_choice='required'
        )

        tool_call = response.choices[0].message.tool_calls[0].function.arguments # NOQA
        try:
            arguments: dict = json.loads(tool_call)
        except Exception as e:
            raise ValueError(f"Failed to parse tool arguments: {e}")
        no_tool_available = arguments['no_tool_available']
        del arguments['no_tool_available']
        if no_tool_available and not any(arguments.values()):
            return []
        else:
            return [name for name, flag in arguments.items() if flag]

    @_check_tools
    def _planner_literal(self, data: DataPacket) -> List[str]:
        '''字面值工具选择器'''
        return [name
                for name in self.tools
                if name.lower() in data.to_str().lower()]

    def _aggregator_md(self, results: Dict[str, DataPacket]) -> str:
        '''markdown格式整合器'''
        return "\n\n---\n\n".join([
            f"## {name}\n\n{data.to_str()}"
            for name, data in results.items()
        ])

    def _aggregator_default(self, results: Dict[str, DataPacket]) -> str:
        '''默认整合器'''
        return "\n\n".join([
            f"{name}:\n{data.to_str()}"
            for name, data in results.items()
        ])

    def _aggregator_noop(self, results: Dict[str, DataPacket]) -> Dict[str, str]: # NOQA
        '''返回字典'''
        return {
            name: data.to_str()
            for name, data in results.items()
        }

    def _default_planner(self, planner_mode: str) -> Callable:
        match planner_mode:
            case 'llm':
                return self._planner_llm
            case 'literal':
                return self._planner_literal
            case _:
                raise ValueError(f"Invalid builtin planner mode {planner_mode}") # NOQA

    def _default_aggregator(self, aggregator_mode: str) -> Callable:
        match aggregator_mode:
            case 'md':
                return self._aggregator_md
            case 'default':
                return self._aggregator_default
            case 'noop':
                return self._aggregator_noop
            case _:
                raise ValueError(f"Invalid builtin aggregator mode {aggregator_mode}") # NOQA

    def _build_tool(self):
        '''构建tool'''
        if not hasattr(self, '_tool'):
            self._tool = ToolBuilder("planner_choser",
                                     "请根据给你的内容或任务选择合适的工具，可以选择多个工具，"
                                     "如果没有合适的工具则请在'no_tool_available'处返回True, "
                                     "并在其余工具处返回False")
            self._tool.add_param(
                name="no_tool_available",
                description="是否所有给你的工具都不合适处理这个问题",
                param_type='boolean',
            )


class MCP:
    '''Multi-Agent 控制器 (Meta Control Policy)'''

    def __init__(self, planner: Callable[[str], List[str]] = None,
                 base_url: str = None, api_key: str = None, model: str = None):
        '''
        Args:
            planner: 高层调度函数，根据用户输入文本选择合适 Agent 名称列表
        '''
        self.tools: Dict[str, Agent] = OrderedDict()
        self.planner = planner or self._default_planner
        self._build_tool()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.roles = {}

    def register(self, name: str, role: str, agent: Agent):
        '''注册 Agent'''
        if name in self.agents and not self.agents[name] is agent:
            raise KeyError(f"Agent '{name}' is already registered")
        self.agents[name] = agent
        self._tool.add_param(
            name=name,
            description=f"这是一个功能为{role}的工具",
            param_type='boolean'
        )
        self.roles[name] = role

    @_check_tools
    def roles(self) -> str:
        '''展示所有 Agent 能力'''
        return "\n".join([
            f"{name}<{type(self.tools[name])}>:\n{role}"
            for name, role in self.roles.items()
        ])

    @_safe_datapacket
    def __call__(self, data: DataPacket) -> dict:
        '''根据输入选择 Agent 执行'''
        content = data.to_str()

        selected = self.planner(content)
        results = {}

        for name in selected:
            agent = self.agents.get(name)
            if agent is None:
                raise ValueError(f"Agent '{name}' not found")
            results[name] = agent(data)

        return results

    @_safe_datapacket
    @_check_tools
    def _default_planner(self, data: DataPacket) -> List[str]:
        '''llm 工具选择器'''
        if self.client is None or self.model is None:
            raise KeyError("Params 'client' and 'model' should pass to use llm planner.") # NOQA
        data_str = data.to_str()
        messages = [
            {"role": "system", "content": "请根据给你的内容或任务选择工具"},
            {"role": "user", "content": data_str}
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=self._tool.build(),
            tool_choice='required'
        )

        tool_call = response.choices[0].message.tool_calls[0].function.arguments # NOQA
        try:
            arguments: dict = json.loads(tool_call)
        except Exception as e:
            raise ValueError(f"Failed to parse tool arguments: {e}")
        no_tool_available = arguments['no_tool_available']
        del arguments['no_tool_available']
        if no_tool_available and not any(arguments.values()):
            return []
        else:
            return [name for name, flag in arguments.items() if flag]

    def __str__(self) -> str:
        return f"<MCP: {len(self.agents)} agents registered>\n" + self.roles()

    def _build_tool(self):
        '''构建tool'''
        self._tool = ToolBuilder("planner_choser",
                                 "请根据给你的内容或任务选择合适的工具，可以选择多个工具，"
                                 "如果没有合适的工具则请在'no_tool_available'处返回True, "
                                 "并在其余工具处返回False")
        self._tool.add_param(
            name="no_tool_available",
            description="是否所有给你的工具都不合适处理这个问题",
            param_type='boolean'
        )
