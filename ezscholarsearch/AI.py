from __future__ import annotations

from .datastructs import Messages
from .utils import FILE_CONFIG, DynamicLogger, ToolBuilder

from openai import AsyncOpenAI, OpenAI
from typing import (Tuple, Dict, Any, Union,
                    Iterable, Callable, Type,
                    Optional, List, Literal,
                    Sequence)
from functools import wraps
from time import sleep, perf_counter
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable as abcIterable
from functools import reduce

import openai
import inspect
import sys
import threading
import warnings
import asyncio
import json

__all__ = [
    # 'AsyncOpenAIClient',
    'OpenAIClient',
    'DataPacket',
    # 'AsyncAIModelFactory',
    # 'AsyncAIModel',
    # 'AsyncDataProcessor',
    # 'AsyncWorkFlow',
    # 'AsyncSequentialBlock',
    # 'AsyncParallelBlock',
    'AIModel',
    'AIModelFactory',
    'WorkFlow',
    'DataProcessor',
    'SequentialBlock',
    'ParallelBlock',
    'SequenceProcessor',
    'MultiThreadsSequenceProcessor',
]


def _handle_openai_error(e: Exception, logger: DynamicLogger):
    '''处理捕获的OpenAI错误'''
    if isinstance(e, openai.BadRequestError):
        print(f"Parameters Invalid, {e}")
        logger.error("BadRequestError, %s", e)
    elif isinstance(e, openai.AuthenticationError):
        print(f"Wrong or Expired API Key, {e}")
        logger.error("AuthenticationError, %s", e)
    elif isinstance(e, openai.RateLimitError):
        print(f"Access Frequency Exceeds Limit, {e}")
        logger.error("RateLimitError, %s", e)
    elif isinstance(e, openai.APIConnectionError):
        print(f"Failed to Connect, {e}")
        logger.error("APIConnectionError, %s", e)
    elif isinstance(e, openai.APITimeoutError):
        print(f"Request Timeout, {e}")
        logger.error("APITimeoutError, %s", e)
    elif isinstance(e, openai.InternalServerError):
        print(f"Failed to Connect the Server, {e}")
        logger.error("InternalServerError, %s", e)
    elif isinstance(e, KeyboardInterrupt):
        print(f"Keyboard Interrupt, {e}")
        logger.error("Keyboard Interrupt, %s", e)
    else:
        print("Exception:", e)
        logger.error("Exception, %s", e)


def openai_error_catcher(func):
    '''捕捉OpenAI错误并记录报告的装饰器

    支持同步和异步函数
    '''
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self_obj = args[0]
                logger = getattr(self_obj, 'logger', None)
                if logger:
                    _handle_openai_error(e, logger)
                else:
                    print(f"[Warning] No logger found on {self_obj}")
                    print(e)
                return None
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self_obj = args[0]
                logger = getattr(self_obj, 'logger', None)
                if logger:
                    _handle_openai_error(e, logger)
                else:
                    print(f"[Warning] No logger found on {self_obj}")
                    print(e)
                return None
        return sync_wrapper


def retry(
    retries: int = 3,
    wait: Union[int, Iterable[Union[int, float]]] = 2,
    exceptions: Union[
        Type[Exception], Tuple[Type[Exception], ...]
        ] = Exception,
    warn: bool = True
):
    """
    支持同步和异步函数的重试装饰器

    Args:
        retries: 重试次数
        wait: 等待时间(秒)，可以是数字或可迭代对象
        exceptions: 触发重试的异常类型
        warn: 重试耗尽后是否发出警告
    """
    if isinstance(wait, (int, float)):
        wait = [wait] * retries
    elif isinstance(wait, Iterable):
        wait = list(wait)
        if len(wait) < retries:
            wait += [wait[-1]] * (retries - len(wait))
    else:
        raise TypeError("wait must be a number or iterable")

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                last_exception: Optional[Exception] = None
                for i in range(retries):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if i < retries - 1:
                            await asyncio.sleep(wait[i])

                if warn:
                    warnings.warn(
                        f"Async function {func.__name__} failed "
                        f"after {retries} retries. "
                        f"Last exception: {str(last_exception)}",
                        RuntimeWarning
                    )
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError("Retry failed")

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any):
                last_exception: Optional[Exception] = None
                for i in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if i < retries - 1:
                            sleep(wait[i])

                if warn:
                    warnings.warn(
                        f"Function {func.__name__} failed "
                        f"after {retries} retries. "
                        f"Last exception: {str(last_exception)}",
                        RuntimeWarning
                    )
                if last_exception:
                    raise last_exception
                else:
                    raise RuntimeError("Retry failed")
            return sync_wrapper
    return decorator


class AsyncOpenAIClient:
    '''异步的OpenAI API客户端接口'''
    def __init__(self, base_url: str, api_key: str,
                 model: str, init_msg: str = None,
                 logger_config: Dict[str, Any] = None):
        '''初始化

        Args:
            base_url: API url
            api_key: API Key
            model: 聊天执行的模型
            init_msg: 可选，初始化消息
            logger_config: 日志记录器的装饰器
        '''
        self._base_url = base_url
        self._api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        if not init_msg:
            init_msg = "你是一个乐于助人的人工智能助手，请用中文回答问题"
        self._init_msg = init_msg
        self.messages = Messages([{"role": "system", "content": init_msg}])
        self.logger = DynamicLogger(
            name=f"{self.__class__.__name__}",
            initial_config=logger_config or FILE_CONFIG,
        )

    @property
    def base_url(self):
        return self._base_url

    @base_url.setter
    def base_url(self, new_url: str):
        '''设置base_url时修改client属性'''
        self._base_url = new_url
        self.client = AsyncOpenAI(base_url=self._base_url,
                                  api_key=self._api_key)

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, new_api_key: str):
        '''设置api_key时修改client'''
        self._api_key = new_api_key
        self.client = AsyncOpenAI(base_url=self._base_url,
                                  api_key=self._api_key)

    @openai_error_catcher
    async def ask(self, prompt: str, /,
                  save_messages: bool = False, *,
                  temperature: float = 0.8,
                  timeout: int = 30,
                  max_tokens: int = 4_000) -> Tuple[str, Dict]:
        '''对单个问题进行提问，返回回答

        Args:
            prompt: 传入的问题
            save_messages: 是否改变自身messages属性
            temperature: 调节回答的丰富性
            timeout: 超时时间
            max_tokens: 输出的最大tokens数

        Returns:
            回复的消息
            tokens使用情况
        '''
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Input cannot be empty.")
        if not (0 <= temperature <= 2.0):
            raise TypeError("Temperature must be within 0.0-2.0.")
        if save_messages:
            self.messages.add({"role": "user", "content": prompt})
            chat_messages = self.messages
        else:
            chat_messages = Messages([
                {"role": "system", "content": self._init_msg},
                {"role": "user", "content": prompt},
            ])
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages.messages,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        response_msg = response.choices[0].message.content
        if save_messages:
            self.messages.add(response.choices[0].message.to_dict())
        return response_msg, response.usage.to_dict()

    @openai_error_catcher
    async def chat(self, *, temperature: float = 0.8,
                   timeout: int = 30, stream: bool = True,
                   max_tokens: int = 4_000) -> None:
        '''与AI进行多轮聊天

        Args:
            temperature: 调节回答的丰富性
            timeout: 超时时间
            stream: 是否启用流式输出
            max_tokens: 输出的最大tokens数
        '''
        if not (0 <= temperature <= 2.0):
            raise TypeError("Temperature must be within 0.0-2.0.")

        print("-"*50+f"\n开始与{self.model}聊天")
        print("输入'q'或'quit'来退出\n")
        while (input_msg := input("你:").strip().lower()) not in ('q', 'quit'):
            if not input_msg:
                print("输入信息不能为空!\n")
            print(f"{self.model}: ")
            start = perf_counter()
            self.messages.add({'role': 'user', 'content': input_msg})

            stop_event = threading.Event()
            animation_thread = threading.Thread(target=self._load_animation,
                                                args=(stop_event,))
            animation_thread.start()

            response = await self.client.completions.create(
                model=self.model,
                messages=self.messages.messages,
                temperature=temperature,
                stream=stream,
                timeout=timeout,
                max_tokens=max_tokens,
            )

            if stream:
                assistant_msg = await self._process_stream(response,
                                                           animation_thread,
                                                           stop_event)
            else:
                assistant_msg = await self._process_nonstream(response,
                                                              animation_thread,
                                                              stop_event)
            end = perf_counter()
            self.messages.add(assistant_msg)
            print(f"模型在{end - start: .3f}内响应\n")
        print("聊天结束\n" + "-"*50)

    async def _process_stream(self, response: openai.Stream,
                              animation_thread: threading.Thread,
                              stop_event: threading.Event) -> Dict[str, str]:
        '''处理流式响应得到的信息'''
        full_msg = []
        try:
            async for chunk in response:
                chunk_content = chunk.choices[0].delta.content

                if not stop_event.is_set() and chunk_content:
                    stop_event.set()
                    animation_thread.join()

                if chunk_content:
                    full_msg.append(chunk_content)
                    print(chunk_content, end='', flush=True)
        except Exception as e:
            print(f"流式响应时发生错误, {e}")
            return {"role": "assistant", "content": ""}

        return {"role": "assistant", "content": "".join(full_msg)}

    async def _process_nonstream(self, response,
                                 animation_thread: threading.Thread,
                                 stop_event: threading.Event
                                 ) -> Dict[str, str]:
        '''处理非流式响应得到的信息'''
        try:
            stop_event.set()
            animation_thread.join()
            response_content = response.choices[0].message.content
            print(response_content, flush=True)
        except Exception as e:
            print(f"获取响应时发生错误 {e}")
            return {"role": "assistant", "content": ""}
        return {"role": "assistant", "content": response_content}

    def save_messages(self, path: str = None, *, name: str = None):
        '''保存messages属性至path或name指向的文件'''
        self.messages.save_to_file(path=path, name=name)

    def clear_messages(self):
        '''清空消息存储'''
        self.messages.clear()

    def _load_animation(self, stop_event: threading.Event) -> None:
        '''加载动画'''
        def sys_prt(s: str):
            sys.stdout.write(s)
            sys.stdout.flush()
        animation_characters = r"\|/-"
        cnt = 0
        sys_prt(" ")
        while not stop_event.is_set():
            sys_prt(f"\b{animation_characters[cnt % 4]}")
            cnt += 1
            sleep(0.1)
        sys_prt("\b")


class OpenAIClient:
    '''同步的OpenAI API客户端'''
    def __init__(self, base_url: str, api_key: str,
                 model: str, init_msg: str = None,
                 logger_config: Dict[str, Any] = None):
        '''初始化

        Args:
            base_url: api访问的url
            api_key: API Key
            model: 使用的模型
            init_msg: 系统消息
            logger_config: logger的配置选项
        '''
        self._base_url = base_url
        self._api_key = api_key
        self.model = model
        if not init_msg:
            init_msg = "你是一个乐于助人的助手，请用中文回答问题"
        self._init_msg = init_msg
        self.messages = Messages([{"role": "system", "content": init_msg}])
        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        self.logger = DynamicLogger(
            name=f"{self.__class__.__name__}",
            initial_config=logger_config or FILE_CONFIG,
        )

    @property
    def base_url(self):
        return self._base_url

    @base_url.setter
    def base_url(self, new_url):
        '''设置base_url时同步修改client'''
        self._base_url = new_url
        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, new_api_key):
        '''修改api_key时同步修改client'''
        self._api_key = new_api_key
        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    @openai_error_catcher
    def ask(self, prompt: str, /,
            save_messages: bool = False, *,
            temperature: float = 0.8,
            timeout: int = 30,
            max_tokens: int = 4_000):
        '''对单个问题进行提问，返回回答

        Args:
            prompt: 传入的问题
            save_messages: 是否改变自身messages属性
            temperature: 调节回答的丰富性
            timeout: 超时时间
            max_tokens: 输出的最大tokens数

        Returns:
            回复的消息
            tokens使用情况
        '''
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Input cannot be empty.")
        if not (0 <= temperature <= 2.0):
            raise TypeError("Temperature must be within 0.0-2.0.")
        if save_messages:
            self.messages.add({"role": "user", "content": prompt})
            chat_messages = self.messages
        else:
            chat_messages = Messages([
                {"role": "system", "content": self._init_msg},
                {"role": "user", "content": prompt},
            ])
        response = self.client.chat.completions.create(
            messages=chat_messages,
            model=self.model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens
        )
        response_content = response.choices[0].message.content
        if save_messages:
            self.messages.add(response.choices[0].message.to_dict())
        return response_content, response.usage.to_dict()

    @openai_error_catcher
    def chat(self, *, temperature: float = 0.8,
             timeout: int = 30, max_tokens: int = 30,
             stream: bool = True):
        '''与AI进行多轮聊天

        Args:
            temperature: 调节回答的丰富性
            timeout: 超时时间
            stream: 是否启用流式输出
            max_tokens: 输出的最大tokens数
        '''
        if not (0 <= temperature <= 2.0):
            raise TypeError("Temperature must be within 0.0~2.0")

        print("-"*50 + f"开始与{self.model}聊天")
        print("输入\"q\"或\"quit\"退出")

        while (input_msg := input("你: ").strip()).lower() not in ("q", "quit"):
            if not input_msg:
                print("输入不能为空")
                continue

            start = perf_counter()

            stop_event = threading.Event()
            animation_thread = threading.Thread(target=self._load_animation,
                                                args=(stop_event,))

            response = self.client.completions.create(
                messages=self.messages.messages,
                stream=stream,
                model=self.model,
                max_tokens=max_tokens,
                timeout=timeout,
                temperature=temperature,
            )

            if stream:
                assistant_msg = self._process_stream(response, stop_event,
                                                     animation_thread)
            else:
                assistant_msg = self._process_nonstream(response, stop_event,
                                                        animation_thread)

            self.messages.add(assistant_msg)

            end = perf_counter()
            print(f"模型在{end - start: .3f}内响应\n")
        print("聊天结束，退出程序\n", "-"*50)

    def save_messages(self, path: str = None, *, name: str = None):
        '''保存messages属性至path或name指向的文件'''
        self.messages.save_to_file(path=path, name=name)

    def clear_messages(self):
        '''清空消息存储'''
        self.messages.clear()

    def _process_stream(self, response: openai.Stream,
                        stop_event: threading.Event,
                        animation_thread: threading.Thread):
        '''处理流式响应'''
        full_content = []
        try:
            for chunk in response:
                chunk_content = chunk.choices[0].delta.content
                if not stop_event.is_set() and chunk_content:
                    stop_event.set()
                    animation_thread.join()

                if chunk_content:
                    full_content.append(chunk_content)
                    print(chunk_content, end='', flush=True)
        except Exception as e:
            print(f"流式响应时发生错误, {e}")
            return {"role": "assistant", "content": ""}
        return {"role": "assistant", "content": "".join(full_content)}

    def _process_nonstream(self, response,
                           stop_event: threading.Event,
                           animation_thread: threading.Thread
                           ):
        '''处理非流式响应'''
        try:
            stop_event.set()
            animation_thread.join()
            response_content = response.choices[0].message.content
            print(response_content, flush=True)
        except Exception as e:
            print(f"响应发生错误, {e}")
            return {"role": "assistant", "content": ""}
        return {"role": "assistant", "content": response_content}

    def _load_animation(self, stop_event: threading.Event):
        '''加载动画'''
        def sys_prt(s: str):
            sys.stdout.write(s)
            sys.stdout.flush()

        animation_characters = r"\|/-"
        cnt = 0
        sys_prt(" ")
        while not stop_event.is_set():
            sys_prt(f"\b{animation_characters[cnt % 4]}")
            cnt += 1
            sleep(0.1)
        sys_prt("\b")


def _strf_datapacket(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self,
                                data: str | dict | DataPacket,
                                *args, **kwargs):
            if isinstance(data, str):
                input_data = DataPacket(content=data)
            elif isinstance(data, dict):
                input_data = DataPacket(content=None, metadata=data)
            elif isinstance(data, DataPacket):
                input_data = data
            else:
                raise TypeError(f"Invalid Input Type {type(data)}")
            return await func(self, input_data, *args, **kwargs)
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(self, data: str | dict | DataPacket, *args, **kwargs):
            if isinstance(data, str):
                input_data = DataPacket(content=data)
            elif isinstance(data, dict):
                input_data = DataPacket(content=None, metadata=data)
            elif isinstance(data, DataPacket):
                input_data = data
            else:
                raise TypeError(f"Invalid Input Type {type(data)}")
            return func(self, input_data, *args, **kwargs)
        return sync_wrapper


def _sequencef_datapacket(func):
    @wraps(func)
    def wrapper(self, data: DataPacket | Any, *args, **kwargs):
        if isinstance(data, DataPacket):
            if isinstance(data.content, abcIterable):
                return func(self, data.content, *args, **kwargs)
            else:
                return func(self, data.metadata.values, *args, **kwargs)
        else:
            if isinstance(data, abcIterable):
                return func(self, data, *args, **kwargs)
            else:
                return func(self, [data], *args, **kwargs)
    return wrapper


def _safe_datapacket(func):
    @wraps(func)
    def wrapper(self, data: Any, *args, **kwargs):
        if isinstance(data, DataPacket):
            return func(self, data, *args, **kwargs)
        elif isinstance(data, dict):
            return func(self, DataPacket(content=None, metadata=data),
                        *args, **kwargs)
        else:
            return func(self, DataPacket(content=data), *args, **kwargs)
    return wrapper


@dataclass
class DataPacket:
    '''信息交互协议

    Args:
        content: 数据内容
        metadata: 数据标记，用于传入的function calling，定义数据的结构
    '''
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_type: type = str):
        '''检测content类型是否如预期'''
        if expected_type and not isinstance(self.content, expected_type):
            return False
        return True

    def to_str(self):
        if self.content:
            return self.content
        return "\n\n---\n\n".join([
            f"## {key}\n\n{value}"
            for key, value in self.metadata.items()
        ])


class AsyncAIModelFactory:
    '''AIModel类的实例化工厂

    根据特定的base_url, api_key, model创建AIModel的实例

    Args:
        base_url: API url
        api_key: API Key
        model: 使用的模型
    '''
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.config = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }

    def __call__(self, prompt, temperature: float = 0.8,
                 functions: List[Dict[str, Any]] = None,
                 function_call: str = "auto",
                 output_type: str = 'default') -> "AsyncAIModel":
        '''创建AIModel实例

        Args:
            prompt: 传入的system prompt
            temperature: AI回答的随机性
            function_call: AI输出的
        '''
        return AsyncAIModel(
            prompt=prompt,
            temperature=temperature,
            **self.config,
            functions=functions,
            function_call=function_call,
            output_type=output_type,
        )


class AsyncAIModel:
    '''特定功能的AI模型实例，支持system prompt和cuntion calling'''
    def __init__(self, base_url: str, api_key: str,
                 model: str, prompt: str,
                 temperature: float = 0.8,
                 functions: List[Dict[str, Any]] = None,
                 function_call: str = 'auto',
                 output_type: str = 'default'):
        '''创建实例

        Args:
            base_url: API URL
            api_key: API Key
            temperature: AI回复的随机性
            functions: 传入function calling的函数
            function_call: 是否使用functions
        '''
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.messages = [{
            "role": "system",
            "content": prompt
        }]
        self.config = {
            "model": model,
            "temperature": temperature,
            "functions": functions,
            "function_call": function_call
        }
        self.output_type = output_type

    @retry()
    @_safe_datapacket
    async def _ask(self, data: Union[DataPacket, str],) -> DataPacket:
        '''根据输入的data调用AI生成回复'''
        if data.content:
            input_messages = {"role": "user", "content": data.content}
        else:
            input_messages = {"role": "user",
                              "content": "\n".join(data.metadata.values())}
        if not input_messages["content"].strip():
            if self.output_type == 'str':
                return ''
            return DataPacket(None)
        response = await self.client.chat.completions.create(
            messages=self.messages + [input_messages],
            **self.config,
        )
        message = response.choices[0].message

        function_call = message.function_call
        if function_call:
            arguments = json.loads(function_call.arguments)
            if self.output_type == 'str':
                return '\n'.join(arguments.values())
            return DataPacket(content=None, metadata=arguments)
        else:
            if self.output_type == 'str':
                return message.content or ""
            return DataPacket(
                content=message.content or "",
            )

    async def __call__(self, data: DataPacket | str):
        return await self._ask(data)


class AsyncDataProcessor:
    '''数据处理器

    在SequentialBlock和ParallelBlock中处理信息
    '''
    def __init__(self, callback: Callable):
        '''
        Args:
            callback: 用于处理数据的函数，需要输入DataPacket，输出DataPacket
        '''
        self.callback = callback

    @_safe_datapacket
    async def _process(self, data):
        if inspect.isasyncgenfunction(self.callback):
            return [item async for item in self.callback(data)]
        elif inspect.iscoroutinefunction(self.callback):
            return await self.callback(data)
        else:
            return await asyncio.to_thread(self.callback, data)

    async def __call__(self, data: DataPacket):
        return await self._process(data)


class AsyncWorkFlow(ABC):
    '''定义工作流抽象类'''
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    async def forward(self, data: str | DataPacket):
        pass

    async def post_execution(self, data: str | DataPacket):
        pass

    async def _post_forward(self, data: str | DataPacket):
        await self.post_execution(data)
        return data

    async def __call__(self, data: str | DataPacket) -> DataPacket:
        data = await self.forward(data)
        data = await self._post_forward(data)
        return data


class AsyncSequentialBlock:
    '''连续工作流板块'''
    def __init__(self, *AIModels: Union[
        "AsyncAIModel", "AsyncSequentialBlock",
        "AsyncParallelBlock", "AsyncDataProcessor"
    ]):
        self.AIModels = AIModels

    async def __call__(self, data: DataPacket):
        new_data = data
        for aimodel in self.AIModels:
            new_data = await aimodel(new_data)
        return new_data


class AsyncParallelBlock:
    '''平行工作流板块'''
    def __init__(self,
                 input_mapper: Optional[
                     Callable[[DataPacket], Dict[str, DataPacket]]
                     ] = None,
                 **models: Union[
                     "AsyncAIModel", "AsyncSequentialBlock",
                     "AsyncParallelBlock", "AsyncDataProcessor"
                 ]):
        self.kw_models = models
        self.input_mapper = input_mapper

    async def __call__(self, data: DataPacket) -> DataPacket:
        if self.input_mapper:
            input_data = self.input_mapper(data)
            if (
                not isinstance(input_data, dict) or
                not all(k in input_data for k in self.kw_models)
            ):
                raise ValueError("Input mapper must "
                                 "return a dict with valid model keys")
        else:
            input_data = {kw: data for kw in self.kw_models}

        tasks = []
        for kw, input_d in input_data.items():
            if kw not in self.kw_models:
                continue
            tmp_output = await self.kw_models[kw](input_d)
            tasks.append((kw, tmp_output))

        if not tasks:
            raise ValueError("No valid keywords matched any models")

        results = await asyncio.gather(*[task[1] for task in tasks],
                                       return_exceptions=True)
        ret_data = {}
        for (kw, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                raise result
            ret_data[kw] = result.content

        return DataPacket(
            content=ret_data,
            metadata={"source": "ParallelBlock",
                      "model_keys": list(self.kw_models.keys())}
        )


class AIModel:
    '''特定功能的AI模型实例，支持system prompt和 tools call'''
    def __init__(self,
                 base_url: str, api_key: str,
                 model: str, system_prompt: str,
                 temperature: float = 0.8,
                 tools: List[Dict[str, Any]] = None,
                 tool_choice: str = 'auto',
                 output_type: Literal['default', 'str'] = 'default',
                 few_shot_messages: List[Dict[str, str]] = None):
        '''创建实例

        Args:
            base_url: API URL
            api_key: API Key
            temperature: AI回复的随机性
            tools: 传入tools call的函数
            tool_call: 何时使用 tools call
        '''
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self._messages = [{
            'role': 'system',
            'content': system_prompt,
        }]
        if few_shot_messages is not None:
            self._messages.extend(few_shot_messages)
        self._config = {
            'model': model,
            'temperature': temperature,
            'tools': tools,
            'tool_choice': tool_choice
        }
        self.output_type = output_type
        if self.output_type not in ('default', 'DataPacket', 'str'):
            raise ValueError("Invalid output type")

    @retry()
    @_strf_datapacket
    def _ask(self, data: DataPacket, output_type: str = None,) -> DataPacket:
        '''根据输入的data调用AI生成回复（兼容 tools call 机制）'''
        input_message = data.content or "\n---\n".join([
            f"# {key}\n\n{value}"
            for key, value in data.metadata.items()
        ])

        if not input_message.strip():
            raise ValueError("Input message cannot be empty")

        response = self._client.chat.completions.create(
            **self._config,
            messages=self._messages + [{"role": "user",
                                        "content": input_message}],
        )

        message = response.choices[0].message
        output_type = output_type or self.output_type

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            tool_call = tool_calls[0]
            arguments_json = tool_call.function.arguments
            try:
                arguments = json.loads(arguments_json)
            except Exception as e:
                raise ValueError(f"Failed to parse tool arguments: {e}")

            if output_type == 'str':
                return '\n---\n'.join([
                    f"# {title}\n\n{content}\n"
                    for title, content in arguments.items()
                ])
            else:
                return DataPacket(content=None,
                                  metadata=arguments,)
        else:
            content = message.content
            if output_type == 'str':
                return content
            else:
                return DataPacket(content=content)

    def __call__(self, data: str | DataPacket,
                 output_type: str = None) -> DataPacket | str:
        return self._ask(data, output_type=output_type)


class AIModelFactory:
    '''创建同步的AIModel'''
    def __init__(self, base_url: str, api_key: str,
                 model: str):
        self.config = {
            'base_url': base_url,
            'api_key': api_key,
            'model': model,
        }

    def __call__(self, system_prompt: str,
                 temperature: float = 0.8,
                 tools: List[Dict[str, Any]] = None,
                 tool_choice: str = 'auto',
                 output_type: Literal['default', 'str'] = 'default',
                 few_shot_messages: List[Dict[str, str]] = None,
                 ):
        return AIModel(
            **self.config,
            system_prompt=system_prompt,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            output_type=output_type,
            few_shot_messages=few_shot_messages,
        )


class CallableABCMeta(ABCMeta):
    def __call__(cls, data):
        instance = super().__call__()
        return instance(data)


class WorkFlow(ABC, metaclass=CallableABCMeta):
    '''自定义工作流元类'''
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, data):
        pass

    def post_execution(self, data):
        pass

    def __call__(self, data):
        data = self.forward(data)
        self.post_execution(data)
        return data


class DataProcessor:
    '''处理datapacket的类

    输入的callback函数需要接收DataPacket并输出DataPacket
    '''
    memory = {}

    def __init__(self,
                 callback: Union[Callable, Sequence[Callable]] = None,
                 mode: Literal['print', 'noop'] = None,
                 name: str = None):
        self.callbacks = []

        match mode:
            case 'print':
                self.callbacks.append(self._print)
            case 'noop':
                self.callbacks.append(self._noop)
            case _:
                pass

        if name and not callback:
            if name not in DataProcessor.memory:
                raise ValueError(f"Unreachable key '{name}'")
            else:
                stored = DataProcessor.memory.get(name)
                if isinstance(stored, list):
                    self.callbacks.extend(stored)
                else:
                    self.callbacks.append(stored)

        if callback:
            if isinstance(callback, Sequence):
                self.callbacks.extend(callback)
            else:
                self.callbacks.append(callback)

            if name:
                DataProcessor.memory[name] = self.callbacks.copy()

        if not self.callbacks:
            raise TypeError("No valid callback provided or initialized.")

    @_strf_datapacket
    def _print(self, data: DataPacket):
        if data.content:
            print(data.content)
        else:
            print("\n".join(data.metadata.values()))
        return data

    def _noop(self, data):
        return data

    @_safe_datapacket
    def __call__(self, data: DataPacket):
        for cb in self.callbacks:
            data = cb(data)
        return data

    def __rshift__(self, other: Callable):
        self.callbacks.append(other)
        return self


class SequentialBlock:
    '''连续AI模块'''
    def __init__(
        self,
        *AIModels: (
            AIModel
            | DataProcessor
            | SequentialBlock
            | ParallelBlock
        )
    ):
        self.AIModels = AIModels

    @_safe_datapacket
    def __call__(self, data: DataPacket):
        for ai in self.AIModels:
            data = ai(data)
        return data


class ParallelBlock:
    """并行 AI 模块"""

    def __init__(self,
                 **models: (
                     AIModel
                     | DataProcessor
                     | SequentialBlock
                     | ParallelBlock
                 )):
        self.models = models

    @_safe_datapacket
    def __call__(self, data: DataPacket):
        inputs = {}

        if data.metadata:
            inputs = {
                key: data.metadata[key]
                for key in self.models
                if key in data.metadata
            }

        if not inputs and data.content:
            inputs = {
                key: data.content
                for key in self.models
            }

        if not inputs:
            raise ValueError("ParallelBlock 输入错误：未提供有效输入。")

        outputs = {
            key: (
                response := self.models[key](input_value)
            ).content or "\n---\n".join([
                f"## {key}\n\n{value}"
                for key, value in response.metadata.items()
            ])
            for key, input_value in inputs.items()
        }

        return DataPacket(content=None, metadata=outputs)


class SequenceProcessor:
    '''平行处理序列'''
    def __init__(self, *callbacks: Callable):
        self.callback = lambda x: reduce(lambda acc, f: f(acc), callbacks, x)

    @_sequencef_datapacket
    def __call__(self, sequence: Iterable = None) -> List[Any]:
        return DataPacket(content=[
            self.callback(task) for task in sequence
        ])


class MultiThreadsSequenceProcessor:
    '''多线程处理任务序列'''
    def __init__(self, *callbacks: Callable, max_workers: int = 5):
        self.callback = lambda x: reduce(lambda acc, f: f(acc), callbacks, x)
        self.max_workers = max_workers

    @_sequencef_datapacket
    def __call__(self, sequence: Iterable):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return DataPacket(
                content=list(executor.map(self.callback, sequence))
            )


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
    '''用于执行特定功能的Model对象'''
    def __init__(self, name: str, role: str, model: str, client: OpenAI,
                 system_prompt: str = None, temperature: float = 0.8):
        self.name = name
        self.role = role
        self.model = model
        self.client = client
        self.memory = [{"role": "system", "content": system_prompt or f"你是{self.role}"}] # NOQA
        if not (0.0 <= temperature <= 2.0):
            warnings.warn("Temperature limit exceeded, setting to default.")
            self.temperature = 0.8
        else:
            self.temperature = temperature

    @_safe_datapacket
    def call(self, data: DataPacket, save_message: bool = True) -> DataPacket:
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


class SuperAgent:
    def __init__(self, planner: Callable[[DataPacket], List[str]] = None,
                 planner_mode: Literal['llm', 'literal'] = 'literal',
                 aggregator: Callable[[Dict[str, DataPacket]], str | dict] = None, # NOQA
                 aggregator_mode: Literal['md', 'default', 'noop'] = 'default',
                 client: OpenAI = None, model: str = None):
        self.planner = planner or self._default_planner(planner_mode)
        self.aggregator = aggregator or self._default_aggregator(aggregator_mode) # NOQA
        self.agents: Dict[str, Model] = {}
        self.client = client
        self.model = model
        self.tool = self._build_tool()

    def register(self, name: str, agent: Model) -> None:
        if name in self.agents and not self.agents[name] is agent:
            raise KeyError(f"Name {name} is already registered to {agent}")
        self.agents[name] = agent
        self.tool.add_param(
            name=name,
            description=f"这是一个功能为{agent.role}的工具",
            param_type='boolean'
        )

    @_safe_datapacket
    def run(self, data: DataPacket):
        agent_names = self.planner(data)
        results = {
            name: self.agents[name].call(data)
            for name in agent_names
        }
        aggregated_result = self.aggregator(results)
        if isinstance(aggregated_result, str):
            return DataPacket(content=aggregated_result)
        elif isinstance(aggregated_result, dict):
            return aggregated_result

    def __str__(self) -> str:
        desc = '<SuperAgent object>'
        return desc + ''.join([f"\n    {name}: {agent.role}"]
                              for name, agent in self.agents.items())

    def __call__(self, data: str | DataPacket):
        return self.run(data)

    def _planner_llm(self, data: DataPacket) -> List[str]:
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
            tools=self.tool.build(),
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

    def _planner_literal(self, data: DataPacket) -> List[str]:
        return [name
                for name in self.agents
                if name.lower() in data.to_str().lower()]

    def _aggregator_md(self, results: Dict[str, DataPacket]) -> str:
        return "\n\n---\n\n".join([
            f"## {name}\n\n{data.to_str()}"
            for name, data in results.items()
        ])

    def _aggregator_default(self, results: Dict[str, DataPacket]) -> str:
        return "\n\n".join([
            f"{name}:\n{data.to_str()}"
            for name, data in results.items()
        ])

    def _aggregator_noop(self, results: Dict[str, DataPacket]) -> Dict[str, str]: # NOQA
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
        if not getattr(self, 'tool'):
            self.tool = ToolBuilder("planner_choser",
                                    "请根据给你的内容或任务选择合适的工具，可以选择多个工具，"
                                    "如果没有合适的工具则请在'no_tool_available'处返回True, "
                                    "并在其余工具处返回False")
            self.tool.add_param(
                name="no_tool_available",
                description="是否所有给你的工具都不合适处理这个问题",
                param_type='boolean'
            )
