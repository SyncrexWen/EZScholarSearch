from .datastructs import Messages
from .utils import FILE_CONFIG, DynamicLogger

from openai import AsyncOpenAI, OpenAI
from typing import (Tuple, Dict, Any, Union,
                    Iterable, Callable, Type,
                    Optional, List)
from functools import wraps
from time import sleep, perf_counter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import openai
import inspect
import sys
import threading
import warnings
import asyncio
import json


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


@dataclass
class DataPacket:
    '''信息交互协议

    Args:
        content: 数据内容
        metadata: 数据标记，用于传入的function calling，定义数据的结构
    '''
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_type: type = str):
        '''检测content类型是否如预期'''
        if expected_type and not isinstance(self.content, expected_type):
            return False
        return True

    def to_str(self):
        return str(self.content)


class AIModelFactory:
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
                 function_call: str = "auto",) -> "AIModel":
        '''创建AIModel实例

        Args:
            prompt: 传入的system prompt
            temperature: AI回答的随机性
            function_call: AI输出的
        '''
        return AIModel(
            prompt=prompt,
            temperature=temperature,
            **self.config,
            functions=functions,
            function_call=function_call,
        )


class AIModel:
    '''特定功能的AI模型实例，支持system prompt和cuntion calling'''
    def __init__(self, base_url: str, api_key: str,
                 model: str, prompt: str,
                 temperature: float = 0.8,
                 functions: List[Dict[str, Any]] = [],
                 function_call: str = 'auto'):
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

    @retry()
    async def _ask(self, data: Union[DataPacket, str]) -> DataPacket:
        '''根据输入的data调用AI生成回复'''
        if isinstance(data, str):
            input_messages = {
                "role": "user",
                "content": data,
            }
        elif isinstance(data, DataPacket):
            if data.validate(str):
                input_messages = {
                    "role": "user",
                    "content": data.content
                }
            else:
                input_messages = {
                    "role": "user",
                    "content": "\n".join(
                        f"{key}: {value}"
                        for key, value in data.metadata.items())
                }
        else:
            raise TypeError(f"Invalid data type {type(data)}")
        if not input_messages["content"].strip():
            return DataPacket(None)
        response = await self.client.chat.completions.create(
            messages=self.messages + [input_messages],
            **self.config,
        )
        message = response.choices[0].message

        if "function_call" in message:
            arguments = json.loads(message["function_call"]["arguments"])
            return DataPacket(
                content=None,
                metadata=arguments
            )
        else:
            return DataPacket(
                content=message.content,
            )

    async def __call__(self, *datas: Union[DataPacket, str]):
        tasks = [self._ask(data) for data in datas]
        results = await asyncio.gather(tasks)
        return results[0] if len(results) == 1 else results


class DataProcesser:
    '''数据处理器

    在SequentialBlock和ParallelBlock中处理信息
    '''
    def __init__(self, callback: Callable):
        '''
        Args:
            callback: 用于处理数据的函数，需要输入DataPacket，输出DataPacket
        '''
        self.callback = callback

    async def _process(self, data: DataPacket) -> DataPacket:
        if inspect.isasyncgenfunction(self.callback):
            return await self.callback(data)
        else:
            return await asyncio.to_thread(self.callback, data)

    async def __call__(self, *datas):
        tasks = [self.callback(data) for data in datas]
        results = asyncio.gather(tasks)
        return results[0] if len(results) == 1 else results


class WorkFlow(ABC):
    '''定义工作流抽象类'''
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    async def forward(self, data: str):
        pass

    async def __call__(self, data: str) -> str:
        return await self.forward(data)


class SequentialBlock:
    '''连续工作流板块'''
    def __init__(self, *AIModels: Union[
        "AIModel", "SequentialBlock", "ParallelBlock", "DataProcesser"
    ]):
        self.AIModels = AIModels

    async def __call__(self, data: DataPacket):
        new_data = data
        for aimodel in self.AIModels:
            new_data = await aimodel(new_data)
        return new_data


class ParallelBlock():
    '''平行工作流板块'''
    def __init__(self,
                 input_mapper: Optional[
                     Callable[[DataPacket], Dict[str, DataPacket]]
                     ] = None,
                 **models: Union[
                     "AIModel", "SequentialBlock",
                     "ParallelBlock", "DataProcesser"
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
                      "model_outputs": list(self.kw_models.keys())}
        )
