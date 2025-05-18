from typing import List, Dict, Tuple, Union, Any, Optional
from datetime import datetime
from copy import deepcopy
from warnings import warn
from dataclasses import dataclass, field, fields


import json
import os

__all__ = [
    'Messages',
    'MessagesMemory',
    'PubMeta',
    'Paper',
]


class Messages:
    '''存储AI交互信息'''
    def __init__(self, messages: Any = None,
                 saving_path: str = None, prefix: str = None,
                 copy: bool = False):
        '''初始化实例

        Args:
            messages: 可选，传入字典或字典列表以初始化
            saving_path: 可选，默认的存储文件夹
            prefix: 可选，存储文件名的前缀
            copy: 传入的messages是否进行深拷贝

        Returns:

        Raises:

        '''
        if not messages:
            messages = []
        elif isinstance(messages, dict):
            messages = [messages]
        else:
            if copy:
                messages = deepcopy(messages)
        self._messages = messages
        if not saving_path:
            saving_path = "Messages Cache"
        self._saving_path = saving_path
        if not os.path.exists(self.saving_path):
            os.makedirs(saving_path)
        self.prefix = prefix
        self.system_msgs = []
        self.user_msgs = []
        self.assistant_msgs = []
        self._update_messgaes(self._messages)

    @property
    def messages(self):
        '''方法化属性_messages'''
        return self._messages

    @messages.setter
    def messages(self, new_msg: List[Dict[str, str]]):
        '''设置messages实际上为添加新的messages'''
        self._messages.append(new_msg)
        self._update_messgaes(new_msg)

    @messages.deleter
    def messages(self):
        '''删除messages实际上为清空操作'''
        self._messages = []
        self.assistant_msgs.clear()
        self.system_msgs.clear()
        self.user_msgs.clear()

    @property
    def saving_path(self):
        '''方法化属性saving_path'''
        return self._saving_path

    @saving_path.setter
    def saving_path(self, newpath: str):
        '''重置saving_path时检测创建文件夹'''
        self._saving_path = newpath
        os.makedirs(newpath, exist_ok=True)

    def save_to_file(self, path: str = None, *, name: str = None) -> None:
        '''存储messages到json文件

        Args:
            path: 可选，存储到的路径
            name: 可选，存储文件的名称，优先使用path

        Rerurns:

        Raises:

        '''
        if not path:
            if not name:
                name = (
                    self.prefix
                    + datetime.now().strftime(r"%H-%M %Y-%m-%d")
                    + 'json'
                )
            else:
                if not name.endswith(".json"):
                    name += ".json"
            path = os.path.join(self.saving_path, name)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.messages, file, ensure_ascii=False, indent=4)

    def load_from_file(self, path: str = None, *,
                       name: str = None) -> "Messages":
        '''从json文件中读取messages消息

        Args:
            path: 可选，文件路径
            name: 文件名称，可选

        Returns:
            Messages对象

        Raises:
            TypeError: path和name均未传入
        '''
        if not path:
            if not name:
                raise TypeError(
                    f"{self.load_from_file.__name__}"
                    "missing at least 1 argument: "
                    "'path' or 'name'."
                )
            if not name.endswith(".json"):
                name += ".json"
            if self.prefix and not name.startswith(self.prefix):
                name = self.prefix + name
            path = os.path.join(self.saving_path, name)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self._messages = data
        self._update_messgaes(data)
        return self

    def update_path(self, new_path: str) -> None:
        '''更新路径，实质使用@saving_path.setter'''
        self.saving_path = new_path

    def add(self, msg: Union[Dict[str, str], "Messages"], /) -> None:
        '''添加消息

        Args:
            msg: 字典、列表或Messages对象

        Returns:
            None

        Raises:
            TypeError: 输入不支持的数据类型
        '''
        if isinstance(msg, Messages):
            self._messages.append(deepcopy(msg.messages))
            self._update_messgaes(msg.messages)
        elif isinstance(msg, list):
            self._messages += msg
            self._update_messgaes(msg)
        elif isinstance(msg, Dict):
            self._messages.append(msg)
            self._update_messgaes([msg])
        else:
            raise TypeError("Parameters must be "
                            "Messages, list of dict or dict.")
        return None

    def pop(self, idx: int = -1, /) -> Dict[str, str]:
        '''弹出对应消息并从消息列表中删除

        Args:
            idx: 弹出的索引位置，默认为-1

        Returns:
            弹出的元素

        Raises:

        '''
        message = self._messages.pop(idx)
        match message['role']:
            case 'system':
                self.system_msgs.remove(message)
            case 'user':
                self.user_msgs.remove(message)
            case 'assistant':
                self.assistant_msgs.remove(message)
        return message

    def get(self, idx: int, /, type: Union[str, None] = None) -> Dict:
        '''获取消息

        Args:
            idx: 消息的对应索引
            type: 可选，应为None, 'system', 'user' 或 'assistant'，获取指定类型的信息

        Returns:
            取得的消息

        Raises:
            IndexError: 索引无效
            ValueError: 传入的数据种类无效
        '''
        def my_get(lst: List[Dict], index: int = idx) -> Dict:
            if not (-len(lst) <= index < len(lst)):
                raise IndexError("Index out of range")
            return lst[index]
        match type:
            case None:
                return my_get(self._messages)
            case 'system':
                return my_get(self.system_msgs)
            case 'user':
                return my_get(self.user_msgs)
            case 'assistant':
                return my_get(self.assistant_msgs)
            case _:
                return ValueError(f"Key Value {type} not exist")

    def clear(self) -> "Messages":
        '''清空messages，实际使用messages.deleter'''
        del self.messages
        self.assistant_msgs.clear()
        self.system_msgs.clear()
        self.user_msgs.clear()
        return self

    def deepcopy(self) -> "Messages":
        '''返回深拷贝的Messages实例'''
        ret_messages = Messages(
            messages=self._messages,
            saving_path=self._saving_path,
            prefix=self.prefix,
            copy=True
        )
        return ret_messages

    def __len__(self):
        '''返回消息数目'''
        return len(self.messages)

    def __add__(self, other) -> "Messages":
        '''重载加法运算符，创建新的Messages对象并返回'''
        if isinstance(other, Messages):
            return Messages(self.messages + other.messages, copy=True)
        elif isinstance(other, list):
            return Messages(self.messages + other)

    def __iadd__(self, other) -> None:
        '''重载+=运算符，实际调用add方法'''
        self.add(other)

    def _update_messgaes(self, new_messages: List[Dict[str, str]]) -> None:
        '''更新消息时将三种消息分类更新'''
        for message in new_messages:
            match message['role']:
                case 'system':
                    self.system_msgs.append(message['content'])
                case 'user':
                    self.user_msgs.append(message["content"])
                case 'assistant':
                    self.assistant_msgs.append(message['content'])
                case _:
                    raise ValueError(f"Invalid Input {message}")
        return None


class MessagesMemory:
    '''与AI聊天的信息存储类

    自动增添按顺序的0基索引，支持通过关键词和索引位置检索消息
    '''
    def __init__(self, saving_path: str = "MessagesMemory Cache",
                 prefix: str = None):
        '''实例初始化

        Args:
            saving_path: 存储文件的文件夹
            prefix: 文件名的前缀

        Returns:

        Raises:

        '''
        self._memory: Dict[str, Messages] = {}
        self._kw: List[str] = []
        self.cnt: int = 0
        self._saving_path = saving_path
        self.prefix = prefix
        os.makedirs(saving_path, exist_ok=True)

    @property
    def index(self):
        '''增添index属性，返回Index列表'''
        return list(range(self.cnt))

    @property
    def kw(self):
        '''增添kw属性，返回keyword列表'''
        warn("You are accessing a private variable, use '.keys()' instead.")
        return self._kw

    @property
    def saving_path(self):
        '''方法化存储路径属性'''
        return self._saving_path

    @saving_path.setter
    def saving_path(self, new_path):
        '''管理存储路径的更改'''
        self._saving_path = new_path
        os.makedirs(new_path, exist_ok=True)

    def add(self, messages: Messages, kw: str, *, copy: bool = False) -> None:
        '''向存储中添加Messages实例

        Args:
            messages: 待添加的Messages实例
            kw: 用于检索该Messages的关键词属性
            copy: 是否对传入的Messages进行深复制

        Returns:

        Raises:
            ValueError: 传入的关键字kw已存在
        '''
        if kw in self._kw:
            raise ValueError(f"{kw} already exists in the memory.")
        self._kw.append(kw)
        if copy:
            self._memory[kw] = messages.deepcopy()
        else:
            self._memory[kw] = messages
        self.cnt += 1

    def insert(self, messages: Messages, kw: str, idx: int) -> None:
        '''在指定的序号前插入Messages实例

        Args:
            messages: 待插入的messages实例
            kw: 该messages实例的关键词
            idx: 指定的索引

        Returns:

        Raises:
            ValueError: 传入的关键词已存在
            IndexError: 传入的索引值无效
        '''
        if kw in self._kw:
            raise ValueError(f"{kw} already exists in the memory.")
        if idx < -1 * self.cnt or idx > self.cnt:
            raise IndexError("Index out of range.")
        self._memory[kw] = messages
        self._kw.insert(idx, kw)
        self.cnt += 1

    def remove(self, kw: str = None, *, idx: int = None) -> None:
        '''根据指定关键词或索引删除Messages条目

        Args:
            kw: 关键词
            idx: 索引

        Returns:

        Raises:
            TypeError: kw和idx都未指定
            ValueError: 传入的kw和idx指向不同元素
            IndexError: 传入的索引无效
            KeyError: 传入的kw不存在
        '''
        kw = self._check_kw_idx_(kw=kw, idx=idx)

        del self._memory[kw]
        self._kw.remove(kw)
        self.cnt -= 1

    def pop(self, idx: int = -1) -> Tuple[Messages, str]:
        '''弹出最后一个Messages条目及关键词

        Args:
            idx: 弹出条目的索引，默认-1

        Returns:
            弹出的Messages对象
            Messages对象的索引

        Raises:
            IndexError: 传入的索引值无效
        '''
        if not (-self.cnt <= idx < self.cnt):
            raise IndexError("Index out of range.")
        kw = self._kw.pop(idx)
        messages = self._memory[kw]
        del self._memory[kw]
        self.cnt -= 1
        return messages, kw

    def get(self, *, kw: str = None, idx: int = None,
            copy: bool = False) -> Messages:
        '''根据关键词或索引获取Messages条目

        Args:
            kw: 关键词
            idx: 索引
            copy: 是否对输出的Messages进行深拷贝

        Returns:
            查询到的Messages对象

        Raises:
            TypeError: kw和idx都未指定
            ValueError: 传入的kw和idx指向不同元素
            IndexError: 传入的索引无效
            KeyError: 传入的kw不存在
        '''
        kw = self._check_kw_idx_(kw=kw, idx=idx)

        if copy:
            return self._memory[kw].deepcopy()
        return self._memory[kw]

    def update(self, new_messages: Messages, *,
               kw: str = None, idx: int = None,
               copy: bool = False) -> None:
        '''使用新条目根据关键词和索引值更新条目

        Args:
            kw: 关键词
            idx: 索引
            copy: 是否对输出的Messages进行深拷贝

        Returns:

        Raises:
            TypeError: kw和idx都未指定
            ValueError: 传入的kw和idx指向不同元素
            IndexError: 传入的索引无效
            KeyError: 传入的kw不存在
        '''
        kw = self._check_kw_idx_(kw=kw, idx=idx)

        if copy:
            new_messages = new_messages.deepcopy()

        self._memory[kw] = new_messages

    def clear(self) -> None:
        '''清除存储'''
        self._memory = {}
        self._kw = []
        self.cnt = 0

    def display(self) -> Dict[Tuple[str, int], Messages]:
        '''展示存储

        生成以(关键词, 索引)的元组为键、Messages为值的字典
        '''
        result = {
            (kw, idx): self._memory[kw]
            for idx, kw in enumerate(self._kw)
        }
        return result

    def save_to_file(self, path: str = None, *, name: str = None) -> None:
        '''将聊天信息存储至json文件

        Args:
            path: 存储的路径
            name: 存储文件的名称，优先使用路径，若未指定name则根据prefix和当前时间自动生成文件名

        Returns:

        Raises:

        '''
        if not path:
            if not name:
                name = self._generate_filename() + ".json"
            elif not name.endswith(".json"):
                name += ".json"
            path = os.path.join(self._saving_path, name)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self._format_json_(), file, ensure_ascii=False, indent=4)

    def load_from_file(self, path: str = None, *,
                       name: str = None) -> "MessagesMemory":
        '''从文件中读取存储数据

        Args:
            path: 文件路径
            name: 文件名

        Returns:
            读取后的MessagesMemory对象

        Raises:
            TypeError: 未传入'path'或'name'参数
        '''
        if not path:
            if not name:
                raise TypeError(f"{self.load_from_file.__name__} "
                                "missing at least 1 argument: "
                                "'path' or 'name'.")
            if not name.endswith(".json"):
                name += '.json'
            if self.prefix and not name.startswith(self.prefix):
                name = self.prefix + name
            path = os.path.join(self.saving_path, name)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.clear()
        self._memory, self._kw, self.cnt = self._decode_json_(data=data)

        return self

    def keys(self) -> Tuple[str]:
        '''返回关键词列表'''
        return self.kw

    def values(self) -> Tuple[Messages]:
        '''返回存储的Messages列表'''
        ret_values = [self._memory[kw] for kw in self._kw]
        return ret_values

    def items(self) -> List[Tuple[str, Messages]]:
        '''返回存储的(关键词, Messages对象)列表'''
        ret_items = [(kw, self._memory[kw]) for kw in self._kw]
        return ret_items

    def __len__(self):
        '''返回存储数目'''
        return self.cnt

    def __getitem__(self, key: int | str):
        '''支持直接使用[]索引'''
        if isinstance(key, int):
            idx = key
            kw = None
        elif isinstance(key, slice):
            kws = self._kw[key]
            ret_lst = [self._memory[kw] for kw in kws]
            return ret_lst
        elif isinstance(key, str):
            idx = None
            kw = key
        else:
            raise TypeError("Indices must be integers, slices or str, "
                            f"not {type(key)}.")
        kw = self._check_kw_idx_(kw=kw, idx=idx)
        return self._memory[kw]

    def _check_kw_idx_(self, kw: str = None, idx: int = None) -> str:
        '''检查关键词与索引值，返回正确的关键词'''
        if kw is None and idx is None:
            raise TypeError(
                f"{self.__class__.__name__}.remove() "
                "missing at least 1 argument: 'kw' or 'idx'."
            )

        if idx is not None:
            if not (-self.cnt <= idx < self.cnt):
                raise IndexError("Index out of range.")
            kw_by_idx = self._kw[idx]
            if kw is not None and kw != kw_by_idx:
                raise ValueError("Given 'kw' and 'idx' "
                                 "point to different elements.")
            kw = kw_by_idx

        if kw not in self._memory:
            raise KeyError(f"Key '{kw}' not found.")

        return kw

    def _generate_filename(self) -> str:
        '''根据prefix和当前时间生成文件名'''
        name = self.prefix + datetime.now().strftime(r"%H-%M %Y-%m-%d")
        return name

    def _format_json_(self) -> Dict[str, Dict[str, str | int | Messages]]:
        '''格式化转化json文件'''
        result = {
            kw: {
                "keyword": kw,
                "index": idx,
                "messages": self._memory[kw].messages
            }
            for idx, kw in enumerate(self._kw)
        }
        return result

    def _decode_json_(
        self, data: Dict
    ) -> Tuple[Dict[str, Messages], List[int], int]:
        '''解析json文件'''
        _memory = {}
        _kw = []
        for kw, item in data.items():
            _kw.append(kw)
            messages = item['messages']
            Msg = Messages(messages=messages, copy=True)
            _memory[kw] = Msg
        return _memory, _kw, len(_kw)


def safe_dataclass(cls):
    '''自动过滤多余字段的装饰器'''
    orig_init = cls.__init__
    cls_fields = {f.name for f in fields(cls)}

    def new_init(self, *args, **kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if k in cls_fields}
        orig_init(self, *args, **clean_kwargs)

    cls.__init__ = new_init
    return cls


@safe_dataclass
@dataclass
class BibMeta:
    '''解析Bib的数据类型'''
    title: Optional[str] = None
    author: Optional[str] = None
    pub_year: Optional[str] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None


@safe_dataclass
@dataclass
class PubMeta:
    '''解析经过scholarly.fill的pub的数据类型'''
    author_id: Optional[str] = None
    bib: Optional["BibMeta"] = field(default_factory=dict)
    citedby: Optional[int] = None
    citedby_url: Optional[str] = None
    pub_url: Optional[str] = None
    eprint_url: Optional[str] = None
    author_pub_id: Optional[str] = None
    num_versions: Optional[int] = None
    versions: Optional[List[dict]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.bib, dict):
            self.bib = BibMeta(**self.bib)


@safe_dataclass
@dataclass
class PaperSection:
    heading: str
    content: str

    def to_dict(self):
        return {"heading": self.heading, "content": self.content}

    def flatten(self):
        return f"##{self.heading}\n\n{self.content}"


@safe_dataclass
@dataclass
class Paper:
    title: Optional[str]
    authors: List[str]
    abstract: Optional[str]
    sections: List[PaperSection]

    def __post_init__(self):
        if isinstance(self.authors, str):
            self.authors = [self.authors]
        elif self.authors is None:
            self.authors = []
        self.sections = [
            section if isinstance(section, PaperSection)
            else PaperSection(**section)
            for section in self.sections
        ]

    def to_dict(self):
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'sections': [section.to_dict() for section in self.sections]
        }

    def to_flatten_dict(self):
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            **{
                f"sections {i}": self.sections[i].flatten()
                for i in range(len(self.sections))
            }
        }

    def flatten(self):
        return "\n---\n".join([
            f"#{key}\n\n{value}"
            for key, value in self.to_flatten_dict().items()
        ])
