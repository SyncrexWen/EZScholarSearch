# API Reference

## safe_dataclass

装饰器 safe_dataclass(cls)

### Description

一个用于自动过滤多余字段的装饰器。常用于解析过程中传入字段可能多于 dataclass 定义字段的情况。它将自动忽略那些未在类定义中出现的字段，避免抛出错误。

### Example

```python
@safe_dataclass
@dataclass
class Example:
    a: int

e = Example(a=1, b=2)  # 不会报错，字段b会被忽略
```

---

## BibMeta

```python
@dataclass
class BibMeta:
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
```

### Description

封装文献 BibTeX 信息的结构体，适用于从 Bib 数据或 Google Scholar 获取的元信息。

---

## PubMeta

```python
@dataclass
class PubMeta:
    author_id: Optional[str] = None
    bib: Optional["BibMeta"] = field(default_factory=dict)
    citedby: Optional[int] = None
    citedby_url: Optional[str] = None
    pub_url: Optional[str] = None
    eprint_url: Optional[str] = None
    author_pub_id: Optional[str] = None
    num_versions: Optional[int] = None
    versions: Optional[List[dict]] = field(default_factory=list)
```

### Description

封装文献出版物的元信息，常用于 scholarly.fill(pub) 的结果封装。

### Notes
	•	若 bib 为字典，会自动转换为 BibMeta 对象。

---

## PaperSection

```python
@dataclass
class PaperSection:
    heading: str
    content: str
```

### Description

表示文献中的一个章节或小节。

### Methods

to_dict() -> dict

返回该章节的字典表示。

flatten() -> str

以 Markdown 格式返回该章节，包含标题和正文。

---

## Paper

```python
@dataclass
class Paper:
    title: Optional[str]
    authors: List[str]
    abstract: Optional[str]
    sections: List[PaperSection]
```

### Description

封装一篇完整文献的结构，包含标题、作者、摘要和各个章节内容。

### Methods

to_dict() -> dict

返回标准字典格式，包括所有字段和章节。

to_flatten_dict() -> dict

将章节展开为扁平化的 Markdown 字符串，并以字典形式返回。

flatten() -> str

以完整 Markdown 字符串的形式返回文献内容，包含所有字段和章节内容，便于展示和写入 .md 文件。

### Notes
	•	authors 会自动转换为列表。
	•	sections 中的元素自动转换为 PaperSection 实例。
