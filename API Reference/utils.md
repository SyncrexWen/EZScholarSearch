## utils Module

模块 `utils` 旨在提供辅助功能，帮助用户完成特定的任务。这里主要提供两方面的辅助功能：

+ `PDFParser` 解析pdf的信息，得到摘要、结论等的内容
+ `ToolBuilder & ToolRegistry` 快速构架 `Tools Call` 中的Tools

### utils.PubDownloader

文献pdf下载器

**属性**

*UNPAYWALL_EMAIL*: unpaywall邮箱

*UNPAYWALL_API*: unpaywall的api

*DEFAULT_DIR*: 默认的保存目录

**方法**

*config_api_email*: 配置unpaywall api 邮箱

*config_saving_dir*: 配置保存路径

*download*: 下载pdf文献
```python
def download(pub: PubMeta, saving_dir: str = DEFAULT_DIR) -> str | None:
```

### utils.BasePDFParser

基础的用于解析pdf文件的类，利用`fitz`来实现对pdf的操作，主要为提取pdf的abstract，conclusion以及pdf中的图片

**方法**

*extract_abstract*: 提取摘要部分
```python
def extract_abstract(pdf_path_or_url) -> str:
```

*extract_conclusion*: 提取结论部分
```python
def extract_conclusion(pdf_path_or_url) -> str:
```

*extract_images*: 提取文件中的图片，并保存到指定位置
```python
def extract_images(pdf_path_or_url, saving_dir="extracted_images"):
```

### utils.AdvancedPDFParser

进阶的用于解析pdf文件的类，可以从`pdfplumber`、`pymupdf`、`fitz`三种方式中选择一种合适的来解析pdf，同时实现extract abstract | conclusion | images 的操作

**方法**

*extract_abstract*: 提取摘要部分
```python
def extract_abstract(pdf_path_or_url) -> str:
```

*extract_conclusion*: 提取结论部分
```python
def extract_conclusion(pdf_path_or_url) -> str:
```

*extract_images*: 提取文件中的图片，并保存到指定位置
```python
def extract_images(pdf_path_or_url, saving_dir="extracted_images"):
```

### utils.GrobidPDFParser

使用Grobid方式来解析pdf，使用之前请确保Grobid已经被部署再本地（Windows系统已经不支持使用这种方法，但是macOS以及Linux仍支持Grobid）

```python
class GrobidPDFParser:
    '''使用 Grobid 解析 PDF 文档的封装类'''
```

**方法**

*config_local_port*: 配置 Grobid 服务的本地端口

```python
@staticmethod
def config_local_port(port: int) -> None
```
+ `port (int): Grobid` 服务运行的端口号

*parse_pdf*: 解析 PDF 文件/URL 并返回结构化论文对象

```python
@staticmethod
def parse_pdf(
    pdf_path_or_url: str,
    clean_tmp_file: bool = True,
    default_port: int = 8070
) -> Paper
```

+ `pdf_path_or_url (str):` 本地 PDF 文件路径或 PDF 文件的 URL
+ `clean_tmp_file (bool): `当解析 URL 时是否删除临时下载文件 (默认: True)
+ `default_port (int):`备用端口号 (默认: 8070)

### utils.ToolBuilder and utils.ToolRegistry

ToolBuilder和ToolRegistry两个部分用于快速构建Tools Call中的Tool

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
        ],
```

**方法**

(`ToolBuider`)

*add_param*: 添加Tool中的参数部分

*build*: 返回一个Tool的字典

*from_dict*: 直接从一个字典得到所需要构建的Tool

(`ToolRegister`)

*register*: 注册一个工具生成器并生成工具实例

*build_all*: 批量获取已注册的工具实例

```python
tool = ToolBuilder(
    name="",
    description=""
)

tool.add_param(
    name="",
    description="",
    type="",
    required=""
)

tool.build()
```

```python
toolRegistry = ToolRegistry()

toolRegistry.register(
    tool,
    name=""
)

toolRegistry.register(
    another_tool,
    name=""
)

toolRegistry.build_all()
```

使用示例：

```python
def test_extract_abstract(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    abstract = utils.BasePDFParser.extract_abstract(Test_PDF_Path)

    (output_dir / "abstract").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abstract/sample1.txt", "w", encoding='utf-8') as f:
        f.write(abstract)

def test_extract_conclusion(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    conclusion = utils.BasePDFParser.extract_conclusion(Test_PDF_Path)

    (output_dir / "conclusion").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conclusion/sample1.txt", "w", encoding='utf-8') as f:
        f.write(conclusion)

def test_extract_images(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    img_count = utils.BasePDFParser.extract_images(Test_PDF_Path, output_dir / "images")

    saved_images = list((output_dir / "images").glob("*"))

```

最终我们在output_dir下得到三个文件

+ abstract
+ conclusion
+ images

示例：
```md
(PDFParser.md)
---
Abstract:
We describe observations and physical characteristics of Earth-crossing asteroid 2024
    YR4, discovered on 2024 December 27 by the Asteroid Terrestrial-impact Last Alert
    System. The asteroid has semi-major axis, a = 2.52 au, eccentricity, e = 0.66, inclination
    i = 3.41◦, and a ∼0.003 au Earth minimum orbit intersection distance. We obtained
    g, r, i, and Z imaging with the Gemini South/Gemini Multi-Object Spectrograph on
    2025 February 7.
    We measured a g-i spectral slope of 13±3 %/100 nm, and color
    indices g-r = 0.70±0.10, r-i = 0.25±0.06, i-Z = -0.27±0.10. 2024 YR4 has a spectrum
    that best matches R-type and Sa-type asteroids and a diameter of ∼30-65 m using our
    measured absolute magnitude of 23.9±0.3 mag, and assuming an albedo of 0.15-0.4.
    The lightcurve of 2024 YR4 shows ∼0.4 mag variations with a rotation period of ∼1170
    s. We use photometry of 2024 YR4 from Gemini and other sources taken between 2024
    December to 2025 February to determine the asteroid’s spin vector and shape, finding
    that it has an oblate, ∼3:1 a:c axial ratio and a pole direction of λ, β = ∼42◦, ∼-25◦.
    Finally, we compare the orbital elements of 2024 YR4 with the NEO population model
    and find that its most likely sources are resonances between the inner and central Main
    Belt.
    Keywords: minor planets, asteroids: individual (2024 YR4), temporarily captured or-
    biters, minimoons
    ∗These authors contributed equally to this work.
    arXiv:2503.05694v1  [astro-ph.EP]  7 Mar 2025

---
Conclusion:
The properties of 2024 YR4 are similar to other small NEOs, that it has a S-complex composition,
    most likely originated from a resonances located inner Main Belt, and likely has moderate reflective
    properties implying a size of ∼42 m. The exact origin within the Main Belt seems less clear, though
    a clue may lie in its spin-vector orientation. As recent asteroid lightcurve spin-vector studies have
    shown, the spin direction of an asteroid can affect the way an asteroid’s orbit evolves due to the
    thermal recoil Yarkovsky effect, with prograde asteroids drifting outward, and retrograde asteroids
    drifting inwards (Bolin et al. 2018a; Hanuˇs et al. 2018b; Athanasopoulos et al. 2022). Since the spin
    direction of 2024 YR4 is retrograde with its -39◦ecliptic latitude, an origin from the inner Main
    Discovery and characterization of 2024 YR4
    13
    Figure 9. Lightcurve fit of 2024 YR4, showing dense photometric observations from VLT and Gemini in
    the r and g filters (top panels), and sparse datasets from ATLAS, Catalina Sky Survey, and Pan-STARRS
    (bottom three panels). The best-fit model is overlaid.
    14
    Bolin et al.
    Figure 10. Convex shape model of 2024 YR4 in three orthogonal views obtained using publicly available
    photometry of 2024 YR4 from the Minor Planet Center archive, VLT archive and Gemini. The X vs. Z and
    Y vs. Z views show the edge on view of the asteroid with the spin axis pointing in the Z direction (ecliptic
    longitude and latitude equal to 42◦and -25 degrees◦). The asteroid is viewed from a pole-on view in the X
    vs. Y view.
    Belt since its retrograde spin would cause it to drift inwards away from the 3:1 MMR at 2.5 au,
    its most likely escape path into the NEO population. Therefore, its suspected retrograde spin may
    imply that it original location was the central Main Belt, located at 2.52 au to 2.82 au from the Sun
    (Nesvorn´y et al. 2015). This would take 2024 YR4 ’s origins away from the inner Main Belt where
    S-types dominate to the central Main Belt where C-types are more plentiful seemingly in tension
    with its taxonomic type (DeMeo & Carry 2014). However, several S-type families are located near
    the 3:1 MMR in the central Main Belt, such as the Rafita family located at 2.58 au, which could be
    a candidate source family for 2024 YR4.
```