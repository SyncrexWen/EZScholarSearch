# flake8: noqa: E501

from __future__ import annotations

from .datastructs import Paper, PubMeta

from scholarly import scholarly
from typing import Dict, List, Union, Optional, Any, Literal
from copy import deepcopy
from urllib.parse import urlparse
from pathlib import Path
from io import BytesIO
from PIL import Image
from datetime import datetime
from bs4 import BeautifulSoup
from warnings import warn

import logging
import logging.config
import importlib
import asyncio
import os
import re
import requests
import fitz
import pdfplumber
import urllib


__all__ = [
    # 'DEFAULT_CONFIG',
    # 'FILE_CONFIG',
    # 'get_default_log_config',
    # 'get_file_log_config',
    # 'LoggingConfigurator',
    # 'DynamicLogger'
    # 'AsyncScholarly',
    'PubDownloader',
    'BasePDFParser',
    'AdvancedPDFParser',
    'GrobidPDFParser',
    'ToolBuilder',
    'ToolRegistry',
]

# 默认的日志器配置，记录到控制台和文件
DEFAULT_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": r"%H-%M-%S %Y-%m-%d",
        },
        "detailed": {
                "format": "%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: "
                "%(message)s",
                "datefmt": r"%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "app.log",
            "maxBytes": 1_048_576,
            "backupCount": 3,
            "encoding": "utf8",
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    }
}


# 记录到文件的日志器配置
FILE_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": r"%H-%M-%S %Y-%m-%d",
        },
        "detailed": {
            "format": "%(asctime)s [%(name)s:%(lineno)d] "
            "%(levelname)s: %(message)s",
            "datefmt": r"%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "app.log",
            "max_bytes": 1_048_576,
            "backupCount": 3,
            "encoding": "utf8",
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file"],
    }
}


def get_default_log_config():
    '''取得默认设置字典的副本'''
    return deepcopy(DEFAULT_CONFIG)


def get_file_log_config():
    '''取得文件日志配置器字典的副本'''
    return deepcopy(FILE_CONFIG)


class LoggingConfigurator:
    '''配置日志器的工具类'''
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        '''从字典中读取配置'''

        formatters = config.get("formatters", {})
        for name, fmt_config in formatters.items():
            logging.config.dictConfig({
                "version": 1,
                "formatters": {name: fmt_config}
            })

        handlers = config.get("handlers", {})
        for name, handler_config in handlers.items():
            class_path = handler_config.pop("class")
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            handler_class = getattr(module, class_name)

            for key, value in handler_config.items():
                if isinstance(value, str) and value.startswith("ext://"):
                    handler_config[key] = importlib.import_module(value[6:])

            handler = handler_class(**handler_config)
            if "formatter" in handler_config:
                formatter = logging.Formatter(
                    **formatters[handler_config["formatter"]]
                )
                handler.setFormatter(formatter)
            logging.getLogger(name).addHandler(handler)

        for logger_name, logger_config in config.get("loggers", {}).items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logger_config.get("level", "DEBUG"))
            for handler_name in logger_config.get("handlers", []):
                logger.addHandler(logging.getLogger(handler_name).handlers[0])

        root_config = config.get("root", {})
        if root_config:
            logging.getLogger().setLevel(root_config.get("level", "DEBUG"))
            for handler_name in root_config.get("handlers", []):
                logging.getLogger().addHandler(
                    logging.getLogger(handler_name).handlers[0]
                )


class DynamicLogger:
    '''动态日志器'''
    def __init__(self, name: str, initial_config: Optional[Dict] = None):
        '''初始化日志器

        Args:
            name: 日志器名称
            initial_config: 可选，初始化配置，默认为默认配置
        '''
        self.name = name
        self.logger = logging.getLogger(name)
        self.config = initial_config or DEFAULT_CONFIG
        self._configure()

    def _configure(self):
        '''调用工具类LoggingConfigurator进行配置更新'''
        LoggingConfigurator.from_dict(self.config)

    def update_config(self, new_config: Dict[str, Any]):
        '''更新配置'''
        self.config.update(new_config)
        self._configure()

    def add_handler(
        self,
        handler_type: str,
        level: Union[int, str],
        formatter: Optional[str] = None,
        **kwargs
    ):
        '''添加日志记录的处理方式'''
        handler_id = f"{handler_type}_{len(self.config['handlers'])}"
        self.config["handlers"][handler_id] = {
            "class": f"logging.{handler_type}",
            "level": (
                logging.getLevelName(level)
                if isinstance(level, int)
                else level
            ),
            "formatter": formatter or "standard",
            **kwargs
        }
        self._configure()

    def debug(self, msg: str):
        '''记录DEBUG级别信息'''
        self.logger.debug(msg)

    def info(self, msg: str):
        '''记录INFO级别的信息'''
        self.logger.info(msg)

    def warning(self, msg: str):
        '''记录WARNING级别的信息'''
        self.logger.warning(msg)

    def error(self, msg: str):
        '''记录ERROR级别的信息'''
        self.logger.error(msg)

    def critical(self, msg: str):
        '''记录CRITICAL级别的信息'''
        self.logger.critical(msg)


class AsyncScholarly:
    '''异步重写scholarly包'''
    @staticmethod
    async def search_author(author_name: str, *,
                            blurred: bool = False, blurred_num: int = 3
                            ) -> list:
        '''按作者搜索作者条目

        Args:
            author_name: 搜索的名称
            blurred: 可选，是否是模糊搜索
            blurred_num: 模糊搜索的条目数目

        Returns:
            作者条目的列表
        '''
        gen = await asyncio.to_thread(
            lambda: scholarly.search_author(author_name)
        )
        return await AsyncScholarly._collect_from_generator(
                gen, blurred_num if blurred else 1
            )

    @staticmethod
    async def fill_author(author: Dict) -> Dict:
        '''提供作者返回更详细的信息'''
        return await asyncio.to_thread(lambda: scholarly.fill(author))

    @staticmethod
    async def search_pubs(query: str, max_num_results: int = 3) -> list:
        '''根据关键词搜索文献

        Args:
            query: 用于搜索的关键词
            max_num_results: 最大返回条目

        Returns:
            条目信息的列表
        '''
        gen = await asyncio.to_thread(
            lambda: scholarly.search_pubs(query)
        )
        return await AsyncScholarly._collect_from_generator(
            gen, count=max_num_results
        )

    @staticmethod
    async def fill_pub(pub: Dict) -> Dict:
        '''根据文献数据返回更详细的信息'''
        return await asyncio.to_thread(lambda: scholarly.fill(pub))

    @staticmethod
    async def fill_pubs(pubs: List[Dict], max_concurrent: int = 5
                        ) -> List[Dict]:
        '''对列表中的文献搜索详细信息并返回'''
        sem = asyncio.Semaphore(max_concurrent)

        async def sem_fill(pub):
            async with sem:
                return await AsyncScholarly.fill_pub(pub)

        tasks = [sem_fill(pub) for pub in pubs]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _collect_from_generator(gen, count: int = 1) -> list:
        '''从生成器中收集条目返回'''
        def collect():
            results = []
            try:
                for _ in range(count):
                    results.append(next(gen))
            except StopIteration:
                pass
            return results
        return await asyncio.to_thread(collect)


class PubDownloader:
    '''文献pdf下载器

    通过经过fill的pub下载pdf文件
    '''
    UNPAYWALL_EMAIL = None
    UNPAYWALL_API = "https://api.unpaywall.org/v2/{}?email="
    DEFAULT_DIR = "./pubs"

    @staticmethod
    def config_api_email(email: str):
        '''配置unpaywall api 邮箱

        Args:
            email: 邮箱
        '''
        PubDownloader.UNPAYWALL_EMAIL = email

    @staticmethod
    def config_saving_dir(dir: str):
        '''配置保存路径

        Args:
            dir: 路径
        '''
        PubDownloader.DEFAULT_DIR = dir

    @staticmethod
    def download(pub: PubMeta, saving_dir: str = DEFAULT_DIR) -> str | None:
        '''下载pdf文献

        Args:
            pub: 经过scholarly.fill的文献
            saving_dir: 保存的路径
        '''
        os.makedirs(saving_dir, exist_ok=True)
        title = PubDownloader._get_filename(
            pub.bib.title or (
                'untitled ' +
                datetime.now().strftime(r"%h%m%s-%Y%M%d")
            )
        )

        # eprint_url download
        pdf_url = pub.eprint_url
        if pdf_url and PubDownloader._is_pdf_path(pdf_url):
            try:
                return PubDownloader._download_pdf(pdf_url, title, saving_dir)
            except Exception:
                # warn(
                #     f"Fail to download {pub.bib.title} through eprint url"
                # )
                pass

        # arXiv download
        url = pub.pub_url or pub.bib.url
        if url and "arxiv.org" in url:
            pdf_url = PubDownloader._parse_arxiv_pdf_url(url)
            if pdf_url:
                try:
                    return PubDownloader._download_pdf(pdf_url,
                                                       title,
                                                       saving_dir)
                except Exception:
                    # warn(
                    #     f"Fail to download {pub.bib.title} through arXiv"
                    # )
                    pass

        # unpaywall download
        doi = pub.bib.doi
        if doi and PubDownloader.UNPAYWALL_API:
            pdf_url = PubDownloader._paarse_unpaywall_pdf_url(doi)
            if pdf_url:
                try:
                    return PubDownloader._download_pdf(pdf_url,
                                                       title,
                                                       saving_dir)
                except Exception:
                    # warn(
                    #     f"Fail to download {pub.bib.title} through unpaywall"
                    # )
                    pass

        return None

    @staticmethod
    def _get_filename(name: str) -> str:
        '''去除非法符号，获取文件名'''
        return (
            re.sub(r'[<>:"/\\|?*]', name).strip()
            or 'untitled ' + datetime.now().strftime(r"%h%m%s-%Y%M%d")
        )

    @staticmethod
    def _download_pdf(pdf_url: str, title: str, saving_dir: str) -> str | None:
        '''通过给定的pdf_url下载文件

        Args:
            pdf_url: pdf文件的url
            title: 文件保存的名称
            saving_dir: 文件保存的路径

        Returns:
            若下载成功，则返回下载保存的路径
            若下载失败，则返回None
        '''
        try:
            response = requests.get(pdf_url, timeout=15)
            if response.ok and "pdf" in response.headers.get(
                "Content-Type", ""
            ):
                path = os.join(saving_dir, f"{title}.pdf")
                with open(path, 'wb') as file:
                    file.write(response.content)
                return path
            else:
                raise ValueError(f"{pdf_url} is not pdf content")
        finally:
            return None

    @staticmethod
    def _is_pdf_path(url: str) -> bool:
        '''检验url路径是否为pdf路径'''
        parsed = urlparse(url)
        return parsed.path.endswith("pdf")

    @staticmethod
    def _parse_arxiv_pdf_url(url: str) -> str | None:
        '''解析arxiv的pdf路径'''
        pair = re.search(r"arxiv\\.org/(abs|pdf)/([0-9]+\\.[0-9]+)", url)
        if pair:
            arxiv_id = pair.group(2)
            return f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        return None

    @staticmethod
    def _parse_unpaywall_pdf_url(doi: str) -> str | None:
        '''解析unpaywall的pdf路径'''
        try:
            api = PubDownloader.UNPAYWALL_API + PubDownloader.UNPAYWALL_EMAIL
            url = api.format(doi)
            response = requests.get(url, timeout=15)
            if response.ok:
                data = response.json()
                return data.get("best_oa_location", {}).get("url_for_pdf")
        finally:
            return None


class BasePDFParser:
    '''解析pdf文件的类'''
    @staticmethod
    def _load_pdf(pdf_path_or_url):
        """支持本地路径或URL"""
        if (pdf_path_or_url.startswith("http://") or
                pdf_path_or_url.startswith("https://")):
            response = requests.get(pdf_path_or_url)
            response.raise_for_status()
            return fitz.open(stream=response.content, filetype="pdf")
        else:
            return fitz.open(pdf_path_or_url)

    @staticmethod
    def extract_abstract(pdf_path_or_url) -> str:
        """提取摘要部分

        Args:
            pdf_path_or_url: pdf路径或url

        Returns:
            摘要
        """
        abstract_text = ""
        with BasePDFParser._load_pdf(pdf_path_or_url) as doc:
            for page in doc:
                text = page.get_text("text")

                abstract_match = re.search(
                    r'Abstract\s*\n(.+?)(?=\n[A-Z]+\s*\n|\Z)',
                    text,
                    re.DOTALL | re.IGNORECASE
                )
                if abstract_match:
                    abstract_text += abstract_match.group(1).strip()
                    break

            if not abstract_text:
                abs_lst = []
                first_page = doc[0]
                blocks = first_page.get_text('blocks')
                for block in blocks[2:5]:
                    abs_lst.append(block[4])
                abstract_text = '\n'.join(abs_lst).strip()

        return abstract_text

    @staticmethod
    def extract_conclusion(pdf_path_or_url) -> str:
        """提取结论部分

        Args:
            pdf_path_or_url: pdf路径或url

        Returns:
            结论
        """
        conclusion_text = ""
        collecting = False
        with BasePDFParser._load_pdf(pdf_path_or_url) as doc:
            for page in doc:
                text = page.get_text("text")

                conclusion_match = re.search(
                    (
                        r'(Conclusion|Conclusions|Concluding Remarks|Summary)'
                        r'\s*\n'
                        r'(.+?)(?=\n[A-Z]+\s*\n|\Z)'
                    ),
                    text, re.DOTALL | re.IGNORECASE
                )

                if conclusion_match:
                    collecting = True
                    start_index = conclusion_match.start()
                    text = text[start_index:]

                if collecting:
                    acknowledgement_match = re.search(
                    (
                        r'(Acknowledgement|Acknowledgements|References|Bibliography|Acknowledgments|Acknowledgment)'
                        r'\s*\n'
                        r'(.+?)(?=\n[A-Z]+\s*\n|\Z)'
                    ),
                    text, re.DOTALL | re.IGNORECASE
                )
                    if acknowledgement_match:
                        end_index = acknowledgement_match.start()
                        conclusion_text += text[:end_index].strip() + "\n"
                        break
                    else:
                        conclusion_text += text.strip() + "\n"
        conclusion_text = re.sub(r'\n{3,}', '\n\n', conclusion_text)  # 减少空行
        conclusion_text = conclusion_text.strip()
        return conclusion_text

    @staticmethod
    def extract_images(pdf_path_or_url, saving_dir="extracted_images"):
        """提取所有图片并保存到指定文件夹

        Args:
            pdf_path_or_url: pdf路径或url
            saving_dir: 保存图片的路径

        Returns:
            成功保存图片的数目
        """
        doc = BasePDFParser._load_pdf(pdf_path_or_url)
        os.makedirs(saving_dir, exist_ok=True)
        count = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(
                    saving_dir,
                    f"figure_Page{page_index+1}_{img_index+1}.{image_ext}"
                )
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                count += 1
        return count


class AdvancedPDFParser:
    @staticmethod
    def _load_pdf(pdf_path_or_url: str, way: str = 'pdfplumber'):
        """Load a PDF from a local path or URL (http, https, ftp, file)."""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if way not in ('pdfplumber', 'pymupdf', 'fitz'):
            raise ValueError(f"Unsupported backend: {way}")

        parsed_url = urlparse(pdf_path_or_url)
        scheme = parsed_url.scheme.lower()

        if scheme in ('http', 'https'):
            try:
                response = requests.get(
                    pdf_path_or_url,
                    timeout=10,
                    allow_redirects=True,
                )
                response.raise_for_status()
                pdf_stream = BytesIO(response.content)
            except requests.RequestException as e:
                logger.error(f"Failed to download PDF "
                             f"from {pdf_path_or_url}: {e}")
                raise
        elif scheme == 'ftp':
            try:
                with urllib.request.urlopen(pdf_path_or_url) as response:
                    pdf_stream = BytesIO(response.read())
            except urllib.error.URLError as e:
                logger.error("Failed to download FTP PDF"
                             f" from {pdf_path_or_url}: {e}")
                raise
        elif scheme == 'file' or not scheme:
            local_path = (
                parsed_url.path if scheme == 'file' else pdf_path_or_url
            )
            if not Path(local_path).is_file():
                raise FileNotFoundError("Local PDF file "
                                        f"not found: {local_path}")
            if way == 'pdfplumber':
                return pdfplumber.open(local_path)
            elif way in ('pymupdf', 'fitz'):
                return fitz.open(local_path)
        else:
            raise ValueError(f"Unsupported URL scheme: {scheme}")

        try:
            if way == 'pdfplumber':
                return pdfplumber.open(pdf_stream)
            elif way in ('pymupdf', 'fitz'):
                return fitz.open(stream=pdf_stream, filetype='pdf')
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    @staticmethod
    def extract_abstract(pdf_path_or_url: str) -> str | None:
        """从pdf文件中提取摘要

        Args:
            pdf_path_or_url: pdf路径或url

        Returns:
            摘要内容，若提取失败，则返回None
        """
        try:
            with AdvancedPDFParser._load_pdf(
                pdf_path_or_url, way='pdfplumber'
            ) as pdf:
                for page in pdf.pages[:5]:
                    text = page.extract_text()
                    if not text:
                        continue
                    match = re.search(
                        (
                            r'(?i)(?:Abstract|摘要)\s*[\n\r:]*([\s\S]+?)(?=\n\s*(?:[1-9]\.|Introduction|关键词|\b\w+\s*:|\Z))'
                        ),
                        text,
                        re.DOTALL | re.IGNORECASE | re.MULTILINE
                    )
                    if match:
                        return match.group(1).strip()
        except Exception as e:
            logging.error("Error extracting abstract "
                          f"from {pdf_path_or_url}: {e}")
        return None

    @staticmethod
    def extract_conclusion(pdf_path_or_url: str) -> str | None:
        """提取pdf的结论

        Args:
            pdf_path_or_url: pdf路径或url

        Returns:
            结论，提取失败则返回None
        """
        try:
            with AdvancedPDFParser._load_pdf(
                pdf_path_or_url, way='pdfplumber'
            ) as pdf:
                conclusion_text = []
                in_conclusion = False
                for page in pdf.pages[-10:]:
                    text = page.extract_text()
                    if not text:
                        continue
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if re.match(
                            r'(?i)(^\s*\d*\.?\s*)(conclusions?|discussion)\b',
                            line
                        ):
                            in_conclusion = True
                            continue
                        if in_conclusion and re.match(
                            r'(?i)^\s*('
                            r'references|appendix|acknowledgments'
                            r')\s*$',
                            line
                        ):
                            in_conclusion = False
                            break
                        if in_conclusion:
                            conclusion_text.append(line)
                return (
                    " ".join(conclusion_text).strip()
                    if conclusion_text
                    else None
                )
        except Exception as e:
            logging.error(f"Error extracting conclusion "
                          f"from {pdf_path_or_url}: {e}")
        return None

    @staticmethod
    def extract_images(
        pdf_path_or_url: str, saving_dir: str = "images"
                       ) -> list[dict]:
        """提取pdf图片并保存至指定路径

        Args:
            pdf_path_or_url: pdf路径或url
            saving_dir: 保存图片文件的路径

        Returns:
            图片信息字典的列表
        """
        try:
            saving_dir = Path(saving_dir)
            saving_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(saving_dir, os.W_OK):
                raise PermissionError(f"No write permission "
                                      f"for directory: {saving_dir}")

            image_info = []
            with AdvancedPDFParser._load_pdf(
                pdf_path_or_url, way='fitz'
            ) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    figure_captions = re.findall(
                        r'(?i)(fig\.?|figure)\s*\d+[^\n]*',
                        text,
                        re.MULTILINE | re.IGNORECASE
                    )

                    images = page.get_images(full=True)
                    for idx_image, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image['image']
                        image_ext = base_image['ext']

                        with Image.open(BytesIO(image_bytes)) as tmp_img:
                            # Skip small images (e.g., logos)
                            if tmp_img.width < 100 or tmp_img.height < 100:
                                continue
                            image_filename = saving_dir / (
                                f"page_{page_num}_img_{idx_image}.{image_ext}"
                            )
                            tmp_img.save(image_filename)

                            img_bbox = page.get_image_bbox(img)
                            caption = next(
                                (
                                    cap for cap in figure_captions
                                    if img_bbox and abs(
                                        text.find(cap) - img_bbox.y0
                                    ) < 200
                                ),
                                'No Caption'
                            )
                            image_info.append({
                                "filename": str(image_filename),
                                "page": page_num,
                                "caption": caption,
                                "size": (tmp_img.width, tmp_img.height)
                            })
            return image_info
        except Exception as e:
            logging.error(f"Error extracting "
                          f"images from {pdf_path_or_url}: {e}")
            return []


class GrobidPDFParser:
    '''使用Grobid解析pdf

    使用之前请确保Grobid被部署到本地
    '''
    local_port = None

    @staticmethod
    def config_local_port(port: int):
        GrobidPDFParser.local_port = port

    @staticmethod
    def parse_pdf(pdf_path_or_url: str,
                  clean_tmp_file: bool = True,
                  default_port: int = 8070) -> Paper:
        '''解析pdf，支持文件路径及pdf url

        Args:
            pdf_path_or_url: pdf文件路径或pdf url
            clean_tmp_file: 是否删除临时文件

        Returns:
            解析的pdf内容字典
        '''
        if not GrobidPDFParser.local_port:
            raise ValueError(
                "Please use GrobidPDFParser.config_local_port to config a port"
                " before parse"
            )

        isurl = False
        port = GrobidPDFParser.local_port or default_port
        GrobidPDFParser.config_local_port(port)

        if not GrobidPDFParser._is_grobid_available(port):
            raise RuntimeError(
                f"Grobid 服务未在 localhost:{port} 上运行。\n"
                f"你可以使用以下命令启动本地 Grobid:\n"
                f"docker run -t --rm -p {port}:8070 lfoppiano/grobid:0.8.0"
            )

        if (
            pdf_path_or_url.startswith("http://")
            or pdf_path_or_url.startswith("https://")
        ):
            pdf_path = GrobidPDFParser._handle_pdf_url(pdf_path_or_url)
            isurl = True
        else:
            pdf_path = pdf_path_or_url
        with open(pdf_path, 'rb') as file:
            response = requests.post(
                (
                    f"http://localhost:{port}"
                    "/api/processFulltextDocument"
                ),
                files={"input": file}
            )
            if response.status_code != 200:
                raise RuntimeError("Fail to request Grobid" + response.text)
        xml_content = response.content
        parsed_content = GrobidPDFParser._parse_grobid_xml(xml_content)
        if isurl and clean_tmp_file:
            os.remove(pdf_path)

        return Paper(**parsed_content)

    @staticmethod
    def _handle_pdf_url(pdf_url: str) -> str:
        '''处理pdf url

        下载临时文件，并返回文件路径
        '''
        os.makedirs("temp-pdfs", exist_ok=True)
        date_str = datetime.now().strftime(r"%h%m%s-%Y%M%d")
        temp_file_name = "temp-" + date_str + ".pdf"
        file_path = os.path.join("temp-pdfs", temp_file_name)

        response = requests.get(pdf_url, timeout=15)
        if response.ok and "pdf" in response.headers.get(
                "Content-Type", ""
                ):
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return file_path
        else:
            raise ValueError("Invalid url input")

    @staticmethod
    def _parse_grobid_xml(xml_text):
        '''模块化解析xml'''
        soup = BeautifulSoup(xml_text, "xml")

        title_tag = soup.find("titleStmt")
        title = None
        if title_tag:
            title_el = title_tag.find("title")
            title = title_el.text.strip() if title_el else None

        authors = []
        for author_tag in soup.find_all("author"):
            pers = author_tag.find("persName")
            if pers:
                forename = pers.find("forename")
                surname = pers.find("surname")
                full_name = []
                if forename:
                    full_name.append(forename.text.strip())
                if surname:
                    full_name.append(surname.text.strip())
                if full_name:
                    authors.append(" ".join(full_name))

        abstract = ""
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            abstract = " ".join(
                p.get_text(strip=True)
                for p in abstract_tag.find_all("p")
            )

        sections = []
        for div in soup.find_all("div"):
            heading_tag = div.find("head")
            if heading_tag:
                heading = heading_tag.get_text(strip=True)
                content = " ".join(
                    p.get_text(strip=True)
                    for p in div.find_all("p")
                )
                if content:
                    sections.append({"heading": heading, "content": content})

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "sections": sections
        }

    @staticmethod
    def _is_grobid_available(port: int) -> bool:
        try:
            test_url = f"http://localhost:{port}/api/isalive"
            response = requests.get(test_url, timeout=2)
            return response.ok
        except Exception:
            return False


class ToolBuilder:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    def __init__(self, name: str = None, description: str = ''):
        self.name = name
        self.description = description
        self.parameters = {
            'type': 'object',
            'properties': {},
            'required': []
        }

    def add_param(self,
                  name: str,
                  description: str = '',
                  param_type: Literal['string', 'integer', 'number', 'boolean',
                                      'array', 'object'] | type = 'string',
                  required: bool = True) -> None:
        if isinstance(param_type, type):
            param_type = ToolBuilder.type_map.get(param_type)
            if not param_type:
                raise ValueError(f"Unsupported Python type: {param_type}")
        self.parameters['properties'][name] = {
            'type': param_type,
            'description': description
        }
        if required:
            self.parameters['required'].append(name)

    def build(self) -> dict:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
        }
    
    def from_dict(self, tool: dict) -> ToolBuilder:
        tool = tool.get("function", tool)
        self.name = tool.get('name', "")
        self.description = tool.get("description", "")
        self.parameters = tool.get("parameters", {"type": "object", "properties": {}, "required": []})
        return self
    
    def __len__(self):
        return len(self.parameters['properties'])


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool_builder: ToolBuilder, name: str):
        self.tools[name] = tool_builder.build()

    def build_all(self, *names):
        if names is None:
            return list(self.tools.values())
        else:
            return [
                self.tools.get(name, None)
                for name in names
            ]


if __name__ == '__main__':
    warn()
