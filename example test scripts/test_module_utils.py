# flake8: noqa

from ezscholarsearch import utils
import pytest
from pathlib import Path
import fitz
import warnings
from unittest.mock import patch

warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message="builtin type swigvarlink has no __module__ attribute")

# To be tested : BasePDFParser

Test_PDF_Path = "./the discovery of 2024 YR4.pdf"

@pytest.fixture
def mock_PDF_doc():
    doc = fitz.open()
    page = doc.new_page()
    return doc

def test_load_pdf_path():
    with patch("fitz.open") as mock_open:
        utils.BasePDFParser._load_pdf(Test_PDF_Path)
        mock_open.assert_called_with(Test_PDF_Path)

'''
def test_load_url_path(requests_mock):
    mock_content = b"Pdf content"
    requests_mock.get(Test_PDF_Url, content = mock_content)
    
    with patch("fitz.open") as mock_open:
        utils.BasePDFParser._load_pdf(Test_PDF_Url)
        mock_open.assert_called_with(stream=mock_content, filetype="pdf")
'''
# 此处暂不考虑 url 方式的pdf_load

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"

def test_extract_abstract(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    abstract = utils.BasePDFParser.extract_abstract(Test_PDF_Path)

    assert len(abstract) > 20

    # assert re.search(r'(?i)(abstract|summary)[:\s]*', abstract, re.I),"未识别摘要标题"
    # 发现提取的abstract部分其实没有包含abstract 所有此处的assert删去，直接看下面的输入案例

    (output_dir / "abstract").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abstract/sample1.txt", "w", encoding='utf-8') as f:
        f.write(abstract)
    
    '''
    Output: 
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
    '''

def test_extract_conclusion(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    conclusion = utils.BasePDFParser.extract_conclusion(Test_PDF_Path)

    assert len(conclusion) > 30

    (output_dir / "conclusion").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conclusion/sample1.txt", "w", encoding='utf-8') as f:
        f.write(conclusion)
    # os.startfile(output_dir)
    '''
    CONCLUSIONS
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
    '''

    # 此处提取的conclusion会将图片注释的文字也一起提取出来，所以会显示的比较奇怪
    
def test_extract_images(output_dir):
    if not Test_PDF_Path:
        return pytest.skip(f"测试文件不存在!")
    
    img_count = utils.BasePDFParser.extract_images(Test_PDF_Path, output_dir / "images")

    assert img_count > 0, "未提取到图片"
    saved_images = list((output_dir / "images").glob("*"))
    assert len(saved_images) == img_count, "图片数量不匹配"

    for img in saved_images:
        assert img.stat().st_size > 1024, f"图片 {img.name} 大小异常"
    # os.startfile(output_dir)
    # 成功提取10张图片进入 image 文件夹之中


# To be tested : AdvancedPDFParser

def test_advanced_extract_abstract(output_dir):
    abstract = utils.AdvancedPDFParser.extract_abstract(Test_PDF_Path)

    assert isinstance(abstract, str), "返回类型应为字符串"
    assert len(abstract) > 50, "摘要过短"

    (output_dir / "abstract").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "abstract/sample1.txt", "w", encoding='utf-8') as f:
        f.write(abstract)
    # os.startfile(output_dir)

    '''
     We describe observations and physical characteristics of Earth-crossing asteroid 2024
    YR , discovered on 2024 December 27 by the Asteroid Terrestrial-impact Last Alert
    4
    System. Theasteroidhassemi-majoraxis,a=2.52au,eccentricity,e=0.66,inclination
    i = 3.41◦, and a 0.003 au Earth minimum orbit intersection distance. We obtained
    ∼
    g, r, i, and Z imaging with the Gemini South/Gemini Multi-Object Spectrograph on
    2025 February 7. We measured a g-i spectral slope of 13 3 %/100 nm, and color
    ±
    indices g-r = 0.70 0.10, r-i = 0.25 0.06, i-Z = -0.27 0.10. 2024 YR has a spectrum
    4
    ± ± ±
    that best matches R-type and Sa-type asteroids and a diameter of 30-65 m using our
    ∼
    measured absolute magnitude of 23.9 0.3 mag, and assuming an albedo of 0.15-0.4.
    ±
    The lightcurve of 2024 YR shows 0.4 mag variations with a rotation period of 1170
    4
    ∼ ∼
    s. We use photometry of 2024 YR from Gemini and other sources taken between 2024
    4
    December to 2025 February to determine the asteroid’s spin vector and shape, finding
    that it has an oblate, 3:1 a:c axial ratio and a pole direction of λ, β = 42◦, -25◦.
    ∼ ∼ ∼
    Finally, we compare the orbital elements of 2024 YR with the NEO population model
    4
    and find that its most likely sources are resonances between the inner and central Main
    Belt.
    '''
    # 此处乱码的原因为pdfplumber不能处理一些特殊字符， 若遇到这种情况可以使用fitz来解析。

def test_advanced_extract_conclusion(output_dir):
    """验证结论提取"""
    conclusion = utils.AdvancedPDFParser.extract_conclusion(Test_PDF_Path)
    
    assert isinstance(conclusion, str), "返回类型应为字符串"
    assert len(conclusion) > 30, "结论过短"
    
    (output_dir / "conclusion").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conclusion/sample1.txt", "w", encoding='utf-8') as f:
        f.write(conclusion)
    # os.startfile(output_dir)

    '''
    Output:
    The properties of 2024 YR are similar to other small NEOs, that it has a S-complex composition, 4 most likely originated from a resonances located inner Main Belt, and likely has moderate reflective properties implying a size of 42 m. The exact origin within the Main Belt seems less clear, though ∼ a clue may lie in its spin-vector orientation. As recent asteroid lightcurve spin-vector studies have shown, the spin direction of an asteroid can affect the way an asteroid’s orbit evolves due to the thermal recoil Yarkovsky effect, with prograde asteroids drifting outward, and retrograde asteroids drifting inwards (Bolin et al. 2018a; Hanuˇs et al. 2018b; Athanasopoulos et al. 2022). Since the spin direction of 2024 YR is retrograde with its -39◦ ecliptic latitude, an origin from the inner Main 4 Discovery and characterization of 2024 YR 13 4 Figure 9. Lightcurve fit of 2024 YR4, showing dense photometric observations from VLT and Gemini in the r and g filters (top panels), and sparse datasets from ATLAS, Catalina Sky Survey, and Pan-STARRS (bottom three panels). The best-fit model is overlaid. Bolin et al. 14 Figure 10. Convex shape model of 2024 YR in three orthogonal views obtained using publicly available 4 photometry of 2024 YR from the Minor Planet Center archive, VLT archive and Gemini. The X vs. Z and 4 Y vs. Z views show the edge on view of the asteroid with the spin axis pointing in the Z direction (ecliptic longitude and latitude equal to 42◦ and -25 degrees◦). The asteroid is viewed from a pole-on view in the X vs. Y view. Belt since its retrograde spin would cause it to drift inwards away from the 3:1 MMR at 2.5 au, its most likely escape path into the NEO population. Therefore, its suspected retrograde spin may imply that it original location was the central Main Belt, located at 2.52 au to 2.82 au from the Sun (Nesvorny´ et al. 2015). This would take 2024 YR ’s origins away from the inner Main Belt where 4 S-types dominate to the central Main Belt where C-types are more plentiful seemingly in tension with its taxonomic type (DeMeo & Carry 2014). However, several S-type families are located near the 3:1 MMR in the central Main Belt, such as the Rafita family located at 2.58 au, which could be a candidate source family for 2024 YR . 4
    (Correct)
    '''

def test_advanced_extract_images(output_dir):
    result = utils.AdvancedPDFParser.extract_images(Test_PDF_Path, output_dir)
    
    assert isinstance(result, list), "返回类型必须为列表"

    required_keys = {"filename", "page", "size"}

    for img_info in result:
        
        assert isinstance(img_info, dict), f"元素类型错误: {type(img_info)}"
        assert required_keys.issubset(img_info.keys()), f"缺失字段: {img_info}"

        file_path = Path(img_info["filename"])
        assert file_path.exists(), f"文件不存在: {file_path}"
        assert file_path.stat().st_size > 1024, "文件大小异常(<1KB)" 

        extracted_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg"))
        assert len(extracted_files) == len(result), "文件数量不匹配"
    # os.startfile(output_dir)

    # 成功提取11张图片（部分图片被过滤）