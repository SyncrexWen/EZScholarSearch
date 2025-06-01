
import logging
from scholarly import scholarly, ProxyGenerator

# 设置日志级别为 DEBUG
logging.getLogger('scholarly').setLevel(logging.DEBUG)


def check_cookies_validity(cookies_file):
    """
    检查 Google Scholar 的 cookies 是否有效。

    Args:
        cookies_file (str): cookies 文件的路径

    Returns:
        bool: True 表示 cookies 有效，False 表示无效
    """
    try:
        # 初始化 ProxyGenerator
        pg = ProxyGenerator()
        # 暂时禁用代理以避免干扰
        # pg.FreeProxies()  # 如果需要代理，可取消注释

        # 获取会话（_new_session 已加载 cookies）
        session = pg.get_session()

        # 发送请求到 Google Scholar 个人主页
        response = session.get('https://scholar.google.com/citations?hl=en', timeout=10)

        # 检查响应状态码
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}")
            if response.status_code == 403:
                print("可能触发了 Google Scholar 的反爬机制（403 Forbidden）")
            elif response.status_code == 429:
                print("请求过于频繁（429 Too Many Requests）")
            return False

        # 检查是否包含登录后的标志（如“My profile”）
        if 'My profile' in response.text:
            print("Cookies 有效，登录成功！")
            return True
        else:
            print("Cookies 无效或未登录，响应中未找到 'My profile'")
            # 检查是否包含 CAPTCHA
            if 'gs_captcha_ccl' in response.text or 'recaptcha' in response.text:
                print("检测到 CAPTCHA，可能需要手动登录解决")
            return False

    except FileNotFoundError:
        print(f"未找到 cookies 文件: {cookies_file}")
        return False
    except Exception as e:
        print(f"检查 cookies 时发生错误: {e}")
        return False


if __name__ == "__main__":
    cookies_file = r"C:\Users\aasus\AppData\Local\Google\Chrome\User Data\Default\Network\cookies.json"
    print(f"正在检查 cookies 文件: {cookies_file}")
    is_valid = check_cookies_validity(cookies_file)
    print(f"Cookies 是否有效: {is_valid}")
