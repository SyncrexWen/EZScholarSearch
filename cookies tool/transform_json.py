import sqlite3
import json

def convert_chrome_cookies_to_json(cookies_db_path, output_json_path):
    try:
        # 连接到 Chrome 的 Cookies 数据库
        conn = sqlite3.connect(cookies_db_path)
        cursor = conn.cursor()

        # 查询 cookies 表（通常名为 cookies）
        cursor.execute("SELECT host_key, name, value, path, expires_utc FROM cookies WHERE host_key LIKE '%.google.com' OR host_key LIKE '%scholar.google.com'")
        rows = cursor.fetchall()

        # 转换为 JSON 格式
        cookies = []
        for row in rows:
            host_key, name, value, path, expires_utc = row
            cookie = {
                "domain": host_key,
                "name": name,
                "value": value,
                "path": path,
                "expires": expires_utc
            }
            cookies.append(cookie)

        # 保存到 JSON 文件
        with open(output_json_path, 'w') as f:
            json.dump(cookies, f, indent=4)

        print(f"Cookies saved to {output_json_path}")
        conn.close()
        return True
    except Exception as e:
        print(f"Error converting cookies: {e}")
        return False

# 示例用法
cookies_db_path = r"C:\Users\aasus\AppData\Local\Google\Chrome\User Data\Default\Network\Cookies"
output_json_path = r"C:\Users\aasus\AppData\Local\Google\Chrome\User Data\Default\Network\cookies.json"
convert_chrome_cookies_to_json(cookies_db_path, output_json_path)