import sys
import logging

# 配置 logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        # 过滤掉仅包含空白符的行（print会自动添加换行符）
        if message.strip() != "":
            self.level(message)

    def flush(self):
        # 需要定义 flush 方法以满足 file-like object 的接口
        pass

# 将 stdout 重定向到 logging 的 info 级别
sys.stdout = LoggerWriter(logging.info)
# 将 stderr 重定向到 logging 的 error 级别 (可选)
sys.stderr = LoggerWriter(logging.error)

# 输出结果示例： 2024-05-20 10:00:00 - INFO - 这行 print 会自动变成一条带时间戳的日志
