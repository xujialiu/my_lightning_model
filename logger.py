import logging


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # 忽略空消息
            self.level(message)

    def flush(self):
        pass


def get_logger(filename="output.log", filemode="a"):
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 日志格式
        filename=filename,  # 日志文件名
        filemode=filemode,  # 写入模式，'w'表示每次运行都会覆盖日志文件，'a'表示追加模式
    )
    return logging.getLogger()
