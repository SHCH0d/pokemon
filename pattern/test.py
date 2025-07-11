import logging

def setup_logger(log_path):
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logger("test.log")
logger.debug("测试日志写入")
print("日志写入完成")