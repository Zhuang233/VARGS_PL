import logging
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def get_logger(log_file: str = "log.txt"):
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        @rank_zero_only
        def init_loger():
            # 控制台输出（可选）
            # console_handler = logging.StreamHandler()
            # console_handler.setLevel(logging.INFO)

            # 文件输出
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 格式
            formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            # console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        init_loger()
    return logger


