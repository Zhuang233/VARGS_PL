import os
import logging
from lightning.pytorch.callbacks import Callback
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

class logger_txt_callback(Callback):
    """
    用于保存训练txt日志的回调类。
    """
    def __init__(self):
         pass

    def on_fit_start(self, trainer, pl_module):
        # 只在训练开始时初始化日志器
        log_dir = pl_module.logger.log_dir
        if log_dir is None:
            log_dir = "./none_logs"
        log_path = os.path.join(log_dir, "train_log.txt")
        pl_module.logger_txt = get_logger(log_path)
        pl_module.logger_txt.info("🚀 Training started")

        pl_module.writer = pl_module.logger.experiment

    def on_train_epoch_end(self,trainer, pl_module):
        # 训练结束时记录当前epoch的训练损失
        loss = trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            pl_module.logger_txt.info(f"📣 Epoch {pl_module.current_epoch} finished. Train Loss: {loss:.4f}")