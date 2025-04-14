from lightning.pytorch.callbacks import Callback

class debugger(Callback):
    """
    用于调试的回调类。
    """
    def __init__(self):
        pass

    def on_after_backward(self, trainer, pl_module):
        return
        pl_module.logger_txt.info("🔍 [Debug] Checking unused parameters (after backward):")
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is None:
                pl_module.logger_txt.info(f"⚠️ Unused parameter: {name}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        return
        #DEBUG是否梯度消失和爆炸
        # 初始化一个字典来存储每一层的梯度范数
        norms = {}

        # 遍历模型的所有参数
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # 计算参数的 L2 范数（即 2-范数）
                norm = param.grad.norm(2)
                # 将范数存储在字典中，键为参数名称
                norms[f'grad_norm_{name}'] = norm

        # 使用 log_dict 记录所有梯度范数
        pl_module.log_dict(norms, prog_bar=True, logger=True)