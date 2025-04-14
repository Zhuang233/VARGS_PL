from lightning.pytorch.callbacks import Callback
from utils.pc import save_point_cloud_as_ply
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class PointCloudSaver(Callback):
    def __init__(self, num_saves=5):
        """
        Args:
            num_saves (int): 希望在整个训练过程中保存点云的次数。
        """
        self.num_saves = num_saves
        self.gt_saved = False
        self.save_interval = None
        self.save_dir = None

    def on_fit_start(self, trainer, pl_module):
        # 计算保存间隔
        self.save_interval = max(1, trainer.max_epochs // self.num_saves)
        log_dir = pl_module.logger.log_dir
        if log_dir is None:
            log_dir = "./none_logs"
        self.save_dir = os.path.join(log_dir, "vis_ply")
        os.makedirs(self.save_dir, exist_ok=True)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, "cached_points"):
            epoch = pl_module.current_epoch
            pc, gt_pc, taxonomy_id, model_id = pl_module.cached_points

            # 保存 Ground Truth 点云
            if not self.gt_saved:
                gt_file = os.path.join(self.save_dir, f"{taxonomy_id}-{model_id}-gt.ply")
                save_point_cloud_as_ply(gt_pc, gt_file)
                self.gt_saved = True

            # 根据保存间隔保存重建的点云
            if epoch % self.save_interval == 0 or epoch == trainer.max_epochs-1:
                recon_file = os.path.join(self.save_dir, f"{taxonomy_id}-{model_id}-epoch{epoch}.ply")
                save_point_cloud_as_ply(pc, recon_file)
        # if hasattr(pl_module, "cached_points"):
        #     epoch = pl_module.current_epoch
        #     pc ,taxonomy_id, model_id = pl_module.cached_points
        #     file_name = pl_module.ply_vis_dir + "/%s-%s-%d.ply" % (taxonomy_id, model_id, epoch)
        #     save_point_cloud_as_ply(pc, file_name)

        #     if pl_module.current_epoch == 0:
        #         src_file = "dataset/ShapeSplatsV1_part20/%s-%s.ply" % (taxonomy_id,model_id)
        #         dst_dir = pl_module.logger.log_dir + "/vis_ply"
        #         shutil.copy(src_file, dst_dir)
        #         shutil.move(dst_dir + "/%s-%s.ply" % (taxonomy_id,model_id), dst_dir + "/%s-%s-gt.ply" % (taxonomy_id,model_id))

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        points_rebuild, taxonomy_id, model_id = outputs
        if batch_idx == 0:
            # 从 batch 中提取并保存经过预处理的 Ground Truth 点云
            _, _, data, _, _, _, _ = batch  # 根据您的数据结构调整索引
            gt_pc = data[0].detach().cpu().numpy()

            pl_module.cached_points = (
                points_rebuild[0].detach().cpu().numpy(),
                gt_pc,
                taxonomy_id[0],
                model_id[0]
            )
        # points_rebuild, taxonomy_id, model_id  = outputs
        # if(batch_idx ==0):
        #     pl_module.cached_points = points_rebuild[0].detach().cpu().numpy(), taxonomy_id[0], model_id[0]