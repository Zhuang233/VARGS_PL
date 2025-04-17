from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from .shapenet_setvec import shapenet
import os
import sys
import torch
import numpy as np
from torchmcubes import marching_cubes
from scipy.spatial import cKDTree

class ShapeNetsetvecDataModule(LightningDataModule):
    def __init__(self,
                batch_size: int = 4,
                num_workers: int = 4,
                transform=None,
                sdf_sampling=True,
                sdf_size=4096,
                surface_sampling=True,
                surface_size=2048,
                return_sdf=True):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.surface_sampling = surface_sampling
        self.surface_size = surface_size
        current_dir = os.path.dirname(__file__)
        self.dataset_folder = os.path.join(current_dir, "shapenetcoreV2") 
        self.return_sdf = return_sdf


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # 分割数据集
    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = shapenet("train", self.transform, self.sdf_sampling, self.sdf_size, self.surface_sampling, self.surface_size, self.dataset_folder, self.return_sdf)
            self.val_dataset = shapenet("test", self.transform, self.sdf_sampling, self.sdf_size, self.surface_sampling, self.surface_size, self.dataset_folder, self.return_sdf)

        if stage in (None, "test"):
            self.test_dataset = shapenet("test", self.transform, self.sdf_sampling, self.sdf_size, self.surface_sampling, self.surface_size, self.dataset_folder, self.return_sdf)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)





def save_obj(filename, vertices, triangles):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for tri in triangles:
            # OBJ 文件的索引从 1 开始
            f.write(f'f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n')

def process_batch(sdfs, points, grid_size=64, iso_level=0.0):
    batch_size = sdfs.shape[0]
    for i in range(batch_size):
        sdf = sdfs[i]       # 形状: (N,)
        pts = points[i]     # 形状: (N, 3)

        # 计算点云的边界框
        min_bound = pts.min(dim=0)[0]
        max_bound = pts.max(dim=0)[0]

        # 创建体素网格
        x = torch.linspace(min_bound[0], max_bound[0], grid_size)
        y = torch.linspace(min_bound[1], max_bound[1], grid_size)
        z = torch.linspace(min_bound[2], max_bound[2], grid_size)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # 形状: (grid_size^3, 3)

        # 使用 cKDTree 进行最近邻搜索
        grid_coords_np = grid_coords.cpu().numpy()
        pts_np = pts.cpu().numpy()
        sdf_np = sdf.cpu().numpy()

        tree = cKDTree(pts_np)
        _, nearest_idx = tree.query(grid_coords_np, k=1)

        sdf_grid = torch.from_numpy(sdf_np[nearest_idx]).reshape(grid_size, grid_size, grid_size)

        # 应用 Marching Cubes 算法
        vertices, triangles = marching_cubes(sdf_grid, iso_level)

        # 将顶点坐标从体素索引空间转换回原始坐标空间
        scale = (max_bound - min_bound) / (grid_size - 1)
        vertices = vertices * scale.numpy() + min_bound.numpy()

        # 保存为 OBJ 文件
        filename = f'output_{i}_{grid_size}.obj'
        save_obj(filename, vertices, triangles)
        print(f'Saved mesh to {filename}')
        break    
if __name__ == "__main__":
    from lightning.pytorch.cli import LightningCLI
    from lightning import LightningModule

    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()
        def forward(self, x): return x
        def training_step(self, batch, batch_idx): return None
        def configure_optimizers(self): return None

    cli = LightningCLI(
        model_class=DummyModel,
        datamodule_class=ShapeNetsetvecDataModule,
        run = False
    )

    cli.datamodule.setup("fit")

    for batch in cli.datamodule.train_dataloader():
        # # print("Data sample:", batch)
        points, labels, surface, num_vol, num_near , path = batch
        print(points.shape)
        print(labels.shape)
        # # 获取当前脚本的目录
        # current_dir = os.path.dirname(os.path.abspath(__file__))

        # # 获取上层目录
        # parent_dir = os.path.dirname(current_dir)

        # # 将上层目录添加到 sys.path
        # sys.path.insert(0, parent_dir)

        # from utils import implicit_surface_to_mesh
        # print(labels.shape)
        # print(points.shape)
        # implicit_surface_to_mesh(labels[0].detach().cpu().numpy(), points[0].detach().cpu().numpy(), "test.off", "test.ply", 256, 16)
        process_batch(labels, points, grid_size=512)



        break
