from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from .shapenet_setvec import shapenet
import os

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
        print("Data sample:", batch)
        break
