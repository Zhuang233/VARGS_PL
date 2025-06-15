from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from .shapenet_gaussian import ShapeNetGaussian
import os

class ShapeNetGSDataModule(LightningDataModule):
    def __init__(self,
                batch_size: int = 4,
                train_split_file: str = None,
                val_split_file: str = None,
                test_split_file: str = None,
                num_workers: int = 4,
                data_split_dir: str = None,
                config_ShapeNetGaussian = None):

        super().__init__()
        self.train_split_file = train_split_file
        self.val_split_file = val_split_file
        self.test_split_file = test_split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split_dir = data_split_dir
        self.config_ShapeNetGaussian = config_ShapeNetGaussian


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # 分割数据集
    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_data_list_path = os.path.join(self.data_split_dir, self.train_split_file)
            val_data_list_path = os.path.join(self.data_split_dir, self.val_split_file)
            self.train_dataset = ShapeNetGaussian(train_data_list_path, self.config_ShapeNetGaussian)
            self.val_dataset = ShapeNetGaussian(val_data_list_path, self.config_ShapeNetGaussian)

        if stage in (None, "test"):
            test_data_list_path = os.path.join(self.data_split_dir, self.test_split_file)
            self.test_dataset = ShapeNetGaussian(test_data_list_path, self.config_ShapeNetGaussian)

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
        datamodule_class=ShapeNetGSDataModule,
        run = False
    )

    cli.datamodule.setup("fit")

    from shapenet_gaussian import unnorm_gs
    for batch in cli.datamodule.train_dataloader():
        # print("Data sample:", batch)
        taxonomy_id, model_id, gs, centroid, scale_factor, scale_c, scale_m = batch
        
        ret = unnorm_gs(gs, centroid, scale_factor, scale_c, scale_m)

        print(ret)
        print(data_original)
        break
