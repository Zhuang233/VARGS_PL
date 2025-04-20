from lightning.pytorch.cli import LightningCLI
from models.diffgs.diffgsvae import DiffGSAutoEncoder

def main():
    cli = LightningCLI(
        # subclass_mode_model=True,
        DiffGSAutoEncoder,
        subclass_mode_data=True,
    )

    # cli.datamodule.setup("fit")

    # for batch in cli.datamodule.train_dataloader():
    #     print("Data sample:", batch)
    #     break

if __name__ == "__main__":
    main()
