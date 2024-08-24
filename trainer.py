from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightningmodule import RETFoundLightning
from datamodule import SingleImageDataModule


checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    filename="retfound-{epoch:02d}-{val_acc:.2f}",
    every_n_epochs=1,
    save_top_k=-1,
)


default_root_dir = (
    "/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/test_datamodule"
)

tensorboard_logger = TensorBoardLogger(save_dir="test_root", name="tensorboard")
csv_logger = CSVLogger(save_dir="test_root", name="csv")
loggers = [tensorboard_logger, csv_logger]

trainer = Trainer(
    devices=[1],
    accelerator="gpu",
    # strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=10,
    callbacks=[checkpoint_callback],
    logger=loggers,
    use_distributed_sampler=True,
    # Add these lines to skip sanity check and enable deterministic mode
)

model = RETFoundLightning()

datamodule = SingleImageDataModule(
    excel_file="data_cleansing/get_macula_table/macula_table.xlsx",
    data_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
)

trainer.fit(model, datamodule=datamodule)
