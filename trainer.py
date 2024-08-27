from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightningmodule import RETFoundLightning

from datamodule import SingleImageDataModule


checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_roc",
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

# trainer = Trainer(
#     devices=[0],
#     accelerator="gpu",
#     # strategy=DDPStrategy(find_unused_parameters=False),
#     max_epochs=10,
#     callbacks=[checkpoint_callback],
#     logger=loggers,
#     use_distributed_sampler=False,
#     limit_train_batches=0.1,
#     limit_val_batches=0.03,
#     limit_test_batches=0.03,
#     benchmark=True,
#     precision="16-mixed",
# )

trainer = Trainer(
    devices=[1],
    accelerator="gpu",
    # strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=10,
    callbacks=[checkpoint_callback],
    logger=loggers,
    use_distributed_sampler=False,
    limit_train_batches=0.1,
    limit_val_batches=0.03,
    limit_test_batches=0.03,
    benchmark=True,
    precision="16-mixed",
)


model = RETFoundLightning(
    use_original_retfound_ckpt="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/my-model/other_models/model-checkpoints/RETFound.pth",
    warmup_epochs=5,
)
datamodule = SingleImageDataModule(
    excel_file="data_cleansing/get_macula_table/macula_table.xlsx",
    data_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
    batch_size=6,
)

trainer.fit(
    model,
    datamodule=datamodule,
)
