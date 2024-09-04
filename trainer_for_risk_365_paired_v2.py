from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from convnext_paired_lightningmodule import PairedConvNextLightning
from datamodule import SingleImageDataModule, PairedImageDataModule

INPUT_SIZE = 512
WARMUP_EPOCHS = 10
EPOCHS = 50
BATCH_SIZE = 1
BASE_LR = 1e-4
DIR_PATH = "./test_convnext_paired"
ACCUMULATE_GRAD_BATCHES = 32

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_roc",
    filename="convnext-{epoch:02d}-{val_auc_roc:.3f}-{val_auc_pr:.3f}",
    every_n_epochs=1,
    save_top_k=-1,
)

tensorboard_logger = TensorBoardLogger(save_dir=DIR_PATH, name="tensorboard")
csv_logger = CSVLogger(save_dir=DIR_PATH, name="csv", flush_logs_every_n_steps=20)
loggers = [tensorboard_logger, csv_logger]

trainer = Trainer(
    devices=[0],
    accelerator="cuda",
    # strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback],
    logger=[tensorboard_logger, csv_logger],
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    use_distributed_sampler=False,
    benchmark=True,
    precision="16-mixed",
    num_sanity_val_steps=0,
    # limit_train_batches=0.01,
    # limit_val_batches=0.03,
    # limit_test_batches=0.03,
)

model = PairedConvNextLightning(
    base_learning_rate=BASE_LR,
    img_size=INPUT_SIZE,
    warmup_epochs=WARMUP_EPOCHS,
    num_classes=2,
)

datamodule = PairedImageDataModule(
    excel_file="/mnt/4T/xujialiu/my_lightning_model/data_cleansing/calculate time diff/disc_macula_table_risk_level.xlsx",
    macula_img_path_col="macula_filename",
    disc_img_path_col="disc_filename",
    label_col="label_365_diff_binary",
    macula_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
    disc_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_disc_all",
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE,
)

trainer.fit(
    model,
    datamodule=datamodule,
)

# nohup /mnt/4T/xujialiu/miniconda/envs/pt/bin/python /mnt/4T/xujialiu/my_lightning_model/trainer_for_risk_365_paired_v2.py > risk_365_convnext_paired_output.log 2>&1 &