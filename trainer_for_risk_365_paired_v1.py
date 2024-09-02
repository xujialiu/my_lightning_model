from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from retfound_paired_lightningmodule import PairedRETFoundLightning


from datamodule import SingleImageDataModule, PairedImageDataModule

INPUT_SIZE = 896
WARMUP_EPOCHS = 10
EPOCHS = 50
BATCH_SIZE = 3
ACCUMULATE_GRAD_BATCHES = 11

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_roc",
    filename="retfound-{epoch:02d}-{val_auc_roc:.2f}",
    every_n_epochs=1,
    save_top_k=-1
)

tensorboard_logger = TensorBoardLogger(save_dir="./risk_365_paired", name="tensorboard")
csv_logger = CSVLogger(save_dir="./risk_365_paired", name="csv", flush_logs_every_n_steps=20)
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

model = PairedRETFoundLightning(
    use_original_retfound_ckpt="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/my-model/other_models/model-checkpoints/RETFound.pth",
    base_learning_rate=5e-4,
    img_size=INPUT_SIZE,
    warmup_epochs=WARMUP_EPOCHS,
    num_classes=2,
)

datamodule = PairedImageDataModule(
    excel_file="/mnt/4T/xujialiu/my_lightning_model/data_cleansing/calculate time diff/disc_macula_table_risk_level.xlsx",
    macula_img_path_col="macula_filename",
    disc_img_path_col = "disc_filename",
    label_col="label_365_diff_binary",
    macula_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
    disc_folder = "/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_disc_all",
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE,
)

trainer.fit(
    model,
    datamodule=datamodule,
)

# nohup /mnt/4T/xujialiu/miniconda/envs/pt/bin/python /mnt/4T/xujialiu/my_lightning_model/trainer_for_risk_365_paired_v1.py > risk_365_paired_output.log 2>&1 &
