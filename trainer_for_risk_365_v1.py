from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from retfound_base_lightningmodule import RETFoundLightning

from datamodule import SingleImageDataModule

INPUT_SIZE = 896
WARMUP_EPOCHS = 10
EPOCHS = 50

checkpoint_callback = ModelCheckpoint(
    monitor="val_auc_roc",
    filename="retfound-{epoch:02d}-{val_auc_roc:.2f}",
    every_n_epochs=1,
    # save_top_k=10,
)

tensorboard_logger = TensorBoardLogger(save_dir="/mnt/4T/xujialiu-ckpt/my_lightning_model/risk_365", name="tensorboard")
csv_logger = CSVLogger(save_dir="/mnt/4T/xujialiu-ckpt/my_lightning_model/risk_365", name="csv", flush_logs_every_n_steps=20)
loggers = [tensorboard_logger, csv_logger]

trainer = Trainer(
    devices=[0],
    accelerator="cuda",
    # strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback],
    logger=[tensorboard_logger, csv_logger],
    accumulate_grad_batches=5,
    use_distributed_sampler=False,
    benchmark=True,
    precision="16-mixed",
    # limit_train_batches=0.05,
    # limit_val_batches=0.03,
    # limit_test_batches=0.03,
)

model = RETFoundLightning(
    use_original_retfound_ckpt="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/my-model/other_models/model-checkpoints/RETFound.pth",
    img_size=INPUT_SIZE,
    warmup_epochs=WARMUP_EPOCHS,
    num_classes=2
)
datamodule = SingleImageDataModule(
    excel_file="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/data_cleansing/calculate time diff/macula_table_risk_level.xlsx",
    img_path_col="macula_filename",
    label_col="label_365_diff_binary",
    data_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
    batch_size=6,
    input_size=INPUT_SIZE,
)

trainer.fit(
    model,
    datamodule=datamodule,
)

# nohup /home/xujialiu/miniconda3/envs/pytorch/bin/python /mnt/4T/xujialiu-ckpt/my_lightning_model/trainer_for_risk_365_v1.py > risk_365_output.log 2>&1 &