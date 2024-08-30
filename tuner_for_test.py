from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from retfound_base_lightningmodule import RETFoundLightning
from lightning.pytorch.tuner import Tuner
from datamodule import SingleImageDataModule

INPUT_SIZE = 896
# 26 448
# 6 896
WARMUP_EPOCHS = 10
EPOCHS = 50

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_auc_roc",
#     filename="retfound-{epoch:02d}-{val_acc:.2f}",
#     every_n_epochs=1,
#     save_top_k=-1,
# )

# tensorboard_logger = TensorBoardLogger(save_dir="test_root", name="tensorboard")
# csv_logger = CSVLogger(save_dir="test_root", name="csv", flush_logs_every_n_steps=20)
# loggers = [tensorboard_logger, csv_logger]

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    # strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=10,
    # callbacks=[checkpoint_callback],
    # logger=loggers,
    accumulate_grad_batches=2,
    use_distributed_sampler=False,
    limit_train_batches=0.05,
    limit_val_batches=0.03,
    limit_test_batches=0.03,
    benchmark=True,
    precision="16-mixed",
)

model = RETFoundLightning(
    use_original_retfound_ckpt="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/my-model/other_models/model-checkpoints/RETFound.pth",
    img_size=INPUT_SIZE,
    warmup_epochs=WARMUP_EPOCHS,
    is_save_confusion_matrix=False,
)

datamodule = SingleImageDataModule(
    excel_file="data_cleansing/get_macula_table/macula_table.xlsx",
    img_path_col="macula_filename",
    label_col="label",
    data_folder="/home/xujialiu/mnt-4T-xujialiu/my_lightning_model/xujialiu-mnt-4t/data_macula_all",
    batch_size=6,
    input_size=INPUT_SIZE,
)

tuner = Tuner(trainer)

# 查找最佳批次大小
tuner.scale_batch_size(model, mode="binsearch", datamodule=datamodule)


# 运行学习率查找器
# lr_finder = tuner.lr_find(model, datamodule=datamodule)
# fig = lr_finder.plot(suggest=True)
# fig.savefig("lr_finder.png")
