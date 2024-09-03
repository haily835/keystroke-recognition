from lightning.pytorch.cli import LightningCLI

from lightning_utils.datamodule import KeyStreamModule
from lightning_utils.module import KeyClf

cli = LightningCLI(
    model_class=KeyClf,
    datamodule_class=KeyStreamModule,
    save_config_callback=None
)