from lightning.pytorch.cli import LightningCLI

from lightning_utils.lm_datamodule import LMKeyStreamModule
from lightning_utils.lm_module import KeyClf

cli = LightningCLI(
    model_class=KeyClf,
    datamodule_class=LMKeyStreamModule,
    save_config_callback=None
)