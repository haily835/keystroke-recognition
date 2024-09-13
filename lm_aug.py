from lightning.pytorch.cli import LightningCLI
from lightning_utils.lm_augmented import LMModule
from lightning_utils.lm_module import KeyClf

cli = LightningCLI(
    model_class=KeyClf,
    datamodule_class=LMModule,
    save_config_callback=None
)
