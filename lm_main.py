from lightning.pytorch.tuner import Tuner
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from lightning_utils import datamodule
from lightning_utils.lm_datamodule import LMKeyStreamModule
from lightning_utils.lm_module import KeyClf
import pandas as pd
import glob

"""
Local: 
python lm_main.py -c configs/lm_clf.yaml \
--trainer.accelerator mps \
--trainer.devices auto \
--data.frames_dir  datasets/video-2/raw_frames \
--data.labels_dir datasets/video-2/labels \
--data.landmarks_dir datasets/video-2/landmarks

Kaggle:

Train:
python lm_main.py -c configs/lm_clf.yaml \
--trainer.accelerator gpu \
--trainer.devices 0,1 \
--trainer.logger.save_dir lm_clf \
--data.frames_dir /kaggle/input/topview-lm/raw_frames \
--data.labels_dir /kaggle/input/topview-lm/labels \
--data.landmarks_dir /kaggle/input/topview-lm/landmarks\
--model.lr 0.001

Test:
python lm_main.py -c configs/lm_clf.yaml \
--trainer.accelerator gpu \
--trainer.devices 0 \
--data.frames_dir /kaggle/input/topview-lm/raw_frames \
--data.labels_dir /kaggle/input/topview-lm/labels \
--data.landmarks_dir /kaggle/input/topview-lm/landmarks\
--model.lr 0.001
--ckpt_path PATH\
"""

cli = LightningCLI(
    model_class=KeyClf,
    datamodule_class=LMKeyStreamModule,
    save_config_callback=None
)
