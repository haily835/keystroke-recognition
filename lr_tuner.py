from lightning.pytorch.tuner import Tuner
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from lightning_utils import datamodule
from lightning_utils.lm_datamodule import LMKeyStreamModule
from lightning_utils.lm_module import KeyClf


def main():
    cli = LightningCLI(
        model_class=KeyClf,
        datamodule_class=LMKeyStreamModule,
        run=False,
    )

    trainer = cli.trainer
    model = cli.model
    dm = cli.datamodule
    # tuner = Tuner(trainer)

    # Run learning rate finder
    # lr_finder = tuner.lr_find(model, datamodule=dm)
    
    # tuner.scale_batch_size(model, datamodule=dm)

    # Results can be found in
    # print(lr_finder.results)
    # print("Tuned batch_size: ", dm.batch_size)

    # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print("Tuned learning_rate: ", new_lr)

    # update hparams of the model
    # model.hparams.lr = new_lr

    # Fit model
    trainer.fit(model, dm)

    # trainer.test(model, dm)

"""
Local: 
python lr_tuner.py -c configs/lm_clf.yaml \
--trainer.accelerator mps \
--trainer.devices auto \
--data.frames_dir  datasets/video-2/raw_frames \
--data.labels_dir datasets/video-2/labels \
--data.landmarks_dir datasets/video-2/landmarks

Kaggle:
python lr_tuner.py -c configs/lm_clf.yaml \
--trainer.accelerator gpu \
--trainer.devices 0,1 \
--data.frames_dir /kaggle/input/topview-lm/raw_frames \
--data.labels_dir /kaggle/input/topview-lm/labels \
--data.landmarks_dir /kaggle/input/topview-lm/landmarks
"""
if __name__ == '__main__':
    main()

