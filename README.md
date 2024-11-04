# Skeleton-based Keystroke recognition 

## Dataset
- Our dataset in the skeleton format (using MediaPipe) can be found in `datasets` folder.
- Raw images are available on Kaggle: https://www.kaggle.com/datasets/haily1234/keystroke-recognition-keyvr

## Training and testing

- keystroke detector to classify idle and typing moments
- keystroke classifier to identify which key is pressed
- training 2 models can be found in `main.ipynb`

## Demo
![Demo](https://github.com/haily835/keystroke-recognition/blob/main/output.mp4)


## Reproduce results
- config_path: path to yaml file associating with checkpoint (ie. Hyperformer detector: ckpts/HyperGT/det/config.yaml; ckpts/HyperGT/clf/config.yaml)
- ckpt_path: path to model checkpoint (ie. Hyperformer detector: ckpts/HyperGT/det/epoch=12-step=6929.ckpt; classifier ckpts/HyperGT/clf/epoch=17-step=7722.ckpt)
- accelerator: none cpu
- test_devices

```
!python train.py test -c {config_path} \
--trainer.accelerator {accelerator} \
--trainer.devices {test_devices} \
--ckpt_path {ckpt_path}
```

## Run the 2 stages on video frames from the pre-train model:
```
!python test.py \
--data_dir ./datasets/KeyVR/landmarks \
--clf_ckpt {clf_ckpt_path} \
--det_ckpt {det_ckpt_path} \
--result_dir stream_results \
--window_size 8 \
--videos 6 7 19 20 27 28 \
--module_classpath lightning_utils.lm_module.LmKeyClf
```

## Record videos with label ground truth.

This script was used to record the video to train the model by opening the webcam. A phone camera can be used if webcams are not available. We install Camo on a smartphone and Camo client on MacOs as an alternative to a webcam. Note that the frame number of the keypress event has some delay. For example in our case, using the usb connection, the delay is 4. That means in the CSV file, user typing key A at frame 4 was recorded, however, the correct frame number is 4 + 4 = 8.

This issue will be further investigated, therefore for now you still need to recheck manually to observe the delay.

```
python ./utils/keystroke_recorder.py
```

