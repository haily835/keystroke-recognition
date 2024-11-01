# Skeleton-based Keystroke recognition 

## Dataset
- Our dataset in the skeleton format (using MediaPipe) can be found in `datasets` folder.
- Raw images are available on Kaggle: 

## Training and testing

- keystroke detector to classify iddle and typing moments
- keystroke classifier to identify which key typed

### Local machine with only cpu for development

1. Training
   Detect model:

```bash
python main.py -c configs/base_detect.yaml fit \
--trainer.accelerator cpu \
--trainer.fast_dev_run True\
--data.frames_dir datasets/video/raw_frames \
--data.labels_dir datasets/video/labels \
--data.num_workers 0 \
--data.idle_gap 2
```


Classifier model:

```bash
python main.py -c configs/base_clf.yaml fit \
--trainer.accelerator cpu \
--trainer.fast_dev_run True\
--data.frames_dir datasets/video/raw_frames \
--data.labels_dir datasets/video/labels \
--data.num_workers 0
```

2. Test: Please pass the ckpt path if you have downloaded pretrained models
   Detect model:

```bash
python main.py -c configs/base_detect.yaml test \
  --trainer.accelerator cpu \
  --trainer.fast_dev_run True\
  --data.frames_dir datasets/video/raw_frames \
  --data.labels_dir datasets/video/labels \
  --data.num_workers 0 \
  --data.idle_gap 2 \
  --ckpt_path CKPT_PATH
```

Classifier model:

```bash
python main.py -c configs/base_clf.yaml test \
  --trainer.accelerator cpu \
  --trainer.fast_dev_run True\
  --data.frames_dir datasets/video/raw_frames \
  --data.labels_dir datasets/video/labels \
  --data.num_workers 0 \
  --ckpt_path CKPT_PATH
```

More details on all available options:

```bash
python main.py -c configs/base_clf.yaml [test or fit] -h
```

### Kaggle with free GPU:

1. Clone the repository, install requirements.

```
!git clone https://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@github.com/haily835/Keystroke-classifier.git
%cd Keystroke-classifier
!pip install -r requirements.txt

```

2. Train:
   Detect model:

```bash
python main.py -c configs/base_detect.yaml fit \
  --trainer.accelerator gpu \
  --trainer.devices 0,1 \
  --data.frames_dir /kaggle/input/single-setting/video/raw_frames \
  --data.labels_dir /kaggle/input/single-setting/video/labels \
  --data.idle_gap 2
```

Classifier model:

```bash
python main.py -c configs/base_clf.yaml fit \
--trainer.accelerator gpu \
--trainer.devices 0,1  \
--data.frames_dir /kaggle/input/single-setting/video/raw_frames \
--data.labels_dir /kaggle/input/single-setting/video/labels
```

2. Test on a single gpu
   Detect model:

```bash
python main.py -c configs/base_detect.yaml test \
  --trainer.accelerator gpu \
  --trainer.devices 0  \
  --data.frames_dir /kaggle/input/single-setting/video/raw_frames \
  --data.labels_dir /kaggle/input/single-setting/video/labels \
  --data.idle_gap 2 \
  --ckpt_path CKPT_PATH
```

Classifier model:

```bash
python main.py -c configs/base_clf.yaml test \
  --trainer.accelerator gpu \
  --trainer.devices 0  \
  --data.frames_dir /kaggle/input/single-setting/video/raw_frames \
  --data.labels_dir /kaggle/input/single-setting/video/labels \
  --ckpt_path CKPT_PATH
```

## Run the 2 stages on a video frames from pre-train model:

```bash
python test.py [-h] [--videos VIDEOS [VIDEOS ...]] [--data_dir DATA_DIR] [--clf_ckpt CLF_CKPT] [--det_ckpt DET_CKPT] [--result_dir RESULT_DIR]  -h, 

Options:
  --help            show this help message and exit
  --videos VIDEOS [VIDEOS ...]
                        List of video paths or a single video path.
  --data_dir DATA_DIR   Dataset directory
  --clf_ckpt CLF_CKPT   Path to the classifier checkpoint file.
  --det_ckpt DET_CKPT   Path to the detector checkpoint file.
  --result_dir RESULT_DIR
                        Directory to save the results.
```

```
!(python test.py --data_dir /kaggle/input/single-setting/video/raw_frames \
--clf_ckpt /kaggle/input/single-setting/pytorch/default/1/clf_epoch32-step57156.ckpt \
--det_ckpt /kaggle/input/single-setting/pytorch/default/1/detect_epoch14-step40935.ckpt)
```
## Record videos with label ground truth.

This script was used to record the video to train the model by opening the webcam. A phone camera can be used if webcams are not available. We install Camo on a smartphone and Camo client on MacOs as an alternative to a webcam. Note that the frame number of the keypress event has some delay. For example in our case, using the usb connection, the delay is 4. That means in the CSV file, user typing key A at frame 4 was recorded, however, the correct frame number is 4 + 4 = 8.

This issue will be further investigated, therefore for now you still need to recheck manually to observe the delay.

```
python ./utils/keystroke_recorder.py
```
