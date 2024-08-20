# Keystroke inference

## Training and testing

- keystroke detector to classify iddle and typing moments
- keystroke classifier to identify which key typed

1. Local machine with only cpu:

Detect model:

- python main.py -c configs/local_detect.yaml fit
- python main.py -c configs/local_detect.yaml test

Classifier model:

* python main.py -c configs/local.yaml fit
* python main.py -c configs/local.yaml test

2. Kaggle with free GPU:

- Clone the repository, install requirements.

```
!git clone https://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@github.com/haily835/Keystroke-classifier.git
%cd Keystroke-classifier
!pip install -r requirements.txt
```

Detect model:

- python main.py -c configs/kaggle_detect.yaml fit
- python main.py -c configs/kaggle_detect.yaml test

Classifier model:

* python main.py -c configs/kaggle_clf.yaml fit
* python main.py -c configs/kaggle_clf.yaml test

Run the 2 stages on a video frames from pre-train model:

```
python test.py [-h] [--videos VIDEOS [VIDEOS ...]] [--data_dir DATA_DIR] [--clf_ckpt CLF_CKPT] [--det_ckpt DET_CKPT] [--result_dir RESULT_DIR]
```

Options:

```
  -h, --help            show this help message and exit
  --videos VIDEOS [VIDEOS ...]
                        List of video paths or a single video path.
  --data_dir DATA_DIR   Dataset directory
  --clf_ckpt CLF_CKPT   Path to the classifier checkpoint file.
  --det_ckpt DET_CKPT   Path to the detector checkpoint file.
  --result_dir RESULT_DIR
                        Directory to save the results.
```

## Record videos with label ground truth.

This script was used to record the video to train the model by opening the webcam. A phone camera can be used if webcams are not available. We install Camo on a smartphone and Camo client on MacOs as an alternative to a webcam. Note that the frame number of the keypress event has some delay. For example in our case, using the usb connection, the delay is 4. That means in the CSV file, user typing key A at frame 4 was recorded, however, the correct frame number is 4 + 4 = 8.

This issue will be further investigated, therefore for now you still need to recheck manually to observe the delay.

```
python ./utils/keystroke_recorder.py
```
