import argparse
import glob
import torch
import pandas as pd
import torchvision
from tqdm import tqdm
import os
from lightning_utils.module import KeyClf
from lightning_utils.dataset import clf_id2label, detect_id2label

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")

def parse_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process videos with specified models.")

    # Add arguments
    parser.add_argument(
        '--videos',
        type=str,
        nargs='+',  # Accept one or more values
        help='List of video paths or a single video path.',
        default= ['video_6', 'video_7'],
        
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Dataset directory',
        default='datasets',
    )


    parser.add_argument(
        '--clf_ckpt',
        type=str,
        help='Path to the classifier checkpoint file.',
        default='ckpts/single-setting/clf_epoch=32-step=57156.ckpt',
        required=False
    )

    parser.add_argument(
        '--det_ckpt',
        type=str,
        help='Path to the detector checkpoint file.',
        default='ckpts/single-setting/det_epoch=16-step=46393.ckpt',
        required=False
    )
    parser.add_argument(
        '--result_dir',
        default='./stream_results',
        type=str,
        help='Directory to save the results.',
        required=False
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    # Access the arguments
    videos = args.videos
    data_dir = args.data_dir
    clf_ckpt = args.clf_ckpt
    det_ckpt = args.det_ckpt
    result_dir = args.result_dir
    
    print(f"Data: {data_dir}")
    print(f"Videos: {videos}")
    print(f"Classifier checkpoint: {clf_ckpt}")
    print(f"Detector checkpoint: {det_ckpt}")
    print(f"Results will be saved in: {result_dir}")

    

    clf = KeyClf.load_from_checkpoint(clf_ckpt, 
                                      model_name='resnet101_clf',
                                      model_str='resnet101(sample_size=360, sample_duration=8, num_classes=len(clf_id2label))',
                                      id2label='clf_id2label',
                                      label2id='clf_label2id')
    det = KeyClf.load_from_checkpoint(det_ckpt, 
                                      model_name='resnet101_clf',
                                      model_str='resnet50(sample_size=360, sample_duration=8, num_classes=len(clf_id2label))',
                                      id2label='clf_id2label',
                                      label2id='clf_label2id'
                                      )
    clf.model.to(device)
    det.model.to(device)
    
    clf.freeze()
    det.freeze()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i in range(len(videos)):
        video_name = videos[i]
        video_path = f"{data_dir}/{video_name}"
        video_name = video_path.split('/')[-1]
        jpgs = sorted(glob.glob(f"{video_path}/*.jpg"))
        print(f"-----Video: {video_name}----")
        print('Total frames: ', len(jpgs))

        curr_frame = 0
        windows = []
        detect_record = []
        clf_record = []

        while curr_frame < tqdm(len(jpgs)):
            image = torchvision.io.read_image(f"{video_path}/frame_{curr_frame}.jpg")

            if len(windows) < 8:
                windows.append(image)

            else:
                frames = torch.stack(windows)
                frames = frames.permute(1, 0, 2, 3).unsqueeze(dim=0).to(device)

                detect_logits = torch.nn.functional.softmax(det.model(frames).squeeze())
                detect_id = torch.argmax(detect_logits, dim=0).item()
                detect_label = detect_id2label[detect_id]

                detect_record.append([curr_frame - 7, detect_logits[1].item()])

                if detect_label == 'active':
                    clf_logits = torch.nn.functional.softmax(clf.model(frames).squeeze(), dim=0)
                    pred_id = torch.argmax(clf_logits, dim=0).item()
                    clf_label = clf_id2label[pred_id]
                    print(f'{curr_frame - 7}-{curr_frame}: {clf_label} with probability {clf_logits[pred_id]}')
                    clf_record.append([curr_frame - 7, clf_label, clf_logits[pred_id].item()])

                windows = windows[1:]
            curr_frame += 1
        
        detect_df = pd.DataFrame({
            'Start frame': [record[0] for record in detect_record],
            'Active Prob': [record[1] for record in detect_record],
        })

        clf_df = pd.DataFrame({
            'Start frame': [record[0] for record in clf_record],
            'Key prediction': [record[1] for record in clf_record],
            'Prob': [record[2] for record in clf_record],
        })

        detect_df.to_csv(f'{result_dir}/{video_name}_detect.csv', index=False)
        clf_df.to_csv(f'{result_dir}/{video_name}_clf.csv', index=False)

if __name__ == "__main__":
    main()
