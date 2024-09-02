import argparse
import glob
import torch
import pandas as pd
import torchvision
import os
from lightning_utils.dataset import clf_id2label, detect_id2label
from models import HyperFormer
from utils.convert_to_mediapipe import process_image

device = (
    "cuda" if torch.cuda.is_available() else "cpu"
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
        default='datasets/video-2/raw_frames',
    )

    parser.add_argument(
        '--window_size',
        type=int,
        help='Window size to scan',
        default=8,
    )

    parser.add_argument(
        '--clf_ckpt',
        type=str,
        help='Path to the classifier checkpoint file.',
        default='ckpts/hyperformer/clf-epoch=28-step=3161.ckpt',
        required=False
    )

    parser.add_argument(
        '--det_ckpt',
        type=str,
        help='Path to the detector checkpoint file.',
        default='ckpts/hyperformer/det-epoch=27-step=3752.ckpt',
        required=False
    )

    parser.add_argument(
        '--result_dir',
        default='./hyperformer_stream_results',
        type=str,
        help='Directory to save the results.',
        required=False
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def get_model_weight_from_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model_weights = checkpoint['state_dict']
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    return model_weights
def main():
    args = parse_arguments()

    # Access the arguments
    videos = args.videos
    data_dir = args.data_dir
    clf_ckpt = args.clf_ckpt
    det_ckpt = args.det_ckpt
    result_dir = args.result_dir
    window_size = args.window_size
    
    print(f"Data: {data_dir}")
    print(f"Videos: {videos}")
    print(f"Window size: {window_size}")
    print(f"Classifier checkpoint: {clf_ckpt}")
    print(f"Detector checkpoint: {det_ckpt}")
    print(f"Results will be saved in: {result_dir}")

    clf = HyperFormer(num_points=21, num_classes=30, num_of_heads=12)
    clf.load_state_dict(get_model_weight_from_ckpt(clf_ckpt))

    det = HyperFormer(num_points=21, num_classes=30, num_of_heads=12)
    det.load_state_dict(get_model_weight_from_ckpt(det_ckpt))
    
    clf.to(device)
    det.to(device)

    clf.eval()
    det.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for video_name in videos:
        video_path = f"{data_dir}/{video_name}"
        print('video_path: ', video_path)
        jpgs = sorted(glob.glob(f"{video_path}/*.jpg"))
        print(f"-----Video: {video_name}----")
        print('Total frames: ', len(jpgs))

        curr_frame = 0
        windows = []
        detect_record = []
        clf_record = []

        while curr_frame < len(jpgs):
            image = process_image(f"{video_path}/frame_{curr_frame}.jpg")

            if len(windows) < window_size:
                windows.append(image)
            else:
                frames = torch.stack(windows)
                frames = frames.permute(3, 0, 2, 1).float().unsqueeze(dim=0).to(device)

                detect_logits = torch.nn.functional.softmax(det(frames).squeeze(), dim=0)
                
                detect_id = torch.argmax(detect_logits, dim=0).item()
                
                detect_label = detect_id2label[detect_id]

                detect_record.append([curr_frame - window_size - 1, detect_logits[1].item()])

                if detect_label == 'active':
                    clf_logits = torch.nn.functional.softmax(clf(frames).squeeze(), dim=0)
                    pred_id = torch.argmax(clf_logits, dim=0).item()
                    clf_label = clf_id2label[pred_id]
                    print(f'{curr_frame - window_size - 1}-{curr_frame}: {clf_label} with probability {clf_logits[pred_id]}')
                    clf_record.append([curr_frame - window_size - 1, clf_label, clf_logits[pred_id].item()])

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