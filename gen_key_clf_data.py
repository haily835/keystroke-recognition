
import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import threading


fps = 30
width, height = 320, 320

alphabet = np.array(['[i]','[s]', 'BackSpace', ',', '.',
                        'a', 'b', 'c', 'd', 'e', 'f', 
                        'g', 'h', 'i', 'j', 'k', 'l', 
                        'm', 'n', 'o', 'p', 'q', 'r', 
                        's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z'])

def resize_image(image_path, target_size=(112, 112)):
    image = Image.open(image_path)
    old_size = image.size
    # Calculate new size and position for centering
    new_size = (320,320)
    left = (new_size[0] - old_size[0]) // 2
    top = (new_size[1] - old_size[1]) // 2
    right = left + old_size[0]
    bottom = top + old_size[1]
    # Create a new image with centered content
    centered_image = Image.new('RGB', new_size)
    centered_image.paste(image, (left, top, right, bottom))
    
    # Resize to target size
    resized_image = centered_image.resize(target_size)
    
    return resized_image


def video_segments(video_name, destination_base):
    # Assuming CSV file structure: Frame, Key
    csv_file = f'./labels/{video_name}.csv'  # Update with your CSV file path
    source_folder = f'./raw_frames/{video_name}'  # Update with your source folder containing video folders
    
    # Read CSV into pandas DataFrame
    df = pd.read_csv(csv_file)

    # Iterate over each row in the CSV
    for index, row in df.iterrows():
        key_frame = int(row['Frame'])  # Frame number where key was pressed
        key_value = row['Key']  # Key pressed

        if key_value not in alphabet: continue
        if key_value == '[s]': key_value = 'Space'
        if key_value == ',': key_value = 'Comma'
        if key_value == '.': key_value = 'Stop'

        destination_key_folder = os.path.join(destination_base, key_value)
        # Create destination key folder if it doesn't exist
        os.makedirs(destination_key_folder, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        destination_path = os.path.join(destination_key_folder, f'{video_name}_{key_frame - 3}_{key_frame + 5}.mp4')
        out = cv2.VideoWriter(destination_path, fourcc, fps, (width, height), isColor=True)
        
        # Copy 8 frames (3 before key_frame, key_frame itself, 4 after key_frame) and double it
        for i in range(key_frame - 3, key_frame + 5):
            source_frame_path = os.path.join(source_folder, f'frame_{i}.png')
            # Check if the frame file exists before copying
            if os.path.exists(source_frame_path):
                resized_image = np.array(resize_image(source_frame_path, target_size=(width, height)))
                # Save the resized image
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR) 
                # out.write(resized_image)
                out.write(resized_image)
            else:
                print(f"Warning: Frame {i} not found for video {video_name}")
        out.release()
    print(f"Dataset creation complete for {video_name}")


if __name__ == '__main__':
    mode = 'test'

    splits = {
        'train': ['video_1', 'video_2', 'video_3', 'video_4', 'video_5', 'video_6', 
                  'video_7', 'video_8', 'video_9', 'video_10', 'video_11', 'video_12',
                  'video_13', 'video_14', 'video_15', 'video_16', 'video_17', 'video_18',
                  'video_19', 'video_20', 'video_21', 'video_22', 'video_23', 'video_24',
                  'video_25', 'video_26', 'video_27'],
        'val': ['video_28', 'video_29', 'video_30', 'video_31', 'video_32'],
        'test': ['video_33', 'video_34', 'video_35', 'video_36'],
    }

    # splits = {
    #     'train': ['video_30'],
    #     'val': ['video_32'],
    # }

    videos = splits[mode]
    print('Create classify videos: ', videos)
    destination_base = f'./datasets/key_clf_data_{width}_{height}/{mode}'
    print('Destination: ', destination_base)
    os.makedirs(destination_base, exist_ok=True)

    threads = []
    for video_name in videos:
        threads.append(threading.Thread(target=video_segments, args=(video_name, destination_base)))
    
    for t in threads: t.start()
    for t in threads: t.join()
