import os
import pandas as pd
from PIL import Image
from keystroke_classify_dataset import resize_image
import cv2
import numpy as np
from tqdm import tqdm
import threading

# Paths and constants
fps = 30
width, height = 224, 224


def video_segments(video_name, destination_base):
    pos_dest_path= os.path.join(destination_base, 'positive')
    neg_dest_path = os.path.join(destination_base, 'negative')

    os.makedirs(pos_dest_path, exist_ok=True)
    os.makedirs(neg_dest_path, exist_ok=True)

    # Read CSV into pandas DataFrame
    csv_file = f'./labels/{video_name}.csv'
    source_folder = f'./videos/{video_name}'
    df = pd.read_csv(csv_file)


    for index, row in df.iterrows():
        key_frame = int(row['Frame'])  # Frame number where key was pressed
        
        if index == 0:
            pos_start = key_frame - 3 if (key_frame - 3 >= 0) else 0
            pos_end = key_frame + 4

            neg_start = 0
            neg_end = pos_start - 1

        # if index < len(df) - 1:
        #     next_key_frame = int(df.loc[index + 1, 'Frame'])
        #     negative_start = key_frame + 1
        #     negative_end = next_key_frame - 16 if next_key_frame - 16 > key_frame else key_frame
        #     # Ensure that the negative segment is exactly 16 frames
        #     negative_end = negative_start + 15
        else:
            # Last key frame, process remaining frames as negative
            prev_key_frame = df.iloc[index - 1]['Frame']
            pos_start = key_frame - 3 if (key_frame - 3 >= 0) else 0 
            pos_end = key_frame + 4
            prev_pos_window_end = prev_key_frame + 4
            if (pos_start - prev_pos_window_end) - 1 >= 8:
                neg_start =  prev_pos_window_end + 1
                neg_end = pos_start - 1
        
        # Positive class video segment
        pos_subfolder = f'{video_name}_{pos_start}_{pos_end}.mp4'
        pos_subfolder_path = os.path.join(pos_dest_path, pos_subfolder)
        # os.makedirs(pos_subfolder_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(pos_subfolder_path, fourcc, fps, (width, height), isColor=True)

        for i in range(pos_start, pos_end + 1):
            source_frame_path = os.path.join(source_folder, f'frame_{i}.png')
            # destination_frame_path_1 = os.path.join(pos_subfolder_path, f'frame_{i}_1.png')
            # destination_frame_path_2 = os.path.join(pos_subfolder_path, f'frame_{i}_2.png')
            
            # Check if the frame file exists before copying
            if os.path.exists(source_frame_path):
                resized_image = np.array(resize_image(source_frame_path, target_size=(width, height)))
                # Save the resized image
                # resized_image.save(destination_frame_path_1)
                # resized_image.save(destination_frame_path_2)
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR) 
                out.write(resized_image)
                out.write(resized_image)
            else:
                print(f"Warning: Frame {i} not found for video {video_name}")
        out.release()

        # Negative class video segment
        j = neg_end 
        while j - 7 >= neg_start:

            neg_subfolder = f'{video_name}_{j - 7}_{j}.mp4'
            neg_subfolder_path = os.path.join(neg_dest_path, neg_subfolder)
            # os.makedirs(neg_subfolder_path, exist_ok=True)
            out = cv2.VideoWriter(neg_subfolder_path, fourcc, fps, (width, height))
            for i in range(j - 7, j + 1):
                source_frame_path = os.path.join(source_folder, f'frame_{i}.png')
                # destination_frame_path_1 = os.path.join(neg_subfolder_path, f'frame_{i}_1.png')
                # destination_frame_path_2 = os.path.join(neg_subfolder_path, f'frame_{i}_2.png')
                
                # Check if the frame file exists before copying
                if os.path.exists(source_frame_path):
                    resized_image = np.array(resize_image(source_frame_path, target_size=(width, height)))
                    # Save the resized image
                    # resized_image.save(destination_frame_path_1)
                    # resized_image.save(destination_frame_path_2)
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR) 
                    out.write(resized_image)
                    out.write(resized_image)
                else:
                    print(f"Warning: Frame {i} not found for video {video_name}")
            out.release()
            j -= 8
    print(f"Formatted keystroke segments for {video_name} creation complete.")


def create_full_dataset(splits, base_folder):
    for mode, videos in splits.items():
        print('Videos: ', videos)
        destination_base = f'./{base_folder}/{mode}'
        print('Destination: ', destination_base)

        threads = []
        for video_name in videos:
            threads.append(threading.Thread(target=video_segments, args=(video_name,destination_base)))
        for t in threads: t.start()
        for t in threads: t.join()

if __name__ == "__main__":
    mode = 'train'

    splits = {
        'train': ['video_0', 'video_1', 'video_2', 'video_3', 'video_4', 'video_5', 'video_6', 'video_7', 'video_8', 'video_9', 'video_10', 'video_11', 'video_12'],
        'val': ['video_13', 'video_14'],
        'test': ['video_15', 'video_16'],
    }

    subset_splits = {
        'train': ['video_2', 'video_3'],
        'val': ['video_0'],
        'test': ['video_1'],
    }

    create_full_dataset(subset_splits, 'keystroke_detect_subset')