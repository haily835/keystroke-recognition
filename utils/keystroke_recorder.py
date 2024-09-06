import tkinter as tk
import pandas as pd
import os
import numpy as np
import time
import cv2
import keyboard
from PIL import Image
import shutil

droid_cam = 'http://192.168.0.58:4747/video'
dataset_name = 'topview-2'

class KeyStrokeRecorder:
    def __init__(self, master: tk.Tk, width = 640, height = 600, cam_url = 0):
        self.start_time = 0
        ### UI
        self.master = master
        self.width = width
        self.height = height
        self.cam_url = cam_url
        self.text_entry = tk.Text(master)
        self.text_entry.pack()
        self.frame_number = -1
        self.text_entry.bind("<Key>", self.log_key_stroke)    
        
        self.record_button = tk.Button(master, text="Record", command=self.record)
        self.record_button.pack()
        
        self.save_button = tk.Button(master, text="Save", command=self.save)
        self.save_button.pack()
        self.discard_button = tk.Button(master, text="Discard", command=self.discard)
        self.discard_button.pack()

        self.recording = False
        ### Prepare the folders, file name
        # Get the last video index
        if not os.path.exists(f'./datasets/{dataset_name}/raw_frames'):
            os.makedirs(f'./datasets/{dataset_name}/raw_frames')
        if not os.path.exists(f'./datasets/{dataset_name}/labels/'):
            os.makedirs(f'./datasets/{dataset_name}/labels/')
        if not os.path.exists(f'./datasets/{dataset_name}/ground_truths'):
            os.makedirs(f'./datasets/{dataset_name}/ground_truths')

        files = os.listdir(f'./datasets/{dataset_name}/raw_frames')
        indices = [int(file.split('_')[1].split('.')[0]) for file in files]
        video_index = 0 if len(indices) == 0 else max(indices) + 1

        self.info_path = f'./datasets/{dataset_name}/video_{video_index}_info.txt'
        self.label_path = f'./datasets/{dataset_name}/labels/video_{video_index}.csv'
        self.ground_truth_path = f'./datasets/{dataset_name}/ground_truths/video_{video_index}.txt'
        self.video_frames_path = f'./datasets/{dataset_name}/raw_frames/video_{video_index}'
        self.labels = pd.DataFrame(columns=['Frame', 'Key'])

    def record(self):
        self.recording = True
        self.cap = cv2.VideoCapture(self.cam_url) 

        if not self.cap.isOpened():
            print('Can not connect to IP camera, using default webcam')
            self.cap = cv2.VideoCapture(0)
        
        if not os.path.exists(self.video_frames_path):
            os.makedirs(self.video_frames_path)

        self.record_button.config(text="Stop", command=self.stop)
        self.text_entry.focus()
       
        self.start_time = time.time()
        
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.frame_number += 1
                self.master.update()
                frame = cv2.resize(frame, (640, 360))
                cv2.imwrite(f'{self.video_frames_path}/frame_{self.frame_number}.jpg', 
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:
                break
            
    def log_key_stroke(self, event):
        key = event.char if event.char else event.keysym  # Handling special keys
        
        if key == 'BackSpace':  # Check if key is delete
            key = 'delete'
        elif key == ' ':  # Check if key is space
            key = 'space'
        elif key == ',':
            key = 'comma'
        elif key == '.':
            key = 'dot'

        self.labels.loc[len(self.labels.index)] = [self.frame_number, key]

    def stop(self):
        self.recording = False
        self.cap.release()

    def save(self):
        self.labels.to_csv(self.label_path, index=False)
        text = self.text_entry.get("1.0", "end-1c")
        with open(self.ground_truth_path, "w") as f:
            f.write(text)
        
        video_length = time.time() - self.start_time
        print('video_length: ', video_length)
        total_frames = self.frame_number + 1
        print('total_frames: ', total_frames)
        total_words = len(text.split(' '))
        print('total_words: ', total_words)
        total_key_press = len(self.labels['Key'])
        print('total_key_press: ', total_key_press)
        cls_distribution = self.labels['Key'].value_counts()
        print('cls_distribution: ', cls_distribution)

        if (video_length//60):
            wpm = total_words // (video_length//60)
            print('wpm: ', wpm)
        
        with open(self.info_path, "w") as info_f:
            info_f.write(f'video_length: {video_length}\n')
            info_f.write(f'total_frames: {total_frames}\n')
            info_f.write(f'total_words: {total_words}\n')
            info_f.write(f'total_key_press: {total_key_press}\n')
            if (video_length//60):
                info_f.write(f'wpm: {wpm}\n')
            info_f.write(f'cls_distribution:\n{cls_distribution}')

        self.master.destroy()

    def discard(self):
        shutil.rmtree(self.video_frames_path)
        print(f"Deleted {self.video_frames_path}!")
        self.master.destroy()


def main():
    root = tk.Tk()
    root.title("KeyStroke Recorder")
    app = KeyStrokeRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()