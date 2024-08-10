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
        if not os.path.exists('./datasets/angle/raw_frames'):
            os.makedirs('./datasets/angle/raw_frames')
        if not os.path.exists('./datasets/angle/labels/'):
            os.makedirs('./datasets/angle/labels/')
        if not os.path.exists('./datasets/angle/ground_truths'):
            os.makedirs('./datasets/angle/ground_truths')

        files = os.listdir('./datasets/angle/raw_frames')
        indices = [int(file.split('_')[1].split('.')[0]) for file in files]
        video_index = 0 if len(indices) == 0 else max(indices) + 1

        self.label_path = f'./datasets/angle/labels/video_{video_index}.csv'
        self.ground_truth_path = f'./datasets/angle/ground_truths/video_{video_index}.txt'
        self.video_frames_path = f'./datasets/angle/raw_frames/video_{video_index}'
        self.labels = pd.DataFrame(columns=['Time', 'Key'])


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
       
        self.start_time = time.time_ns()
        
        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                self.frame_number += 1
                self.master.update()
                cv2.imwrite(f'{self.video_frames_path}/frame_{self.frame_number}.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:
                break
            
    def log_key_stroke(self, event):
        key = event.char if event.char else event.keysym  # Handling special keys
        if key == '\x08':  # Check if key is delete
            key = '[d]'
        elif key == ' ':  # Check if key is space
            key = '[s]'
        self.labels.loc[len(self.labels.index)] = [self.frame_number, key]

    def stop(self):
        self.recording = False
        self.cap.release()

    def save(self):
        self.labels.to_csv(self.label_path, index=False)
        text = self.text_entry.get("1.0", "end-1c")
        with open(self.ground_truth_path, "w") as f:
            f.write(text)

        
        print("Video length:", (time.time() - self.start_time) )
        print("- Typed text stored in:", self.ground_truth_path)
        print("- Video frames:", self.video_frames_path)
        print("- Labels of frames:", self.label_path)
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
