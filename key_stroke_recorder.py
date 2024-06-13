import tkinter as tk
import pandas as pd
import os
import numpy as np
import time
import cv2

labels = pd.DataFrame(columns=['Frame', 'Key'])
frame_number = 0


def log_key_stroke(event):
    key = event.char if event.char else event.keysym  # Handling special keys
    if key == '\x08':  # Check if key is delete
        key = '[d]'
    elif key == ' ':  # Check if key is space
        key = '[space]'

    labels.loc[len(labels.index)] = [frame_number, key]

class KeyStrokeRecorder:
    def __init__(self, master, width = 320, height = 400, cam_url = 'http://192.168.0.58:4747/video'):
        self.start_time = 0
        ### UI
        self.master = master
        self.width = width
        self.height = height
        self.cam_url = cam_url
        self.text_entry = tk.Text(master)
        self.text_entry.pack()
        self.text_entry.bind("<Key>", log_key_stroke)

        self.canvas = tk.Canvas(master, width=width, height=height)
        self.canvas.pack()
        
        self.button = tk.Button(master, text="Record", command=self.record)
        self.button.pack()

        self.recording = False


        # Create directories if not exists
        if not os.path.exists('videos'):
            os.makedirs('videos')
        if not os.path.exists('labels'):
            os.makedirs('labels')
        if not os.path.exists('ground_truths'):
            os.makedirs('ground_truths')

        ### Prepare the folders, file name
        # Get the last video index
        files = os.listdir('videos')
        indices = [int(file.split('_')[1].split('.')[0]) for file in files if file.endswith('.mp4')]
        video_index = 0 if len(indices) == 0 else max(indices) + 1

        self.label_path = f'labels/video_{video_index}.csv'
        self.ground_truth_path = f'ground_truths/video_{video_index}.txt'
        self.video_frames_path = f'videos/video_{video_index}'
        self.video_landmarks_path = f'videos/video_landmarks_{video_index}'
        self.video_path = f'videos/video_{video_index}.mp4'

        if not os.path.exists(self.video_frames_path):
            os.makedirs(self.video_frames_path)
        if not os.path.exists(self.video_landmarks_path):
            os.makedirs(self.video_landmarks_path)

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (320, 320))

    
    def record(self):
        self.recording = True

       
        self.cap = cv2.VideoCapture(self.cam_url) 
        if not self.cap.isOpened():

            print('Can not connect to IP camera, using default webcam')
            self.cap = cv2.VideoCapture(0)


        self.button.config(text="Stop", command=self.stop)
        global frame_number
        self.start_time = time.time()

        while self.recording:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 320))
                cv2.imwrite(f'{self.video_frames_path}/frame_{frame_number}.png', frame)
                self.out.write(frame)
                frame_number += 1
    
                frame =tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.master.update()
                
            else:
                break

    def stop(self):
        self.recording = False
        self.cap.release()
        self.out.release()
        labels.to_csv(self.label_path, index=False)

        text = self.text_entry.get("1.0", "end-1c")
        with open(self.ground_truth_path, "w") as f:
            f.write(text)
        # self.hand_landmark()
        print("Video length:", (time.time() - self.start_time) )
        print("- Typed text stored in:", self.ground_truth_path)
        print("- Video recorded:", self.video_path)
        print("- Video frames:", self.video_frames_path)
        print("- Labels of frames:", self.label_path)
        # print("- Landmarks of frames:", self.video_landmarks_path)

        self.master.destroy()
            
def main():
    root = tk.Tk()
    root.title("KeyStroke Recorder")
    app = KeyStrokeRecorder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
