import cv2
import csv
import keyboard
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create CSV file for logging frame numbers and keys
    csv_filename = 'keypress_log.csv'
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame Number', 'Key Pressed', 'Timestamp'])
    
    # Create text file for logging typed text
    text_filename = 'typed_text_log.txt'
    with open(text_filename, mode='w') as text_file:
        text_file.write('Typed Text Log\n')
        text_file.write('-----------------\n')
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        frame_number += 1
        cv2.imshow('Webcam', frame)
        
        # Check if any key is pressed
        if keyboard.is_pressed():
            key_pressed = keyboard.read_event().name
            print(f"Key pressed: {key_pressed}")
            
            # Log to CSV
            with open(csv_filename, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([frame_number, key_pressed, datetime.now()])
            
            # Log to text file
            with open(text_filename, mode='a') as text_file:
                text_file.write(f"Frame {frame_number}: Key pressed - {key_pressed}\n")
        
        # Press 'q' on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
