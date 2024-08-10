import cv2
import socket
import numpy as np
import csv
import time

def send_frames(host, port):
    cap = cv2.VideoCapture('http://192.168.0.58:4747/video')  # Open default camera (change index if multiple cameras)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        print(f"Connected to {host}:{port}")

        # CSV file setup
        day_time = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"label_{day_time}.csv"
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame Number', 'Key Press'])

            frame_number = -1

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from camera.")
                    break
                
                cv2.imshow('Frame', frame)

                # Record frame number and key press to CSV on key press 'q' for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key != 255:  # Check if any key is pressed
                    csv_writer.writerow([frame_number, chr(key)])
                    print([frame_number, chr(key)])

                # Encode frame as JPEG
                result, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not result:
                    print("Error: Failed to encode image.")
                    continue

                # Convert image data to bytes
                image_data = encoded_image.tobytes()
                image_size = len(image_data)

                # Send image size first
                s.sendall(image_size.to_bytes(8, 'big'))

                # Send image data
                s.sendall(image_data)

                frame_number += 1

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        s.close()

if __name__ == "__main__":
    host = '127.0.0.1'  # Localhost
    port = 65432  # Same port as sender
    send_frames(host, port)
