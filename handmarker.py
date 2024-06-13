import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def hand_landmark(image):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    landmarks = []
    detection_result = detector.detect(image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    return detection_result
    
def draw_landmarks_on_image(save_img_path, save_3d_path, rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  all_handlandmarks = []
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    pos_3ds = []
    for landmark in hand_landmarks:
      pos_3ds.append([landmark.x, landmark.y, landmark.z])

    all_handlandmarks.append(pos_3ds)

  
  all_handlandmarks = np.array(all_handlandmarks)
  print(all_handlandmarks.shape)
  np.save(save_3d_path, all_handlandmarks)


  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]


    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    cv2.imwrite(save_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


video_name = 'video_4'

jpg_files = [file for file in os.listdir(f'./videos/{video_name}/') if file.endswith('.png')]
if not os.path.exists(f'./videos/{video_name}_handlandmarks'): os.makedirs(f'./videos/{video_name}_handlandmarks')


for jpg in jpg_files:
  image = mp.Image.create_from_file(f'./videos/{video_name}/{jpg}')
  detection_result = hand_landmark(image)
  save_img_path = f'./videos/{video_name}_handlandmarks/{jpg}'
  save_3d_path = f'./videos/{video_name}_handlandmarks/{jpg}.npy'
  draw_landmarks_on_image(save_img_path, save_3d_path, image.numpy_view(), detection_result)