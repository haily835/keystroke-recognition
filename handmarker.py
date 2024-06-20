import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
from tqdm import tqdm

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


  
def draw_landmarks_on_image(save_img_path, rgb_image, detection_result, saved_hand_img = False):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  all_handlandmarks = []
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    pos_3ds = []
    for landmark in hand_landmarks:
      pos_3ds.append([landmark.x, landmark.y, landmark.z])
    if len(pos_3ds) == 21:
      all_handlandmarks.append(pos_3ds)

  
  all_handlandmarks = np.array(all_handlandmarks)
  
  if not saved_hand_img: 
    return all_handlandmarks

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
  return all_handlandmarks

# videos = os.listdir('videos')
# print('videos: ', videos)

# mediapip detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

videos = ['video_7', 'video_8', 'video_9', 'video_10', 'video_11', 'video_12', 'video_13', 'video_14', 'video_15', 'video_16', 'video_0', 'video_1']
for video_name in videos:
  jpg_files = [file for file in os.listdir(f'./videos/{video_name}/') if file.endswith('.png')]
    
  # if not os.path.exists(f'./videos/{video_name}_handlandmarks'): os.makedirs(f'./videos/{video_name}_handlandmarks')

  save_3d_path = f'./videos/{video_name}.npy'
  all_landmarks = []
  missed = []
  for jpg in tqdm(jpg_files):
    image = mp.Image.create_from_file(f'./videos/{video_name}/{jpg}')
    detection_result = detector.detect(image)
    save_img_path = f'./videos/{video_name}_handlandmarks/{jpg}'
    
    landmarks = draw_landmarks_on_image(save_img_path, image.numpy_view(), detection_result)
    
    if (landmarks.shape[0] == 2 and landmarks.shape[1] == 21 and landmarks.shape[2] == 3):
      all_landmarks.append(landmarks)
    else: missed.append(jpg)

  all_landmarks = np.array(all_landmarks)
  print("Landmarks of all frames shape:", all_landmarks.shape, f". Missed: {missed} images")
  np.save(save_3d_path, all_landmarks)