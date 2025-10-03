import os
import cv2
import mediapipe as mp
import numpy as np
import zarr


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Folders containing videos/images
DATA_FOLDERS = ['correct', 'incorrect']

# Map folder name to label
LABEL_MAP = {
    'correct': 1,
    'incorrect': 0
}
# Train or test
SPLIT = "train"

annotated_saved = False
all_samples = []
all_labels = []

for folder in DATA_FOLDERS:
    label = LABEL_MAP[folder]
    folder_path = os.path.join(os.getcwd(), "data", SPLIT, folder)
    
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        file_path = os.path.join(folder_path, file)

        if ext in IMAGE_EXTENSIONS:
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5) as pose:
                image = cv2.imread(file_path)
                if image is None:
                    continue
                image_height, image_width, _ = image.shape
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    continue

                frame_array = []
                for lm in results.pose_landmarks.landmark[11:]:
                    frame_array.append([lm.x, lm.y, lm.z, lm.visibility])

                # Pelvis as center
                left_hip = results.pose_landmarks.landmark[23]
                right_hip = results.pose_landmarks.landmark[24]
                pelvis = [
                    (left_hip.x + right_hip.x) / 2,
                    (left_hip.y + right_hip.y) / 2,
                    (left_hip.z + right_hip.z) / 2,
                    0.0
                ]
                frame_array.append(pelvis)

                frame_array = np.array(frame_array).T
                all_samples.append(frame_array[np.newaxis, :, np.newaxis, :])
                all_labels.append(label)

                # Save one annotated image for sanity check
                if not annotated_saved:
                    annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    cv2.imwrite('annotated_image.png', annotated_image)
                    annotated_saved = True

        elif ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(file_path)
            frames_list = []

            with mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break

                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    if not results.pose_landmarks:
                        continue

                    frame_array = []
                    for lm in results.pose_landmarks.landmark[11:]:
                        frame_array.append([lm.x, lm.y, lm.z, lm.visibility])

                    # Pelvis as center
                    left_hip = results.pose_landmarks.landmark[23]
                    right_hip = results.pose_landmarks.landmark[24]
                    pelvis = [
                        (left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2,
                        (left_hip.z + right_hip.z) / 2,
                        0.0
                    ]
                    frame_array.append(pelvis)

                    frame_array = np.array(frame_array).T[:, np.newaxis, :]
                    frames_list.append(frame_array)

            cap.release()

            if frames_list:
                video_array = np.concatenate(frames_list, axis=1)
                all_samples.append(video_array[np.newaxis, ...])
                all_labels.append(label)

# Pad each sample to have same number of frames
max_frames = max(sample.shape[2] for sample in all_samples)
padded_samples = []
for sample in all_samples:
    f = sample.shape[2]
    if f < max_frames:
        pad_width = ((0, 0), (0, 0), (0, max_frames - f), (0, 0))
        sample = np.pad(sample, pad_width, mode='constant', constant_values=0)
    padded_samples.append(sample)


# # Save to .npy files
# np.save('landmarks.npy', np.concatenate(padded_samples, axis=0))
# np.save('labels.npy', np.array(all_labels))
# print("Preprocessing complete. Saved landmarks.npy and labels.npy")

# Save to zarr file
landmarks = np.concatenate(padded_samples, axis=0)
labels = np.array(all_labels)
print(f"There are {landmarks.shape[0]} samples")
print(f"The maximum length is {landmarks.shape[2]}")
# with zarr.open(f'data/{SPLIT}_data.zarr', mode='w') as f:
#     f['landmark'] = landmarks
#     f['label'] = labels