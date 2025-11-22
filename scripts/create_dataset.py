import os
import cv2
import mediapipe as mp
import numpy as np
import zarr
from scipy.interpolate import interp1d


# def resample_pose_sequence(poses, l):
#     T, J, C = poses.shape
#     old_t = np.arange(T)
#     new_t = np.linspace(0, T-1, l)

#     out = np.zeros((l, J, C), dtype=np.float32)
#     for j in range(J):
#         for c in range(C):
#             f = interp1d(old_t, poses[:, j, c], kind="linear")
#             out[:, j, c] = f(new_t)
#     return out


def resample_pose_sequence(poses, l):
    T, J, C = poses.shape
    if T == l:
        return poses
    elif T < l:
        out = np.zeros((l, J, C), dtype=poses.dtype)
        out[:T] = poses
        return out
    else:
        idx = np.round(np.linspace(0, T - 1, l)).astype(int)
        return poses[idx]


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Folders containing videos/images
DATA_FOLDERS = ['squat_correct', 'squat_wrong']

# Map folder name to label
LABEL_MAP = {
    class_name: i for i, class_name in enumerate(DATA_FOLDERS)
}
# Train or test
SPLIT = "test"

annotated_saved = False
all_samples = []
all_labels = []

for folder in DATA_FOLDERS:
    label = LABEL_MAP[folder]
    folder_path = os.path.join(os.getcwd(), "data", "squat_and_push_up_4_classes", SPLIT, folder)
    
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
                pelvis = np.array([
                    (left_hip.x + right_hip.x) / 2,
                    (left_hip.y + right_hip.y) / 2,
                    (left_hip.z + right_hip.z) / 2,
                    0.0
                ], dtype=float)
                frame_array = np.array(frame_array, dtype=float)

                # Center coords by subtracting by pelvis coords and set pevlis xyz coords to [0, 0, 0]
                frame_array[:, :3] -= pelvis[:3]
                pelvis_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
                frame_array = np.vstack([frame_array, pelvis_zero])

                frame_array = frame_array.T
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
                        frame_array.append(np.array([lm.x, lm.y, lm.z, lm.visibility], dtype=np.float32))

                    # Pelvis as center
                    left_hip = results.pose_landmarks.landmark[23]
                    right_hip = results.pose_landmarks.landmark[24]
                    pelvis = np.array([
                        (left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2,
                        (left_hip.z + right_hip.z) / 2,
                        0.0
                    ], dtype=np.float32)
                    frame_array.append(pelvis)
                    frame_array = np.stack(frame_array, axis=0)  # [num_joints, num_channels]

                    # Center coords by subtracting by pelvis coords and set pevlis xyz coords to [0, 0, 0]
                    frame_array[:, :3] -= pelvis[:3]
                    frames_list.append(frame_array)

            cap.release()

            if frames_list:
                video_array = np.stack(frames_list, axis=0)  # [num_frames, num_joints, num_channels]
                # video_array = resample_pose_sequence(video_array, 150)
                all_samples.append(video_array)
                all_labels.append(label)


# # Set  all data frames to 200
l = 200
processed_samples = []
for sample in all_samples:
    f = sample.shape[0]

    if f < l:
        # Zero pad
        pad_width = ((0, l - f), (0, 0), (0, 0))
        sample = np.pad(sample, pad_width, mode='constant', constant_values=0)

    elif f > l:
        # Resample down to 200 frames
        old_indices = np.linspace(0, f - 1, f)
        new_indices = np.linspace(0, f - 1, l)

        # Resample along the frame dimension
        num_dims = sample.shape[1] * sample.shape[2]
        reshaped = sample.reshape(f, num_dims).T  # flatten everything except frames
        resampled = np.array([np.interp(new_indices, old_indices, row) for row in reshaped])
        sample = resampled.T.reshape(l, sample.shape[1], sample.shape[2])

    processed_samples.append(sample)
all_samples = processed_samples

# Save to zarr file
landmarks = np.stack(all_samples, axis=0)
labels = np.array(all_labels)
print(f"Data shape: {landmarks.shape}")
with zarr.open(f'data/pose/{SPLIT}.zarr', mode='w') as f:
    f['landmark'] = landmarks
    f['label'] = labels
