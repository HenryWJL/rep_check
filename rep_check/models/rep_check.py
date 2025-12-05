import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from einops import rearrange
from typing import Dict
from rep_check.models.gcn import STGCN
from rep_check.utils.normalizer import Normalizer


class RepCheck(nn.Module):

    def __init__(
        self,
        cls_model: STGCN,
        seq_len: int
    ) -> None:
        super().__init__()
        self.cls_model = cls_model
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.normalizer = Normalizer()
        self.seq_len = seq_len

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def forward(self, pose_landmarks: torch.Tensor) -> torch.Tensor:
        batch_size = pose_landmarks.shape[0]
        pose_landmarks = self.normalizer.normalize(rearrange(pose_landmarks, 'b c t v -> (b t v) c'))
        pose_landmarks = rearrange(pose_landmarks, "(b t v) c -> b c t v", b=batch_size, t=self.seq_len)
        return self.cls_model(pose_landmarks)
    
    def predict(self, video: np.ndarray, device: torch.device) -> int:
        """
        Args:
            video: (T, H, W, C)
        """
        # Model prediction
        self.eval()
        pose_landmarks = []
        for frame in video:
            results = self.mp_pose.process(frame)
            if results.pose_landmarks:
                pose_landmark = []
                for lm in results.pose_landmarks.landmark[11:]:
                    pose_landmark.append(np.array([lm.x, lm.y, lm.z, lm.visibility]))
                # Pelvis as center
                left_hip = results.pose_landmarks.landmark[23]
                right_hip = results.pose_landmarks.landmark[24]
                pelvis = np.array([
                    (left_hip.x + right_hip.x) / 2,
                    (left_hip.y + right_hip.y) / 2,
                    (left_hip.z + right_hip.z) / 2,
                    0.0
                ])
                pose_landmark.append(pelvis)
                pose_landmark = np.stack(pose_landmark)
                # Center coords by subtracting by pelvis coords
                pose_landmark[:, :3] -= pelvis[:3]
                pose_landmarks.append(pose_landmark)
            else:
                # If no pose detected, fill zeros
                pose_landmarks.append(np.zeros((23, 4), dtype=np.float32))
        pose_landmarks = np.stack(pose_landmarks)
        # Resample to a fixed length
        len = pose_landmarks.shape[0]
        if len < self.seq_len:
            # Zero pad
            pad_width = ((0, self.seq_len - len), (0, 0), (0, 0))
            pose_landmarks = np.pad(pose_landmarks, pad_width, mode='constant', constant_values=0)
        elif len > self.seq_len:
            # Downsample
            old_indices = np.linspace(0, len - 1, len)
            new_indices = np.linspace(0, len - 1, self.seq_len)
            flattened = pose_landmarks.flatten(1)
            downsampled = np.array([np.interp(new_indices, old_indices, row) for row in flattened])
            pose_landmarks = downsampled.reshape(self.seq_len, *pose_landmarks.shape[1:])

        pose_landmarks = torch.from_numpy(pose_landmarks).float().to(device)
        pose_landmarks = rearrange(pose_landmarks, "t v c -> 1 c t v")
        with torch.no_grad():
            logits = self.forward(pose_landmarks)
            pred = torch.argmax(logits, dim=1).item()
            # pred_prob = torch.sigmoid(logits)
            # pred = (pred_prob > 0.5).long()
        return pred

    def load_state_dict(self, state_dict) -> None:
        self.cls_model.load_state_dict(state_dict['model'])
        self.normalizer = state_dict['normalizer']

    def state_dict(self) -> Dict:
        return dict(
            model=self.cls_model.state_dict(),
            normalizer=self.normalizer
        )