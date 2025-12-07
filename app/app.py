import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import av
import cv2
import hydra
import numpy as np
import os
import streamlit as st
import torch
from torch.serialization import add_safe_globals
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from rep_check.utils.normalizer import Normalizer
from rep_check.models.rep_check import RepCheck


CONFIG_DIR = (Path(__file__).parent.parent / "rep_check" / "configs").resolve()
DEFAULT_CHECKPOINTS = {
    "squat": Path(__file__).parent.parent / "checkpoints" / "squat.pth",
    "push_up": Path(__file__).parent.parent / "checkpoints" / "push_up.pth",
}
MAX_WEBRTC_FRAMES = 300  # cap to avoid memory blowup (~10s at 30fps)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Reduce Mediapipe/TF/absl logging noise in the Streamlit console
os.environ.setdefault("GLOG_minloglevel", "2")  # suppress info/warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass


class FrameCollector(VideoProcessorBase):
    """Collect frames from the browser webcam via WebRTC."""

    def __init__(self):
        self.frames: List[np.ndarray] = []

    def recv(self, frame):  # type: ignore[override]
        img = frame.to_ndarray(format="rgb24")
        self.frames.append(img)
        if len(self.frames) > MAX_WEBRTC_FRAMES:
            self.frames.pop(0)
        return frame
TASK_TO_CONFIG = {
    "squat": "train_rep_check_squat.yaml",
    "push_up": "train_rep_check_push_up.yaml",
}
TASK_LABELS: Dict[str, Dict[int, str]] = {
    "squat": {0: "Correct rep", 1: "Needs work"},
    "push_up": {0: "Correct rep", 1: "Needs work"},
}
TASK_CUES: Dict[str, Dict[int, str]] = {
    "squat": {
        0: "Solid squat - hips stay balanced over mid-foot. Keep braced and stay consistent.",
        1: "Watch knees and depth: push them out over toes, keep heels down, and brace to avoid collapsing forward.",
    },
    "push_up": {
        0: "Smooth push-up - keep glutes tight and finish with elbows locked out.",
        1: "Keep a straight line from head to heels, elbows at ~45 deg, and avoid sagging hips.",
    },
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(task: str, checkpoint_path: str, device: str) -> RepCheck:
    config_name = TASK_TO_CONFIG[task]
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name=config_name)
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)
    # Checkpoint produced by CheckpointManager contains model + normalizer
    add_safe_globals([Normalizer])
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def read_video(path: Path, target_size: Tuple[int, int] = (320, 320)) -> np.ndarray:
    container = av.open(str(path))
    frames: List[np.ndarray] = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if target_size is not None:
            img = cv2.resize(img, target_size)
        frames.append(img)
    if not frames:
        raise ValueError("No frames found. Please upload a valid video.")
    return np.stack(frames)


def record_from_webcam(duration_sec: int = 5, fps: int = 24) -> np.ndarray:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions.")
    frames: List[np.ndarray] = []
    total_frames = duration_sec * fps
    for _ in range(total_frames):
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames captured from the webcam.")
    resized_frames = [cv2.resize(frame, (320, 320)) for frame in frames]
    return np.stack(resized_frames)


def webcam_frame_collector(frame):
    """Collect frames from the browser webcam via WebRTC."""
    img = frame.to_ndarray(format="rgb24")
    frames = st.session_state.setdefault("webcam_frames", [])
    frames.append(img)
    if len(frames) > MAX_WEBRTC_FRAMES:
        frames.pop(0)
    return frame


def save_video(frames: np.ndarray, fps: int = 24) -> str:
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = cv2.VideoWriter(tmp_file.name, fourcc, fps, (width, height))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    return tmp_file.name


def run_prediction(model: RepCheck, frames: np.ndarray, device: str) -> int:
    return model.predict(frames, device=torch.device(device))


def set_captured_video(frames: np.ndarray, fps: int = 24, preview_path: str | None = None) -> None:
    """Persist frames for inference and a preview path for playback."""
    st.session_state["video_frames"] = frames
    # Use provided preview (e.g., original upload) else save a temp mp4 of the processed frames
    if preview_path is None:
        preview_path = save_video(frames, fps=fps)
    st.session_state["video_preview_path"] = preview_path
    st.session_state["video_message"] = f"Captured {frames.shape[0]} frames"


def describe_class(task: str, pred_idx: int) -> str:
    labels = TASK_LABELS[task]
    cues = TASK_CUES[task]
    label = labels.get(pred_idx, f"Class {pred_idx}")
    cue = cues.get(pred_idx, "Keep working on consistent technique rep to rep.")
    return f"{label}: {cue}"


def main():
    st.set_page_config(page_title="RepCheck - Squat Coach", layout="wide")
    st.title("RepCheck: AI Rep Coach")
    st.write(
        "Upload a video or record with your camera to get instant feedback. "
        "The model tracks your pose, classifies the rep, and surfaces an actionable cue for the next rep."
    )

    # Initialize session storage for captured media
    st.session_state.setdefault("video_frames", None)
    st.session_state.setdefault("video_preview_path", None)
    st.session_state.setdefault("video_message", None)

    task = st.radio("Choose task", options=["squat", "push_up"], index=0, horizontal=True)
    labels = TASK_LABELS[task]

    default_ckpt = DEFAULT_CHECKPOINTS.get(task)
    default_ckpt_str = str(default_ckpt) if default_ckpt and default_ckpt.exists() else ""

    with st.sidebar:
        st.header("Session Settings")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=default_ckpt_str,
            key=f"ckpt_{task}",
            help="Path to the trained RepCheck checkpoint (.pth).",
        )
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        if device == "auto":
            device = _default_device()
        st.caption(f"Using device: **{device}**")
        st.markdown(
            "Classes:\n" + "\n".join([f"- {idx}: {name}" for idx, name in labels.items()])
        )

    input_mode = st.radio(
        "Choose input",
        options=["Upload video", "Use webcam"],
        help="Upload a clip or record via the browser camera.",
    )

    video_frames: np.ndarray | None = None
    video_preview_path: str | None = None

    if input_mode == "Upload video":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            try:
                frames = read_video(tmp_path)
                set_captured_video(frames, preview_path=str(tmp_path))
                st.success(f"Loaded {frames.shape[0]} frames from {uploaded.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read video: {exc}")
    elif input_mode == "Use webcam":
        st.info("Start the camera, perform the movement, then click 'Use captured clip'.")
        ctx = webrtc_streamer(
            key="repcheck-webrtc",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=FrameCollector,
            rtc_configuration=RTC_CONFIG,
        )

        frames = []
        if ctx and ctx.video_processor:
            frames = ctx.video_processor.frames

        st.caption(f"Frames buffered: {len(frames)} (buffer ~10s).")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use captured clip"):
                if not frames:
                    st.warning("No frames captured yet.")
                else:
                    video_frames = np.stack(frames)
                    set_captured_video(video_frames)
                    st.success(st.session_state["video_message"])
        with col2:
            if st.button("Clear buffer") and ctx and ctx.video_processor:
                ctx.video_processor.frames = []
                st.info("Cleared captured frames.")

    if st.session_state.get("video_preview_path"):
        preview_path = Path(st.session_state["video_preview_path"])
        if preview_path.exists():
            st.video(str(preview_path))
        else:
            st.warning("Preview not available (file missing). Try capturing again.")
        if st.session_state.get("video_message"):
            st.caption(st.session_state["video_message"])

    stored_frames = st.session_state.get("video_frames")

    if st.button("Run RepCheck", disabled=stored_frames is None or not checkpoint_path):
        if not checkpoint_path:
            st.warning("Please provide a checkpoint path.")
            return
        if stored_frames is None:
            st.warning("Please upload or record a video first.")
            return
        try:
            with st.spinner("Loading model..."):
                model = load_model(task, checkpoint_path, device)
            with st.spinner("Analyzing your movement..."):
                pred_idx = run_prediction(model, stored_frames, device)
            st.subheader("Result")
            st.success(describe_class(task, pred_idx))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run inference: {exc}")

    st.divider()
    st.caption(
        "Tips: aim for a clear side/45 deg angle, ensure the full body stays in frame, "
        "and keep lighting consistent for best pose tracking."
    )


if __name__ == "__main__":
    main()
