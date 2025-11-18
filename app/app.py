import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch

from rep_check.models.gcn import STGCN
from rep_check.models.rep_check import RepCheck
from rep_check.utils.graph import MediaPipeGraph


SEQ_LEN = 150
CLASS_LABELS: Dict[int, str] = {
    0: "Clean rep",
    1: "Knee valgus",
    2: "Insufficient depth",
    3: "Forward lean / lumbar rounding / heel lift",
}

CLASS_CUES: Dict[int, str] = {
    0: "Nice! Keep knees tracking over toes, brace the core, and continue to drive evenly through mid-foot.",
    1: "Push knees over the toes and screw feet into the floor to avoid them caving in (valgus).",
    2: "Sit hips lower until the hip crease drops below the knee while keeping tension in the brace.",
    3: "Keep the chest proud, brace the core, and keep heels down as you drive up to avoid tipping forward or rounding.",
}


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: str) -> RepCheck:
    graph = MediaPipeGraph(num_node=23, max_hop=1, dilation=1, strategy="spatial")
    cls_model = STGCN(
        in_channels=4,
        num_classes=len(CLASS_LABELS),
        graph=graph,
        temporal_kernel_size=9,
        edge_weights=True,
        dropout=0.5,
    )
    model = RepCheck(cls_model=cls_model, seq_len=SEQ_LEN)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def read_video(path: Path, target_size: Tuple[int, int] = (640, 360)) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
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
    height, width, _ = frames[0].shape
    resized_frames = [
        cv2.resize(frame, (width, height)) for frame in frames
    ]
    return np.stack(resized_frames)


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


def describe_class(pred_idx: int) -> str:
    label = CLASS_LABELS.get(pred_idx, f"Class {pred_idx}")
    cue = CLASS_CUES.get(pred_idx, "Keep working on consistent technique rep to rep.")
    return f"{label}: {cue}"


def main():
    st.set_page_config(page_title="RepCheck - Squat Coach", layout="wide")
    st.title("RepCheck: AI Squat Form Coach")
    st.write(
        "Upload a squat video or record with your camera to get instant feedback. "
        "The model tracks your pose, classifies the rep, and surfaces an actionable cue for the next rep."
    )

    with st.sidebar:
        st.header("Session Settings")
        checkpoint_path = st.text_input(
            "Checkpoint path",
            help="Path to the trained RepCheck checkpoint (.pth).",
        )
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        if device == "auto":
            device = _default_device()
        st.caption(f"Using device: **{device}**")
        st.markdown(
            "Classes:\n"
            f"- 0: {CLASS_LABELS[0]}\n"
            f"- 1: {CLASS_LABELS[1]}\n"
            f"- 2: {CLASS_LABELS[2]}\n"
            f"- 3: {CLASS_LABELS[3]}"
        )

    input_mode = st.radio(
        "Choose input",
        options=["Upload video", "Use webcam (5s capture)"],
        help="Upload a clip or let RepCheck capture 5 seconds from your camera.",
    )

    video_frames: np.ndarray | None = None
    video_preview_path: str | None = None

    if input_mode == "Upload video":
        uploaded = st.file_uploader("Upload a squat video", type=["mp4", "mov", "avi"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            try:
                video_frames = read_video(tmp_path)
                video_preview_path = str(tmp_path)
                st.success(f"Loaded {video_frames.shape[0]} frames from {uploaded.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read video: {exc}")
    else:
        st.info("Click the button to capture a 5-second clip from your camera.")
        duration = st.slider("Capture duration (seconds)", min_value=3, max_value=10, value=5, step=1)
        fps = st.slider("Capture FPS", min_value=15, max_value=30, value=24, step=1)
        if st.button("Record"):
            try:
                with st.spinner("Recording from webcam..."):
                    video_frames = record_from_webcam(duration_sec=duration, fps=fps)
                video_preview_path = save_video(video_frames, fps=fps)
                st.success(f"Captured {video_frames.shape[0]} frames from webcam.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Webcam capture failed: {exc}")

    if video_preview_path:
        st.video(video_preview_path)

    if st.button("Run RepCheck", disabled=video_frames is None or not checkpoint_path):
        if not checkpoint_path:
            st.warning("Please provide a checkpoint path.")
            return
        if video_frames is None:
            st.warning("Please upload or record a video first.")
            return
        try:
            with st.spinner("Loading model..."):
                model = load_model(checkpoint_path, device)
            with st.spinner("Analyzing your squat..."):
                pred_idx = run_prediction(model, video_frames, device)
            st.subheader("Result")
            st.success(describe_class(pred_idx))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run inference: {exc}")

    st.divider()
    st.caption(
        "Tips: aim for a clear side/45° angle, ensure the full body stays in frame, "
        "and keep lighting consistent for best pose tracking."
    )


if __name__ == "__main__":
    main()
