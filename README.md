# RepCheck

Pose-based rep quality classifier using ST-GCN. A Streamlit UI is included to run the trained model on an uploaded clip or a quick webcam capture.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the UI

```bash
streamlit run app/app.py
```

- Pick a task in the UI (squat or push_up).
- Point the checkpoint path to your trained weight file. By default the app looks for `checkpoints/squat.pth` and `checkpoints/push_up.pth`.
- Upload a video (`.mp4/.mov/.avi`) or capture ~5 seconds from the webcam, then click **Run RepCheck**.

Tips: keep your whole body in frame, use side or 45 degree angles, and consistent lighting for smoother pose tracking.
