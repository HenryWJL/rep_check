import av
import cv2
import click
import hydra
import torch
import numpy as np
from pathlib import Path


TASK_TO_CONFIG = {
    "squat": "train_rep_check_squat.yaml",
    "push_up": "train_rep_check_push_up.yaml"
}

LABEL_TO_CLASS = {
    0: "correct",
    1: "wrong"
}

@click.command(help="Evaluate models.")
@click.option("-t", "--task", type=click.Choice(["squat", "push_up"]), required=True, help="Task name.")
@click.option("-v", "--video", type=str, required=True, help="Video path.")
@click.option("-c", "--checkpoint", type=str, default="", help="Pretrained checkpoint.")
@click.option("-d", "--device", type=str, default="cuda", help="Device type.")
def main(task, video, checkpoint, device):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.parent.joinpath("rep_check", "configs")),
        version_base=None 
    ):
        cfg = hydra.compose(config_name=TASK_TO_CONFIG[task])
        device = torch.device(device)
        # Load model
        model = hydra.utils.instantiate(cfg.model)
        model.to(device)
        if Path(checkpoint).is_file():
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        # Load video
        container = av.open(video)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (320, 320))
            frames.append(img)
        frames = np.stack(frames)
        pred = model.predict(frames, device)
        print("Predicted class: ", LABEL_TO_CLASS[pred])


if __name__ == "__main__":
    main()