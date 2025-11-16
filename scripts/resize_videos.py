import os
import subprocess

# -------- CONFIG --------
ROOT = "data/rep_check_dataset"                  # your root directory
TARGET_SIZE = (320, 320)       # resize to W x H (width, height)
VIDEO_EXT = ".mp4"
FFMPEG = "ffmpeg"
# -------------------------


def resize_and_rename_videos(root_dir):
    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            continue

        print(f"\nProcessing split: {split}")

        # Iterate through class folders
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue

            print(f"  Processing class: {cls}")

            videos = sorted(
                [f for f in os.listdir(cls_path) if f.lower().endswith(VIDEO_EXT)]
            )

            # Temp directory to avoid name conflicts
            tmp_dir = os.path.join(cls_path, "_tmp_processed")
            os.makedirs(tmp_dir, exist_ok=True)

            # Resize individual videos
            for idx, vid_name in enumerate(videos, start=1):
                input_path = os.path.join(cls_path, vid_name)

                # Format the new filename
                new_name = f"{idx:03d}{VIDEO_EXT}"
                output_path = os.path.join(tmp_dir, new_name)

                # ffmpeg command
                cmd = [
                    FFMPEG, "-y",
                    "-i", input_path,
                    "-vf", f"scale={TARGET_SIZE[0]}:{TARGET_SIZE[1]},setsar=1",
                    "-metadata:s:v", "rotate=0",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    output_path
                ]

                print(f"    → {new_name}")
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # Replace old videos with new ones
            for f in os.listdir(cls_path):
                path = os.path.join(cls_path, f)
                if os.path.isfile(path):
                    os.remove(path)

            # Move processed videos back
            for f in os.listdir(tmp_dir):
                os.rename(os.path.join(tmp_dir, f), os.path.join(cls_path, f))

            os.rmdir(tmp_dir)


if __name__ == "__main__":
    resize_and_rename_videos(ROOT)
    print("\nDone!")