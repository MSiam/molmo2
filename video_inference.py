import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2TokenizerFast, AutoTokenizer
from PIL import Image
import requests
import torch
import os
import numpy as np
from glob import glob
from os.path import join, exists
import subprocess

def encode_frames_to_video(frames_dir, output_path, fps, native_fps:int=None,
                           start_frame=None, end_frame=None):
    """Convert a directory of image frames to an H.264 .mp4 using ffmpeg.

    Frames are first subsampled for FPS conversion (if native_fps differs from
    fps), then sliced to [start_frame, end_frame]. Frame indices refer to
    positions *after* subsampling.

    Args:
        frames_dir: Directory containing image frames (jpg/png).
        output_path: Path for the output .mp4 file.
        fps: Target framerate for the output video.
        native_fps: Original framerate of the frame sequence. When provided
            and different from fps, every round(native_fps / fps)-th frame
            is kept (e.g. native_fps=24, fps=6 keeps every 4th frame).
            None means all frames are used as-is.
        start_frame: First frame index to include (inclusive, 0-based,
            after subsampling). None means start from the beginning.
        end_frame: Last frame index to include (inclusive, 0-based,
            after subsampling). None means include through the last frame.
    """
    image_files = get_image_files(frames_dir)
    if not image_files:
        print(f"No frames in {frames_dir}")
        return None

    if native_fps is not None and native_fps != fps:
        subsample_factor = max(1, round(native_fps / fps))
        image_files = image_files[::subsample_factor]

    # pro
    end_frame = end_frame + 1 if end_frame is not None else None
    image_files = image_files[start_frame:end_frame]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write a concat demuxer file list for the selected frames
    filelist_path = output_path + ".filelist.txt"
    with open(filelist_path, 'w') as f:
        for img in image_files:
            f.write(f"file '{img}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", filelist_path,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

def get_image_files(image_folder):
    exts = ['jpg', 'jpeg', 'png', 'gif']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(image_folder, f'*.{ext}')))
        files.extend(glob(os.path.join(image_folder, f'*.{ext.upper()}')))
    return sorted(files)


checkpoint_dir = "allenai/Molmo2-8B"  # or path to a converted HF checkpoint

model = AutoModelForImageTextToText.from_pretrained(
    checkpoint_dir,
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(
    checkpoint_dir,
    tokenizer=tokenizer,
    trust_remote_code=True,
    padding_side="left",
    local_files_only=True
)

video_dir = "/media/ivulab/riemann/DAVIS_DATAEngine/JPEGImages/parkour/"
video_path = "./tmp/tmp.mp4"
encode_frames_to_video(video_dir, video_path, fps=30)
video_messages = [
    {
        "role": "user",
        "content": [
            dict(type="text", text="What the man is doing in this video?"),
            dict(type="video", video=video_path),
        ]
    }
]

inputs = processor.apply_chat_template(
    video_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    padding=True,
)

inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(**inputs, max_new_tokens=200)

generated_tokens = output[0, inputs['input_ids'].size(1):]
print(processor.decode(generated_tokens, skip_special_tokens=True))
