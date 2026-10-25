import argparse
import os
from glob import glob
import subprocess
from tqdm import tqdm

def get_image_files(image_folder):
    exts = ['jpg', 'jpeg', 'png', 'gif']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(image_folder, f'*.{ext}')))
        files.extend(glob(os.path.join(image_folder, f'*.{ext.upper()}')))
    return sorted(files)


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

def split_match(split_names, root):
    for name in split_names:
        if name in root:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Molmo2 Data as mp4 videos")
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    if args.split == "train":
        split_names = ["all", "train"]
    else:
        split_names = [args.split]

    out_path = os.path.join(args.out_dir, args.split, "videos")
    os.makedirs(out_path, exist_ok=True)

    in_path = os.path.join(args.in_dir, args.split, "JPEGImages")
    if not os.path.exists(in_path):
        # Try accessing dataset without split
        in_path = os.path.join(args.in_dir, "JPEGImages")
        if not os.path.exists(in_path):
            raise Exception(f"Input director doesnt exist {in_path}")


    videos = []
    for root, dirs, files in os.walk(in_path):
        if len(files) != 0 and files[0].split('/')[-1].split('.')[1] == 'jpg' and split_match(split_names, root):
            videos.append(root)
    videos = sorted(videos)

    for vid_idx, video in enumerate(tqdm(videos)):
        if vid_idx < args.start_idx:
            continue

        in_video_dir = video #os.path.join(in_path, video)
        out_video_path = video.replace(in_path, out_path) #os.path.join(out_path, video+".mp4")
        encode_frames_to_video(in_video_dir, out_video_path+".mp4", fps=6)



