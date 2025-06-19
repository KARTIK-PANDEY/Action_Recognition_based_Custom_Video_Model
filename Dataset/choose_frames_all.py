import os
import shutil
import sys

# Get total video length and start second from command-line arguments
seconds = int(sys.argv[1])
start = int(sys.argv[2])

# List of frame numbers to extract (1 frame every 30 frames, i.e. every second)
frame_indices = [(i * 30 + 1) for i in range(start, seconds + 1)]

# Output directory
output_dir = os.path.join('.', 'choose_frames_all')
os.makedirs(output_dir, exist_ok=True)

# Walk through the frames directory
for root, dirs, files in os.walk('./frames'):
    files = sorted(files)
    video_id = os.path.basename(root)

    for file in files:
        if "checkpoint" in file or "Store" in file or not file.endswith(".jpg"):
            continue

        try:
            frame_number = int(file.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            continue  # Skip malformed filenames

        if frame_number in frame_indices:
            new_filename = f"{video_id}_{frame_number:06d}.jpg"
            src_path = os.path.join(root, file)
            dst_path = os.path.join(output_dir, new_filename)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Source file not found: {src_path}")
