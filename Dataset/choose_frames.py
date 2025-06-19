import os
import shutil
import sys

# Parse command-line arguments
seconds = int(sys.argv[1])
start = int(sys.argv[2])

# Frame indices to select: [1, 31, 61, ..., 301]
frames = range(start, seconds + 1)
num_frames = [i * 30 + 1 for i in frames]

# Walk through ./frames
for filepath, dirnames, filenames in os.walk('./frames'):
    filenames = sorted(filenames)
    
    if not filenames:
        continue

    # Get folder name (e.g., '1') from path like './frames/1'
    temp_name = os.path.basename(filepath)
    dst_dir = os.path.join('./choose_frames', temp_name)
    os.makedirs(dst_dir, exist_ok=True)

    for filename in filenames:
        if "checkpoint" in filename or "Store" in filename:
            continue

        if not filename.endswith('.jpg') or '_' not in filename:
            continue

        try:
            frame_number = int(filename.split('_')[1].split('.')[0])
        except ValueError:
            continue

        if frame_number in num_frames:
            srcfile = os.path.join(filepath, filename)
            dstfile = os.path.join(dst_dir, filename)
            shutil.copy(srcfile, dstfile)
