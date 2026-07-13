import os
import subprocess

files = os.listdir("video")

for file in files:
    # Remove the file extension (.webm, .mp4, etc.)
    file_without_ext = os.path.splitext(file)[0]

    tutorial_number = file_without_ext.split(" [")[0].split(" #")[1]
    file_name = file_without_ext.split(" ｜ ")[0]

    output_file = f"audio/{tutorial_number}_{file_name}.mp3"

    print(output_file)

    subprocess.run([
        "ffmpeg",
        "-i", f"video/{file}",
        output_file
    ])