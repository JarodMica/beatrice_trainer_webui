'''
If a Dataset is already split into small audio segments, this does all of the data transformation needed for beatrice
'''

import os
import subprocess
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import multiprocessing

VALID_AUDIO_EXT = [
    ".mp3",   # MPEG Layer 3 Audio
    ".wav",   # Waveform Audio File Format
    ".aac",   # Advanced Audio Coding
    ".flac",  # Free Lossless Audio Codec
    ".ogg",   # Ogg Vorbis
    ".m4a",   # MPEG-4 Audio
    ".opus",  # Opus Audio Codec
    ".mp4"
]

def select_directory(title="Select Folder"):
    """
    Open a dialog to select a directory.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Set the window to be topmost
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(folder_selected)  # Convert to Path object

def get_files(directory_path):
    audio_files_to_process = [file for file in directory_path.rglob("*") if file.is_file() and file.suffix.lower() in VALID_AUDIO_EXT]
    return audio_files_to_process

def split_large_clip_task(task):
    audio_file, start_time, end_time, output_path = task
    extract_audio_clip(audio_file, start_time, end_time, output_path)

def split_audio_files(directory_path, output_directory):
    audio_files = get_files(directory_path)
    os.makedirs(output_directory, exist_ok=True)
    
    tasks = []
    
    for file in audio_files:
        clip_duration = get_audio_duration(file)
        num_clips = int(clip_duration // 9)
        remainder = clip_duration % 9
        start_time = 0
        clip_name_base = Path(file).stem
        
        for i in range(num_clips):
            end_time = start_time + 9
            new_clip_path = output_directory / f"{clip_name_base}_part_{i+1:02d}.wav"
            tasks.append((file, start_time, end_time, new_clip_path))
            start_time = end_time

        if remainder > 0:
            new_clip_path = output_directory / f"{clip_name_base}_part_{num_clips+1:02d}.wav"
            tasks.append((file, start_time, start_time + remainder, new_clip_path))

    # Process all tasks in parallel
    with multiprocessing.Pool() as pool:
        pool.map(split_large_clip_task, tasks)

def get_audio_duration(audio_file):
    """
    Get the duration of the audio file in seconds.
    """
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        print(f"Error retrieving duration for {audio_file}. Skipping file.")
        return 0

def extract_audio_clip(audio_file, start_time, end_time, output_path):
    """
    Extract a segment from the audio file.
    """
    original_sample_rate = get_audio_sample_rate(audio_file)
    original_codec = get_audio_codec(audio_file)
    
    duration = end_time - start_time
    command = [
        "ffmpeg", "-y", "-i", str(audio_file), "-ss", str(start_time),
        "-t", str(duration), "-acodec", original_codec, "-ar", str(original_sample_rate), str(output_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"Error processing {audio_file}, skipping file...")

def get_audio_sample_rate(audio_file):
    """
    Get the sample rate of the audio file.
    """
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        print(f"Error retrieving sample rate for {audio_file}, defaulting to 16000 Hz.")
        return 24000

def get_audio_codec(audio_file):
    """
    Get the codec of the audio file.
    """
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError:
        print(f"Error retrieving codec for {audio_file}, defaulting to pcm_s16le.")
        return "pcm_s16le"

if __name__ == "__main__":
    input_directory = select_directory("Select Folder Containing Audio Files")
    output_directory = input_directory.parent / f"{input_directory.stem}_split"

    split_audio_files(input_directory, output_directory)

    print(f"Audio files have been split and saved in: {output_directory}")
