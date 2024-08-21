import os
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
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

def get_audio_files(directory_path):
    """
    Recursively get all audio files in the directory and its subdirectories.
    """
    audio_files = [file for file in directory_path.rglob("*") if file.suffix.lower() in VALID_AUDIO_EXT]
    return audio_files

def convert_to_mono(audio_file):
    temp_output_path = audio_file.with_suffix(".tmp.wav")
    
    command = [
        "ffmpeg", "-y", "-i", str(audio_file), "-ac", "1", str(temp_output_path)
    ]
    
    try:
        subprocess.run(command, check=True)
        os.replace(temp_output_path, audio_file)  # Replace original file with the mono version
    except subprocess.CalledProcessError:
        print(f"Error converting {audio_file} to mono, skipping file...")
        if temp_output_path.exists():
            temp_output_path.unlink()  # Remove temporary file if conversion fails

if __name__ == "__main__":
    results_directory = select_directory("Select the 'results' Folder")
    audio_files_list = get_audio_files(results_directory)
    with multiprocessing.Pool() as pool:
        pool.map(convert_to_mono, audio_files_list)

    print(f"All audio files in '{results_directory}' have been converted to mono.")
