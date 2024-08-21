import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import whisperx
from whisperx.utils import get_writer
import subprocess
import wave
import multiprocessing
    
def load_whisperx(model_name):
    global whisper_model
    whisper_model = whisperx.load_model(model_name, "cuda", compute_type="float16")

def run_whisperx(audio_file, output_dir, language=None, chunk_size=20, no_align=False):
    global whisper_model
    audio = whisperx.load_audio(audio_file)
    if language == "None":
        result = whisper_model.transcribe(audio=audio, chunk_size=chunk_size)
    else:
        result = whisper_model.transcribe(audio=audio, language=language, chunk_size=chunk_size)

    if not no_align:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)
    
    # whisperx align for some reason doesn't include langauge tag which is needed for get_writer in return for result
    if "language" not in result:
        result["language"] = language
    
    # Create output directory for each audio file
    audio_name = Path(audio_file).stem
    audio_output_dir = Path(output_dir) / audio_name
    os.makedirs(audio_output_dir, exist_ok=True)

    # Write SRT file
    srt_writer = get_writer("srt", audio_output_dir)
    srt_writer(result, audio_output_dir, {"max_line_width": None, "max_line_count": None, "highlight_words": False})
    
    # Generate audio clips based on the SRT file
    srt_file = audio_output_dir / f"{audio_name}.srt"
    generate_audio_clips(audio_file, srt_file, audio_output_dir)

def generate_audio_clips(audio_file, srt_file, output_dir):
    with open(srt_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    tasks = []
    for i in range(0, len(lines), 4):
        if len(lines[i:i+4]) < 4:
            continue
        
        time_line = lines[i+1].strip()
        start_time, end_time = time_line.split(" --> ")
        start_time_sec = srt_time_to_seconds(start_time)
        end_time_sec = srt_time_to_seconds(end_time)

        clip_output_path = output_dir / f"{Path(audio_file).stem}_clip_{i//4+1:03d}.wav"
        tasks.append((audio_file, start_time_sec, end_time_sec, clip_output_path))

    # Use multiprocessing to split the audio clips in parallel
    with multiprocessing.Pool() as pool:
        pool.starmap(extract_audio_clip, tasks)

def srt_time_to_seconds(srt_time):
    hours, minutes, seconds = map(float, srt_time.replace(",", ".").split(":"))
    return hours * 3600 + minutes * 60 + seconds

def extract_audio_clip(audio_file, start_time, end_time, output_path):
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

def split_clips(output_dir):
    """
    Split clips longer than 9 seconds into smaller clips.
    """
    for subdir in Path(output_dir).iterdir():
        if subdir.is_dir():
            for clip_file in subdir.glob("*.wav"):
                clip_duration = get_audio_duration(clip_file)
                if clip_duration > 9:
                    split_large_clip(clip_file, clip_duration)

def get_audio_duration(clip_file):
    """
    Get the duration of the audio clip in seconds.
    """
    with wave.open(str(clip_file), 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def split_large_clip(clip_file, clip_duration):
    """
    Split the clip into smaller clips no longer than 9 seconds.
    Replace the original clip with the smaller ones.
    """
    num_clips = int(clip_duration // 9)
    remainder = clip_duration % 9
    start_time = 0
    clip_name_base = Path(clip_file).stem
    
    new_clips = []
    for i in range(num_clips):
        end_time = start_time + 9
        new_clip_path = clip_file.parent / f"{clip_name_base}_part_{i+1:02d}.wav"
        extract_audio_clip(clip_file, start_time, end_time, new_clip_path)
        new_clips.append(new_clip_path)
        start_time = end_time
    
    if remainder > 0:
        new_clip_path = clip_file.parent / f"{clip_name_base}_part_{num_clips+1:02d}.wav"
        extract_audio_clip(clip_file, start_time, start_time + remainder, new_clip_path)
        new_clips.append(new_clip_path)
    
    # Remove the original clip
    clip_file.unlink()

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
    audio_files_to_process = [file for file in directory_path.iterdir() if file.is_file() and file.suffix.lower() in VALID_AUDIO_EXT]
    return audio_files_to_process

if __name__ == "__main__":

    
    whisper_model = None
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
    
    directory_path = select_directory()
    audio_files_list = get_files(directory_path)
    output_directory = "results"
    os.makedirs(output_directory, exist_ok=True)

    load_whisperx("large-v3")

    for file in audio_files_list:
        run_whisperx(file, output_directory)

    # Now split any long clips in the results
    split_clips(output_directory)
