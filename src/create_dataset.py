import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import subprocess
import shutil
from pydub import AudioSegment

whisper_model = None

def select_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Set the window to be topmost
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(folder_selected)  # Convert to Path object

def parse_main_folder(main_folder_path):
    subfolders_list = [folder_name for folder_name in os.listdir(main_folder_path)]
    return subfolders_list

def convert_to_mono(audio_file):
    audio = AudioSegment.from_file(audio_file)
    mono_audio = audio.set_channels(1)
    mono_audio.export(audio_file, format="wav")
    
def load_whisperx(model_name):
    global whisper_model
    whisper_model = whisperx.load_model(model_name, "cuda", compute_type="float16")
    
# why whisperx?  Better than VAD, and no need to make a hugging face token for pyannote
def run_whisperx(audio_file, language=None):
    chunk_size = 10
    
    load_whisperx("large-v3")
    global whisper_model
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio=audio, chunk_size=chunk_size)
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)
    
    # whisperx align for some reason doesn't include langauge tag which is needed for get_writer in return for result
    if "language" not in result:
        result["language"] = language
    
    audio_output_dir = os.path.dirname(audio_file)
    srt_output_name = os.path.join(audio_output_dir, os.path.splitext(os.path.basename(audio_file))[0])
    
    # Write SRT file
    srt_writer = get_writer("srt", srt_output_name)
    srt_writer(result, srt_output_name, {"max_line_width": None, "max_line_count": None, "highlight_words": False})
        
def process_audio_files(folder_path, output_folder_name):
    base_folder_name = os.path.basename(folder_path)
    dest_dir = os.path.join(output_folder_name, base_folder_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    audio_list_to_process = []
    for file in os.listdir(folder_path):
        src_file_path = os.path.join(folder_path, file)
        dest_file_path = os.path.join(dest_dir, file)
        shutil.copy(src_file_path, dest_dir)
        audio_list_to_process.append(dest_file_path)
    
    #convert each file to mono
    import multiprocessing
    with multiprocessing.Pool() as pool:
        pool.map(convert_to_mono,audio_list_to_process)
    
    #run audio split with whisperx
    for file in audio_list_to_process:
        run_whisperx(file)
    
    #split audio file based on timings outputted from whisper
    
    #trim each audio segment, grouping to 9 seconds or less

if __name__ == "__main__":
    import whisperx
    from whisperx.utils import get_writer
    main_folder_path = select_folder()
    subfolders_list = parse_main_folder(main_folder_path)
    
    while True:
        try:
            output_folder_name = input("Give some name for the output folder: ")
            os.makedirs(output_folder_name, exist_ok=False)
            break
        except:
            print("Use a different name, folder already exists")
    
    for folder in subfolders_list:
        folder_path = os.path.join(main_folder_path, folder)
        process_audio_files(folder_path, output_folder_name)