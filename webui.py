import os
import sys
import shutil

import gradio as gr
import webbrowser
import socket
import tqdm

from multiprocessing import Pool, cpu_count
import pysrt
from pydub import AudioSegment

def get_port_available(start_port=7860, end_port=7865):
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) == 0
    webui_port = None         
    while webui_port == None:
        for i in range (start_port, end_port):
            if is_port_in_use(i):
                print(f"Port {i} is in use, moving 1 up")
            else:
                webui_port = i
                break
    return webui_port

def is_correct_dataset_structure(folder_to_analyze):
    for item in os.listdir(folder_to_analyze):
        path_to_item = os.path.join(folder_to_analyze, item)
        if os.path.isdir(path_to_item):
            pass
        else:
            return False
    return True

def get_available_datasets(root="datasets"):
    list_of_datasets = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root,folder))]
    return list_of_datasets

def folder_to_process_proxy(folder_to_analyze):
    folder_check = is_correct_dataset_structure(folder_to_analyze)
    if folder_check==False:
        raise gr.Error("Please check the folder structure and make sure it contains ONLY folders")
    return gr.Dropdown(value=folder_to_analyze)

def load_whisperx(model_name):
    global whisper_model
    whisper_model = whisperx.load_model(model_name, "cuda", compute_type="float16")

def run_whisperx_transcribe(audio_file_path, chunk_size=15, language=None):
    global whisper_model
    audio = whisperx.load_audio(audio_file_path)
    result = whisper_model.transcribe(audio=audio, chunk_size=chunk_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda", return_char_alignments=False)
    
    # whisperx align for some reason doesn't include langauge tag which is needed for get_writer in return for result
    if "language" not in result:
        result["language"] = language
    
    return result

def run_whisperx_srt(transcription_result, output_directory):
    srt_writer = get_writer("srt", output_directory)
    srt_writer(transcription_result, output_directory, {"max_line_width": None, "max_line_count": None, "highlight_words": False})

def process_speaker_folder(file_info, progress_bar=None):
    folder_path, audio_file, srt_file = file_info

    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1) # Mono conversion
    subs = pysrt.open(srt_file)

    base_name = os.path.basename(folder_path)
    segment_counter = 1 

    for idx, sub in enumerate(tqdm.tqdm(subs, desc="Processing Subtitles", leave=False, file=sys.stdout)):
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
        duration = end_time - start_time

        max_segment_duration = 8000  

        while duration > max_segment_duration:
            segment_end_time = start_time + max_segment_duration
            segment = audio[start_time:segment_end_time]
            output_file = f"{folder_path}/{base_name}_{segment_counter}.wav"
            segment.export(output_file, format="wav")
            start_time = segment_end_time
            duration = end_time - start_time
            segment_counter += 1

        if duration > 0:
            segment = audio[start_time:end_time]
            output_file = f"{folder_path}/{base_name}_{segment_counter}.wav"
            segment.export(output_file, format="wav")
            segment_counter += 1

        if progress_bar:
            progress_bar.update(1)

    os.remove(audio_file)
    os.remove(srt_file)

def split_by_srt(folder_path, progress_bar=None):
    file_pairs = []
    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.mp3', '.m4a')): 
            audio_file = os.path.join(folder_path, file)
            srt_file = os.path.join(folder_path, file.rsplit('.', 1)[0] + '.srt')
            if os.path.exists(srt_file):
                file_pairs.append((folder_path, audio_file, srt_file))

    with Pool(cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_speaker_folder, file_pairs), total=len(file_pairs), desc="Processing Files", file=sys.stdout))


def process_proxy(folder_to_process_path, progress = gr.Progress(track_tqdm=True)):
    training_root = "training"
    training_destination = os.path.join(training_root, os.path.basename(folder_to_process_path))
    
    try:
        os.makedirs(training_destination, exist_ok=False)
    except FileExistsError:
        raise gr.Error("Remove existing training folder")

    load_whisperx('large-v3')
    
    if not is_correct_dataset_structure(folder_to_process_path):
        raise gr.Error("Invalid folder structure. Ensure the folder contains ONLY subfolders.")

    speaker_folders_list = [os.path.join(folder_to_process_path, folder) for folder in os.listdir(folder_to_process_path)]
    
    for speaker_folder_path in tqdm.tqdm(speaker_folders_list, desc="Processing Speakers", file=sys.stdout):
        speaker_folder_dest = os.path.join(training_destination, os.path.basename(speaker_folder_path))
        os.makedirs(speaker_folder_dest, exist_ok=False)

        for file in tqdm.tqdm(os.listdir(speaker_folder_path), desc="Processing Files", file=sys.stdout, leave=False):
            file_path = os.path.join(speaker_folder_path, file)
            copied_path = os.path.join(speaker_folder_dest, file)
            shutil.copy(file_path, copied_path)
            
            transcription_result = run_whisperx_transcribe(copied_path)
            run_whisperx_srt(transcription_result, speaker_folder_dest)
            
            file_name = os.path.splitext(file)[0]
            srt_orig_path = os.path.join(speaker_folder_dest, f"{os.path.basename(speaker_folder_path)}.srt")
            srt_new_path = os.path.join(speaker_folder_dest, f"{file_name}.srt")
            os.rename(srt_orig_path, srt_new_path)
            
    for folder in tqdm.tqdm(os.listdir(training_destination), desc="Splitting by SRT", file=sys.stdout):
        folder_path = os.path.join(training_destination, folder)
        split_by_srt(folder_path, progress_bar=progress)
        
    return "Transcription and processing completed successfully!"

if __name__ == "__main__":
    # Keep the hefty imports away from multiprocessing 
    import whisperx
    from whisperx.utils import get_writer

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
    
    with gr.Blocks() as demo:
        with gr.Tabs("Create Dataset"):
            with gr.Row():
                with gr.Column():
                    list_of_datasets = get_available_datasets()
                    folder_to_process = gr.Dropdown(choices=list_of_datasets, value=None, label="Dataset to Process")
                    process_button = gr.Button(value="Begin Process")
                with gr.Column():
                    console_output = gr.Textbox(label="Progress Console")

                process_button.click(fn=process_proxy,
                                     inputs=folder_to_process,
                                     outputs=console_output
                                     )
                folder_to_process.change(fn=folder_to_process_proxy,
                                         inputs=folder_to_process,
                                         outputs=folder_to_process
                                         )
                
            
    port = get_port_available()
    webbrowser.open(f"http://localhost:{port}")
    demo.launch()