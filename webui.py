import os
import sys
import shutil

import gradio as gr
import webbrowser
import socket
import tqdm
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time
import math

from multiprocessing import Pool, cpu_count
import pysrt
from pydub import AudioSegment

from gradio_utils.utils import get_available_items, refresh_dropdown_proxy, move_existing_folder, get_port_available, launch_tensorboard_proxy

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
    if len(os.listdir(folder_to_analyze)) <= 0:
        return False
    for item in os.listdir(folder_to_analyze):
        path_to_item = os.path.join(folder_to_analyze, item)
        if os.path.isdir(path_to_item):
            pass
        else:
            return False
    return True

def folder_to_process_proxy(folder_to_analyze):
    folder_check = is_correct_dataset_structure(folder_to_analyze)
    if folder_check==False:
        raise gr.Error("Please check the folder structure and make sure it contains ONLY folders and that it's NOT empty")
    return gr.Dropdown(value=folder_to_analyze)

def load_whisperx(model_name=None, progress=None):
    import whisperx
    # import whisper
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        raise gr.Error("Non-Nvidia GPU detected, or CUDA not available")
    try:
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="float16")
    except Exception as e: # for older GPUs
        print(f"Non float16 compatible GPU: {e}")
        whisper_model = whisperx.load_model(model_name, device, download_root="whisper_models", compute_type="int8")
    print("Loaded Whisper model")
    return whisper_model

def run_whisperx_transcribe(audio_file_path, chunk_size=15, language=None):
    
    audio = whisperx.load_audio(audio_file_path)
    result = whisper_model.transcribe(audio=audio, batch_size=16, chunk_size=chunk_size)

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
        if file.endswith(('.wav', '.mp3', '.m4a', ".mp4")): 
            audio_file = os.path.join(folder_path, file)
            srt_file = os.path.join(folder_path, file.rsplit('.', 1)[0] + '.srt')
            if os.path.exists(srt_file):
                file_pairs.append((folder_path, audio_file, srt_file))

    with Pool(cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_speaker_folder, file_pairs), total=len(file_pairs), desc="Processing Files", file=sys.stdout))

def process_proxy(folder_to_process_path, progress = gr.Progress(track_tqdm=True)):
    global whisper_model
    training_root = "training"
    training_destination = os.path.join(training_root, os.path.basename(folder_to_process_path))
    
    try:
        os.makedirs(training_destination, exist_ok=False)
    except FileExistsError:
        raise gr.Error("Remove existing training folder")

    whisper_model = load_whisperx('large-v3')
    
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
        
    return "Dataset creation completed successfully!"

def count_items_in_directory(root):
        file_count = 0
        
        # Use rglob to iterate through all files recursively
        for item in root.rglob('*'):  # The '*' pattern matches everything
            if item.is_file():
                file_count += 1
        
        return file_count
    
def find_largest_folder(directory):
    max_file_count = 0
    largest_folder = None
    
    # Iterate through each folder in the given directory
    for root, dirs, files in os.walk(directory):
        # If there are files in the current folder, compare file counts
        if files:
            file_count = len(files)
            if file_count > max_file_count:
                max_file_count = file_count
                largest_folder = root
    
    # print(f"Largest folder: {largest_folder} with {max_file_count} files.")
    return max_file_count
    
def training_calculations(total_audio_files, batch_size, epochs):
    batches_per_epoch = total_audio_files // batch_size
    n_steps = epochs * batches_per_epoch
    return [batches_per_epoch, n_steps]

def recommendation_proxy(data_dir, epochs):
    # Based on one of my models, 2k audio files in a training folder, at about 50 epoches it sounds decent
    # Exposed to this speaker a total of 140k times thoughout training
    # 2k at 5 epochs also sounded fine, going to recommend the lowest
    recommended_exposure = 10000
    largest_file_count = find_largest_folder(data_dir)
    user_value = largest_file_count * epochs
    recommended_epochs = math.ceil(recommended_exposure / largest_file_count)
    if user_value < recommended_exposure:
        return (f"Your largest dataset folder contains {largest_file_count} files, so at {epochs} epochs, "
            f"the model will be exposed to this speaker {user_value} times.\n"
            f"Recommended exposure is {recommended_exposure} times, so I'd suggest {recommended_epochs} epochs.\n\n"
            f"You may find it not necessary to train longer but that is up to you to decide.")
    else:
        return f"The amount of training is sufficient.  If the model is lacking after, I recommend either adding more audio files, or training for longer."

def training_proxy(data_dir, batch_size, epochs, num_workers, resume, save_interval, log_interval, progress=gr.Progress(track_tqdm=True)):
    # pathlib used here cuz of beatrice trainer
    from beatrice_trainer.src.train import run_training
    output_name = os.path.basename(data_dir)
    output_dir = os.path.join("trained_models", output_name)
    models_output_dir = Path(os.path.join("trained_models", output_name, "models"))
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    
    total_audio_files = count_items_in_directory(data_dir)
    
    training_values = training_calculations(total_audio_files, batch_size, epochs)
    # calculate batches per epoch
    batches_per_epoch = training_values[0]
    # calculate total steps needed
    n_steps = training_values[1]
    # warmup steps
    # just going with half based on the initial config set by the okada
    warmup_steps = n_steps // 2
    
    # Max number of warmup_steps hardset at 10k
    if warmup_steps > 10000:
        warmup_steps = 10000
    
    def update_configurations(config, batch_size, n_steps, num_workers, warmup_steps):
        config['batch_size'] = batch_size
        config['n_steps'] = n_steps
        config['num_workers'] = num_workers
        config['warmup_steps'] = warmup_steps
        return config
    
    config_path = Path('assets/default_config.json')
    with config_path.open('r') as file:
        config = json.load(file)
        
    config = update_configurations(config, batch_size, n_steps, num_workers, warmup_steps)
    updated_config_path = output_dir / 'updated_config.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with updated_config_path.open('w') as file:
        json.dump(config, file, indent=4)

    
    # data_dir, out_dir, resume=False, config=None
    try:
        run_training(data_dir, models_output_dir, batches_per_epoch, save_interval, log_interval , resume, updated_config_path)
    except Exception as e:
        raise gr.Error(e)
    
if __name__ == "__main__":
    # Keep the hefty imports away from multiprocessing 
    import whisperx
    from whisperx.utils import get_writer
    import torch

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
    
    def load_settings():
        settings_file = 'configs/settings.json'
        
        if not os.path.exists(settings_file):
            settings = {"custom_theme": True, "dark_mode": True}
            save_settings(settings) 
        else:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        
        return settings

    def save_settings(settings):
        os.makedirs(os.path.dirname('configs/settings.json'), exist_ok=True)
        with open('configs/settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    settings = load_settings()
    if settings.get("custom_theme", True):
        theme = gr.themes.Glass(
            primary_hue="zinc",
            secondary_hue="slate",
            neutral_hue="orange",
            text_size="lg"
        ).set(
            body_background_fill_dark='*primary_900',
            body_text_color='*primary_950',
            body_text_color_subdued='*neutral_500',
            embed_radius='*radius_md',
            border_color_accent_subdued_dark='*neutral_950',
            border_color_primary_dark='*secondary_800',
            color_accent_soft='*primary_400',
            block_border_width_dark='0',
            block_label_border_width_dark='None',
            block_shadow_dark='*primary_600 0px 0px 5px 0px',
            button_border_width='2px',
            button_border_width_dark='0px',
            button_shadow_hover='*block_shadow',
            button_large_radius='*radius_md',
            button_small_radius='*radius_md',
            button_small_text_weight='500',
            button_primary_border_color='*primary_500',
            button_primary_border_color_dark='*primary_950'
            
        )
    else:
        theme = gr.themes.Default()

    def toggle_theme():
        settings = load_settings()
        settings["custom_theme"] = not settings.get("custom_theme", False)
        save_settings(settings)
        if settings['custom_theme']:
            gr.Info("Gradio will boot up with custom theme on next start up.")
        else:
            gr.Info("Gradio will boot up with the default theme on next start up.")
            
    def toggle_dark_mode():
        settings = load_settings()
        settings["dark_mode"] = not settings.get("dark_mode", True)
        save_settings(settings)
        if settings['dark_mode']:
            gr.Info("Gradio will boot up with dark mode on next start up.")
        else:
            gr.Info("Gradio will boot up with light mode on next start up.")

    # Construct the JavaScript based on dark mode setting
    js_dark_mode = "document.querySelector('body').classList.add('dark');" if settings.get("dark_mode", True) else "document.querySelector('body').classList.remove('dark');"

    js = f"""
        function createGradioAnimation() {{
            var container = document.createElement('div');
            container.id = 'gradio-animation';
            container.style.fontSize = '2em';
            container.style.fontWeight = 'bold';
            container.style.textAlign = 'center';
            container.style.marginBottom = '20px';
            container.style.position = 'absolute';
            container.style.left = '-100%'; // Start off-screen to the left
            container.style.top = '20px'; // Adjust this value as needed to position the header vertically
            container.style.transition = 'left 1s ease-out'; // Animate the position
            container.style.zIndex = '1000'; // Ensure it stays on top of other elements
            container.style.whiteSpace = 'nowrap'; // Prevent text wrapping
            container.style.overflow = 'hidden'; // Ensure overflow is handled properly
            container.style.textOverflow = 'ellipsis'; // Show ellipsis if text overflows

            var text = 'Beatrice Voice Changer Training Webui';
            container.innerText = text;

            var gradioContainer = document.querySelector('.gradio-container');
            gradioContainer.style.position = 'relative'; // Ensure the parent is positioned relatively
            gradioContainer.style.paddingTop = '60px'; // Reserve space at the top to avoid overlap (adjust this value if needed)
            gradioContainer.insertBefore(container, gradioContainer.firstChild);

            // Trigger the animation to slide the text to the center
            setTimeout(function() {{
                container.style.left = '50%';
                container.style.transform = 'translateX(-50%)'; // Center the container
            }}, 100);

            {js_dark_mode} // Apply dark mode based on setting
            return 'Animation created';
        }}
    """
        
    with gr.Blocks(js=js, theme=theme) as demo:
        with gr.Tab("Create Dataset"):
            with gr.Row():
                with gr.Column():
                    hidden_dataset_textbox = gr.Textbox(value="datasets", visible=False)
                    list_of_datasets = get_available_items(root="datasets", directory_only=True)
                    folder_to_process = gr.Dropdown(choices=list_of_datasets, value=None, label="Dataset to Process")
                    refresh_datasets_button = gr.Button(value="Refresh Datasets Available")
                    move_existing_folder_button = gr.Button(value="Move Existing Folder")
                    process_button = gr.Button(value="Begin Process", variant="primary")
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
                
                destination_root = gr.Textbox(value="training/moved_training_datasets", visible=False)
                source_root = gr.Textbox(value="training", visible=False)
                move_existing_folder_button.click(fn=move_existing_folder,
                                                  inputs=[source_root, 
                                                          folder_to_process, 
                                                          destination_root]
                )
                
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    hidden_train_textbox = gr.Textbox(value="training", visible=False)
                    TRAINING_SETTINGS = {}
                    list_of_training_datasets = get_available_items(root="training", directory_only=True)
                    TRAINING_SETTINGS["dataset_name"] = gr.Dropdown(label="Dataset to Train", choices=list_of_training_datasets, value=list_of_training_datasets[0] if list_of_training_datasets else '')
                    refresh_training_available_button = gr.Button(value="Refresh Training Datasets Available")
                    TRAINING_SETTINGS["batch_size"] = gr.Slider(label="Batch Size", minimum=1, maximum=64, value=4, step=1)
                    TRAINING_SETTINGS["epochs"] = gr.Slider(label="Number of Epochs", minimum=1, maximum=1000, value=20, step=1)
                    TRAINING_SETTINGS["num_workers"] = gr.Slider(label="Number of Workers",minimum=1, maximum=32, value=4, step=1)
                    TRAINING_SETTINGS["save_interval"] = gr.Slider(label="Save Interval in Epochs", minimum=1, maximum= 200, value= 5, step=1)
                    TRAINING_SETTINGS["log_interval"] = gr.Slider(label="Console Log Interval", minimum=10, maximum=1000, step=10)
                    TRAINING_SETTINGS["resume"] = gr.Checkbox(label="Resume Training", value=False)
                    # TRAINING_SETTINGS["warmup_steps"] =

                    html_value = '''<h2>What are Batches</h2>
                                    <p>Bunches or groups of files that are processed at once by the model before updating gradients (model predictions --> loss calc --> gradient update). A batch size of 1 trains on a single audio file at a time, a batch size of 8 trains on 8 audio files at a time.</p>

                                    <h3>Batch Size:</h3>
                                    <p>The number of audio files processed per step. The higher the value, the faster training is but also incurs more VRAM usage.</p>

                                    <h2>What are Epochs</h2>
                                    <p>A complete pass through the entire dataset where the model has "been trained on" all of the audio samples a single time.</p>

                                    <h3>Number of Epochs:</h3>
                                    <p>The amount of epochs you want to train the model for. The higher the value, the better the model may sound but it will take longer to finish.</p>

                                    <h2>What are Workers</h2>
                                    <p>Processes or "sorters" that go through the dataset to curate and create the batches needed for training.</p>

                                    <h3>Number of Workers:</h3>
                                    <p>The amount of workers created to sort the data. The higher the value, the faster data gets prepared, but may cause unnecessary overhead if your GPU isn't fast enough. Recommend to leave at default value.</p>

                                    <h3>Save Interval in Epochs:</h3>
                                    <p>How often a model is saved. For larger datasets, I'd save at lower intervals as you will need fewer epochs to complete training. For smaller datasets, I'd save at larger intervals to reduce how much space training will take up to complete.</p>

                                    <h3>Console Log Interval:</h3>
                                    <p>How often training loss is output to the terminal and for tensorboard. Recommend 10-100.</p>
                                    '''
                with gr.Column():   
                    recommendation_console = gr.Textbox(label="Jarods's Recommendation")
            with gr.Row():
                output_console = gr.Textbox(label="Training Console")
            with gr.Row():
                with gr.Column():
                    start_train_button = gr.Button(value="Start Training", variant="primary")
                with gr.Column():
                    launch_tb_button = gr.Button(value="Launch Tensorboard")
            with gr.Row():
                gr.HTML(value=html_value)
        with gr.Tab("Settings"):
            dark_mode_btn = gr.Button("Dark Mode", variant="primary")
            toggle_theme_btn = gr.Button("Toggle Custom Theme", variant="primary")
            
        for key, component in TRAINING_SETTINGS.items():
                if isinstance(component, gr.Dropdown):
                    component.change(
                        fn=recommendation_proxy, 
                        inputs=[TRAINING_SETTINGS["dataset_name"], 
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )
                elif isinstance(component, gr.Slider):
                    component.release(
                        fn=recommendation_proxy, 
                        inputs=[TRAINING_SETTINGS["dataset_name"], 
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )

        start_train_button.click(fn=training_proxy,
                                 inputs=[
                                     TRAINING_SETTINGS["dataset_name"],
                                     TRAINING_SETTINGS["batch_size"],
                                     TRAINING_SETTINGS["epochs"],
                                     TRAINING_SETTINGS["num_workers"],
                                     TRAINING_SETTINGS["resume"],
                                     TRAINING_SETTINGS["save_interval"],
                                     TRAINING_SETTINGS["log_interval"]
                                         ],
                                 outputs=output_console
                                 )
        
        launch_tb_button.click(fn=launch_tensorboard_proxy)
        
        hidden_option1 = gr.Textbox(value="directory", visible=False)
        hidden_option2 = gr.Textbox(value="files", visible=False)
        
        hidden_extensions1 = gr.Textbox(value="[]", visible=False)
        
        refresh_training_available_button.click(fn=refresh_dropdown_proxy,
                                                inputs=[
                                                    hidden_train_textbox, hidden_extensions1, hidden_option1
                                                    ],
                                                outputs=[
                                                    TRAINING_SETTINGS["dataset_name"]
                                                ]
        )
        
        refresh_datasets_button.click(fn=refresh_dropdown_proxy,
                                                inputs=[
                                                    hidden_dataset_textbox, hidden_extensions1, hidden_option1
                                                    ],
                                                outputs=[
                                                    folder_to_process
                                                ]
        )
        
        toggle_theme_btn.click(toggle_theme)
        dark_mode_btn.click(toggle_dark_mode)

        dark_mode_btn.click(
            None,
            None,
            None,
            js="""() => {
            if (document.querySelectorAll('.dark').length) {
                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
            } else {
                document.querySelector('body').classList.add('dark');
            }
        }""",
            show_api=False,
        )
            
    port = get_port_available()
    webbrowser.open(f"http://localhost:{port}")
    demo.launch()