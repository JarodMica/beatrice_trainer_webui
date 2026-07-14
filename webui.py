import gzip
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "modules"
TRAINER_ROOT = MODULES_DIR / "beatrice_trainer"
TRAINER_ENTRYPOINT = TRAINER_ROOT / "beatrice_trainer" / "__main__.py"
for local_module in (TRAINER_ROOT, MODULES_DIR / "gradio_utils"):
    sys.path.insert(0, str(local_module))
os.environ["PATH"] = f"{BASE_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / "torch_home"))

AUDIO_FILE_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".fla",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

import gradio as gr
import tqdm

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
    if not folder_to_analyze or not os.path.isdir(folder_to_analyze):
        return False
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
    whisper_download_root = BASE_DIR / "whisper_models"
    cached_model_dir = (
        whisper_download_root / f"models--Systran--faster-whisper-{model_name}"
    )
    local_files_only = cached_model_dir.is_dir()
    try:
        whisper_model = whisperx.load_model(
            model_name,
            device,
            download_root=str(whisper_download_root),
            compute_type="float16",
            vad_method="silero",
            local_files_only=local_files_only,
        )
    except Exception as e: # for older GPUs
        print(f"Non float16 compatible GPU: {e}")
        whisper_model = whisperx.load_model(
            model_name,
            device,
            download_root=str(whisper_download_root),
            compute_type="int8",
            vad_method="silero",
            local_files_only=local_files_only,
        )
    print("Loaded Whisper model")
    return whisper_model

def run_whisperx_transcribe(audio_file_path, chunk_size=15, language=None):
    
    audio = whisperx.load_audio(audio_file_path)
    result = whisper_model.transcribe(
        audio=audio, batch_size=16, chunk_size=chunk_size
    )
    detected_language = result.get("language") or language
    if not detected_language:
        raise gr.Error("WhisperX did not detect an audio language.")

    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language, device="cuda"
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device="cuda",
        return_char_alignments=False,
    )

    # Alignment may omit the language required by WhisperX's SRT writer.
    result["language"] = detected_language
    
    return result

def run_whisperx_srt(transcription_result, audio_file_path, output_directory):
    srt_writer = get_writer("srt", output_directory)
    srt_writer(
        transcription_result,
        audio_file_path,
        {
            "max_line_width": None,
            "max_line_count": None,
            "highlight_words": False,
        },
    )

def process_speaker_folder(file_info, progress_bar=None):
    folder_path, audio_file, srt_file = file_info

    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1) # Mono conversion
    subs = pysrt.open(srt_file)

    # Multiple source files from one speaker are processed in parallel. Include
    # the source stem so their independently numbered segments cannot collide.
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
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

    if not file_pairs:
        return
    worker_count = min(cpu_count(), len(file_pairs))
    with Pool(worker_count) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_speaker_folder, file_pairs), total=len(file_pairs), desc="Processing Files", file=sys.stdout))

def process_proxy(folder_to_process_path, progress = gr.Progress(track_tqdm=True)):
    global whisper_model
    training_root = "training"
    training_destination = os.path.join(training_root, os.path.basename(folder_to_process_path))
    
    if not is_correct_dataset_structure(folder_to_process_path):
        raise gr.Error("Invalid folder structure. Ensure the folder contains ONLY subfolders.")

    whisper_model = load_whisperx('large-v3')

    try:
        os.makedirs(training_destination, exist_ok=False)
    except FileExistsError:
        raise gr.Error("Remove existing training folder")

    speaker_folders_list = [os.path.join(folder_to_process_path, folder) for folder in os.listdir(folder_to_process_path)]
    
    for speaker_folder_path in tqdm.tqdm(speaker_folders_list, desc="Processing Speakers", file=sys.stdout):
        speaker_folder_dest = os.path.join(training_destination, os.path.basename(speaker_folder_path))
        os.makedirs(speaker_folder_dest, exist_ok=False)

        for file in tqdm.tqdm(os.listdir(speaker_folder_path), desc="Processing Files", file=sys.stdout, leave=False):
            file_path = os.path.join(speaker_folder_path, file)
            if not os.path.isfile(file_path):
                continue
            if Path(file).suffix.lower() not in AUDIO_FILE_SUFFIXES:
                print(f"Skipping unsupported file: {file_path}")
                continue

            copied_path = os.path.join(speaker_folder_dest, file)
            shutil.copy(file_path, copied_path)

            transcription_result = run_whisperx_transcribe(copied_path)
            run_whisperx_srt(
                transcription_result, copied_path, speaker_folder_dest
            )
            
    for folder in tqdm.tqdm(os.listdir(training_destination), desc="Splitting by SRT", file=sys.stdout):
        folder_path = os.path.join(training_destination, folder)
        split_by_srt(folder_path, progress_bar=progress)
        
    return "Dataset creation completed successfully!"

def count_audio_files(root):
    root = Path(root)
    return sum(
        1
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in AUDIO_FILE_SUFFIXES
    )


def training_calculations(total_audio_files, batch_size, epochs):
    batches_per_epoch = total_audio_files // batch_size
    n_steps = epochs * batches_per_epoch
    return batches_per_epoch, n_steps


def recommendation_proxy(data_dir, batch_size, epochs):
    if not data_dir:
        return "Select a training dataset."
    total_audio_files = count_audio_files(data_dir)
    batches_per_epoch, requested_steps = training_calculations(
        total_audio_files, batch_size, epochs
    )
    if batches_per_epoch <= 0:
        return (
            f"Batch size {batch_size} is larger than the {total_audio_files} "
            "supported audio files in this dataset."
        )

    recommended_steps = 10000
    recommended_epochs = math.ceil(recommended_steps / batches_per_epoch)
    message = (
        f"{total_audio_files} audio files / batch {batch_size} = "
        f"{batches_per_epoch} steps per epoch.\n"
        f"{epochs} epochs will run {requested_steps} optimizer steps.\n"
        f"The Beatrice 2.0.0-rc.0 default is {recommended_steps} steps "
        f"(about {recommended_epochs} epochs for this dataset)."
    )
    if requested_steps < recommended_steps:
        message += "\nA shorter run is useful for testing but may underfit the voice."
    return message


def _checkpoint_iteration(checkpoint_path):
    import torch

    with gzip.open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
    try:
        return int(checkpoint["iteration"]), dict(checkpoint.get("h", {}))
    finally:
        del checkpoint


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _stream_subprocess(command, cwd):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        creationflags=creationflags,
    )
    output_queue = queue.Queue()

    def read_output():
        assert process.stdout is not None
        while chunk := process.stdout.read(1):
            output_queue.put(chunk)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    completed_lines = deque(maxlen=200)
    current_line = ""
    last_update = 0.0

    try:
        while process.poll() is None or reader.is_alive() or not output_queue.empty():
            try:
                chunk = output_queue.get(timeout=0.2)
            except queue.Empty:
                chunk = None

            if chunk in {"\r", "\n"}:
                if current_line:
                    if chunk == "\r" and completed_lines:
                        completed_lines[-1] = current_line
                    else:
                        completed_lines.append(current_line)
                    current_line = ""
            elif chunk is not None:
                current_line += chunk

            now = time.monotonic()
            if now - last_update >= 0.5:
                visible_output = "\n".join(completed_lines)
                if current_line:
                    visible_output += f"\n{current_line}"
                yield visible_output
                last_update = now

        if current_line:
            completed_lines.append(current_line)
        final_output = "\n".join(completed_lines)
        if process.returncode != 0:
            raise gr.Error(
                f"Beatrice trainer exited with code {process.returncode}.\n\n"
                f"{final_output[-8000:]}"
            )
        yield f"{final_output}\n\nTraining completed successfully."
    finally:
        _terminate_process_tree(process)


def training_proxy(
    data_dir,
    batch_size,
    epochs,
    num_workers,
    resume,
    save_interval,
    evaluation_interval,
):
    if not data_dir:
        raise gr.Error("Select a training dataset.")
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise gr.Error(f"Training dataset does not exist: {data_dir}")
    if not TRAINER_ENTRYPOINT.is_file():
        raise gr.Error(f"Beatrice trainer entrypoint is missing: {TRAINER_ENTRYPOINT}")

    total_audio_files = count_audio_files(data_dir)
    batches_per_epoch, requested_steps = training_calculations(
        total_audio_files, batch_size, epochs
    )
    if batches_per_epoch <= 0:
        raise gr.Error(
            f"Batch size {batch_size} is larger than the {total_audio_files} "
            "supported audio files in this dataset. Choose a smaller batch size."
        )

    output_name = data_dir.name
    output_dir = BASE_DIR / "trained_models" / output_name
    models_output_dir = output_dir / "models"
    latest_checkpoint = models_output_dir / "checkpoint_latest.pt.gz"
    trainer_config = models_output_dir / "config.json"
    default_config = TRAINER_ROOT / "assets" / "default_config.json"

    if resume:
        if not latest_checkpoint.is_file():
            raise gr.Error(
                f"Cannot resume because {latest_checkpoint} was not found."
            )
        initial_iteration, checkpoint_config = _checkpoint_iteration(
            latest_checkpoint
        )
        if initial_iteration < 0:
            raise gr.Error(
                f"Cannot resume because {latest_checkpoint} has no valid iteration."
            )
        if trainer_config.is_file():
            config = json.loads(trainer_config.read_text(encoding="utf-8"))
        else:
            config = checkpoint_config
        n_steps = initial_iteration + requested_steps
        warmup_steps = int(config.get("warmup_steps", 5000))
    else:
        if latest_checkpoint.exists():
            raise gr.Error(
                f"{latest_checkpoint} already exists. Enable Resume Training or "
                "move/remove the existing model directory."
            )
        config = json.loads(default_config.read_text(encoding="utf-8"))
        initial_iteration = 0
        n_steps = requested_steps
        warmup_steps = min(n_steps // 2, 5000)

    config.update(
        {
            "batch_size": int(batch_size),
            "n_steps": int(n_steps),
            "num_workers": int(num_workers),
            "warmup_steps": int(warmup_steps),
            "save_interval": max(1, int(save_interval) * batches_per_epoch),
            "evaluation_interval": max(
                1, int(evaluation_interval) * batches_per_epoch
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_config_path = output_dir / "updated_config.json"
    updated_config_path.write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )

    command = [
        sys.executable,
        str(TRAINER_ENTRYPOINT),
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(models_output_dir),
        "--config",
        str(updated_config_path),
    ]
    if resume:
        command.append("--resume")

    summary = (
        f"Starting Beatrice 2.0.0-rc.0 training at iteration "
        f"{initial_iteration:,}; target iteration {n_steps:,}.\n"
        f"Command: {subprocess.list2cmdline(command)}"
    )
    yield summary
    for output in _stream_subprocess(command, BASE_DIR):
        yield f"{summary}\n\n{output}"
    
if __name__ == "__main__":
    # Keep the hefty imports away from multiprocessing 
    import whisperx
    from whisperx.utils import get_writer
    import torch

    whisper_model = None

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
                    TRAINING_SETTINGS["dataset_name"] = gr.Dropdown(label="Dataset to Train", choices=list_of_training_datasets, value=list_of_training_datasets[0] if list_of_training_datasets else None)
                    refresh_training_available_button = gr.Button(value="Refresh Training Datasets Available")
                    TRAINING_SETTINGS["batch_size"] = gr.Slider(label="Batch Size", minimum=1, maximum=64, value=4, step=1)
                    TRAINING_SETTINGS["epochs"] = gr.Slider(label="Number of Epochs", minimum=1, maximum=1000, value=20, step=1)
                    TRAINING_SETTINGS["num_workers"] = gr.Slider(label="Number of Workers", minimum=0, maximum=32, value=4, step=1)
                    TRAINING_SETTINGS["save_interval"] = gr.Slider(label="Save Interval in Epochs", minimum=1, maximum=200, value=50, step=1)
                    TRAINING_SETTINGS["evaluation_interval"] = gr.Slider(label="Evaluation Interval in Epochs", minimum=1, maximum=200, value=50, step=1)
                    TRAINING_SETTINGS["resume"] = gr.Checkbox(label="Resume Training (epochs are additional)", value=False)
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

                                    <h3>Evaluation Interval in Epochs:</h3>
                                    <p>How frequently to generate validation audio and metrics. Evaluation can take noticeably longer than a normal training step.</p>

                                    <h3>Resume Training:</h3>
                                    <p>Continues from checkpoint_latest.pt.gz. Number of Epochs is treated as additional training, while the original warmup schedule is preserved.</p>

                                    <h3>Compatibility:</h3>
                                    <p>This WebUI trains Beatrice 2.0.0-rc.0 models for current Beatrice VST/VCClient releases. Alpha.2 checkpoints cannot be resumed here.</p>
                                    '''
                with gr.Column():   
                    recommendation_console = gr.Textbox(label="Jarods's Recommendation")
            with gr.Row():
                output_console = gr.Textbox(label="Training Console", lines=20)
            with gr.Row():
                with gr.Column():
                    start_train_button = gr.Button(value="Start Training", variant="primary")
                with gr.Column():
                    stop_train_button = gr.Button(value="Stop Training", variant="stop")
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
                                TRAINING_SETTINGS["batch_size"],
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )
                elif isinstance(component, gr.Slider):
                    component.release(
                        fn=recommendation_proxy,
                        inputs=[TRAINING_SETTINGS["dataset_name"],
                                TRAINING_SETTINGS["batch_size"],
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )

        training_event = start_train_button.click(
            fn=training_proxy,
            inputs=[
                TRAINING_SETTINGS["dataset_name"],
                TRAINING_SETTINGS["batch_size"],
                TRAINING_SETTINGS["epochs"],
                TRAINING_SETTINGS["num_workers"],
                TRAINING_SETTINGS["resume"],
                TRAINING_SETTINGS["save_interval"],
                TRAINING_SETTINGS["evaluation_interval"],
            ],
            outputs=output_console,
        )
        stop_train_button.click(fn=None, cancels=[training_event])

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
    if os.environ.get("BEATRICE_NO_BROWSER") != "1":
        webbrowser.open(f"http://localhost:{port}")
    demo.launch(server_port=port)