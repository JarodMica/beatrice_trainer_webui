# Beatrice 2.0 Trainer WebUI

Windows-focused Gradio interface and offline-package tooling for training **Beatrice 2.0.0-rc.0** voice-conversion models.

The trainer is maintained separately and pinned here as a Git submodule:

- `modules/beatrice_trainer` → [JarodMica/beatrice_trainer](https://github.com/JarodMica/beatrice_trainer)
- `modules/gradio_utils` → [JarodMica/gradio_utils](https://github.com/JarodMica/gradio_utils)
- Official Beatrice project: <https://prj-beatrice.com/>
- Official upstream trainer: <https://huggingface.co/fierce-cats/beatrice-trainer>

## Compatibility warning

This branch trains the RC0 model architecture and exports RC0 paraphernalia for current Beatrice VST/VCClient releases.

- Alpha.2 checkpoints cannot be resumed with this trainer.
- Alpha.2 and RC0 packages should be kept as separate installations.
- If an exported model cannot be loaded, update the Beatrice VST or voice-changing client before troubleshooting the model.

## Requirements

- Windows 10/11
- NVIDIA GPU with a current driver
- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Git with Git LFS
- FFmpeg available on `PATH` for manual installations

The locked CUDA environment uses Python 3.11, PyTorch/TorchAudio 2.8.0, and CUDA 12.8 wheels.

## Manual installation

Clone the WebUI and its exact pinned submodule commits:

```powershell
git lfs install
git clone --recurse-submodules https://github.com/JarodMica/beatrice_trainer_webui.git
cd beatrice_trainer_webui
uv sync --locked --extra cu128
.\launch_webui.bat
```

Do **not** use `git submodule update --remote` for a release checkout. The superproject commit intentionally pins reviewed trainer and utility revisions.

For TensorBoard:

```powershell
.\launch_tensorboard.bat
```

## Offline Windows package

The supporter package includes:

- Relocatable Python 3.11 runtime
- Locked CUDA 12.8 PyTorch environment
- Beatrice RC0 source and pretrained assets
- WhisperX and faster-whisper large-v3
- Silero VAD, English alignment, and UTMOS caches
- FFmpeg and FFprobe

Unzip the package and run `launch_webui.bat`. No Python, uv, Git, or model download is required for the validated English dataset workflow.

Install RC0 as a separate package from legacy Alpha.2 releases. Raw `datasets/` and processed `training/` audio can be copied between installations, but checkpoints, trainer modules, assets, and runtimes cannot.

## Usage

### 1. Create a dataset

Place source audio under one dataset directory with one subdirectory per speaker:

```text
datasets/
└── my_voice/
    ├── speaker_one/
    │   ├── recording_a.wav
    │   └── recording_b.flac
    └── speaker_two/
        └── recording.wav
```

In **Create Dataset**:

1. Select the dataset.
2. Click **Begin Process**.
3. WhisperX transcribes and aligns each supported source file.
4. Segments are written to `training/<dataset>/<speaker>/`.

Segment filenames include the original source stem, preventing files from different recordings from overwriting each other.

### 2. Train

In **Train**:

1. Select a processed training dataset.
2. Choose batch size and number of epochs.
3. Review the displayed optimizer-step calculation.
4. Set save/evaluation intervals in epochs.
5. Click **Start Training**.

The official RC0 default is 10,000 optimizer steps. The recommendation panel converts this target into an approximate epoch count for the selected dataset and batch size.

Outputs are written to:

```text
trained_models/<dataset>/
├── updated_config.json
└── models/
    ├── checkpoint_latest.pt.gz
    ├── checkpoint_<dataset>_<step>.pt.gz
    ├── config.json
    └── paraphernalia_<dataset>_<step>/
```

Use **Stop Training** to cancel the trainer process tree safely.

### Resume behavior

Enable **Resume Training** to continue from `checkpoint_latest.pt.gz`.

- The entered epoch count means **additional epochs**.
- The WebUI reads the checkpoint iteration and adds the requested steps.
- The original warmup schedule is preserved.
- RC0 `.pt.gz` checkpoints are required; legacy `.pt` checkpoints are rejected.

## Development validation

```powershell
uv sync --locked --extra cu128
uv run python -m py_compile webui.py
uv run python webui.py
```

Before releasing changes, manually verify dataset transcription/alignment, fresh training, checkpoint resume, and RC0 model loading with a current Beatrice client.

## Acknowledgements

Thanks to Project Beatrice, w-okada, and their contributors for the underlying trainer, runtime format, and voice-changing clients.

The original example audio in this repository comes from LibriTTS-R under CC BY 4.0: <https://www.openslr.org/141/>.

## License

WebUI code in this repository is MIT licensed. Refer to the upstream projects and bundled assets for their respective licenses.
