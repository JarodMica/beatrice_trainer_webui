# Beatrice V2 Training Webui
This webui is designed to train models for Beatrice v2 which is compatible with w-okada's realtime voice changing client v2: https://github.com/w-okada/voice-changer

The code to train beatrice models it adapted from: https://huggingface.co/fierce-cats/beatrice-trainer

The latest version of w-okada is 2.0.61-alpha as of writing this readme

## Requirements
- Nvidia GPU (any 3rd to 4th gen should be sufficient)
- The Ultimate Vocal Remover - https://github.com/Anjok07/ultimatevocalremovergui

## Installation
As with a majority of my packages/repos, official support will be for Windows only.  Linux shouldn't have much of an issue, just some pathing changes may be necessary.  Pull request are accepted, though, I won't be able to actively maintain any Linux additions.

### Windows Package
Is available for Youtube Channel Members at the Supporter (Package) level: https://www.youtube.com/channel/UCwNdsF7ZXOlrTKhSoGJPnlQ/join

1. After downloading the zip file, unzip it.
2. Launch the webui with launch_webui.bat

### Windows Manual Installation
#### Prerequisites
- Python 3.11 - https://www.python.org/downloads/release/python-3119/
- git - https://git-scm.com/downloads

1. Clone the repository
    ```
    git clone https://github.com/JarodMica/beatrice_trainer_webui.git
    ```
2. Navigate into the repo
    ```
    cd .\beatrice_trainer_webui\
    ```
3. Setup a virtual environment, specifying python 3.11
    ```
    py -3.11 -m venv venv
    ```
4. Activate venv. If you've never run venv before on windows powershell, you will need to change ExecutionPolicy to RemoteSigned: https://learn.microsoft.com/en-us/answers/questions/506985/powershell-execution-setting-is-overridden-by-a-po
    ```
    .\venv\Scripts\activate
    ```
5. Run the requirements.txt
    ```
    pip install -r .\requirements.txt
    ```
6. Uninstall and reinstall torch manually.  Other packages will install torch without cuda, to enable cuda, you need the prebuilt wheels.
    > torch 2.4.0 causes issues with ctranslate (causes issue with whisperx) so make sure you do this step

    ```
    pip uninstall torch
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
7. Initialized submodules 
    ```
    git submodule init
    git submodule update --remote
    ```
8. Install submodules into venv
    ```
    pip install .\modules\beatrice_trainer\
    pip install .\modules\gradio_utils\
    ```
9. Grab the assets from the original beatrice HuggingFace repo at this hash here: https://huggingface.co/fierce-cats/beatrice-trainer/tree/be628e89d162d0d1aa038f57f19e1f578b7e6328

    The easiest way is to clone the repo, checkout at that specific hash, then copy and paste ```assets``` into the root folder of the beatrice trainer webui
    ```
    git clone https://huggingface.co/fierce-cats/beatrice-trainer.git
    cd beatrice-trainer
    git checkout be628e89d162d0d1aa038f57f19e1f578b7e6328
    cd ..
    ```

    The folder structure should look like this:
    ```
    beatrice_trainer_webui\assets
    ```
10. Run the webui
    ```
    python webui.py
    ```
11. (Optional) Make a .bat file to automatically run the webui.py each time without having to activate venv each time. How to: https://www.windowscentral.com/how-create-and-run-batch-file-windows-10
    ```
    call venv\Scripts\activate
    python webui.py
    ```