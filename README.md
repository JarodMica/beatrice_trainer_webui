# Beatrice V2 Training Webui
This webui is designed to train models for Beatrice v2 which is compatible with w-okada's realtime voice changing client v2: https://github.com/w-okada/voice-changer

The code to train beatrice models is adapted from: https://huggingface.co/fierce-cats/beatrice-trainer

The latest version of w-okada is 2.0.61-alpha as of writing this readme

## Requirements
- Nvidia GPU (any 3rd to 4th gen should be sufficient)
- The Ultimate Vocal Remover - https://github.com/Anjok07/ultimatevocalremovergui

## Installation
As with a majority of my packages/repos, official support will be for Windows only.  Linux shouldn't have much of an issue, just some pathing changes may be necessary.  Pull request are accepted, though, I won't be able to actively maintain any Linux additions.

### Windows Package
Will be available for Youtube Channel Members at the Supporter (Package) level: https://www.youtube.com/channel/UCwNdsF7ZXOlrTKhSoGJPnlQ/join

1. After downloading the zip file, unzip it.
2. Launch the webui with launch_webui.bat

### Windows Manual Installation
#### Prerequisites
- Python 3.11 - https://www.python.org/downloads/release/python-3119/
- git - https://git-scm.com/downloads

0. Install FFMPEG, overall, just a good tool to have and is needed for the repo.
    - https://www.youtube.com/watch?v=JR36oH35Fgg&ab_channel=Koolac

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
    > torch 2.4.0 causes issues with CTranslate2 (causes issue with whisperx) so make sure you do this step

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

## Usage
There are 3 tabs: Create Dataset, Train, and Settings.

### Create Dataset
This tab is where you create your dataset.  Follow the steps below to get a feel for doing this.

1. Obtain audio data to train on the speaker
    - This can be a podcast, audiobook, youtube video, etc.  Basically, anything that has audio (even songs, but not recommended)
    - One large audio file is recommended, but several smaller files can be used too.
2. Navigate into the WebUI ```datasets``` folder in the file explorer.  Create a new folder in here and name it whatever you want the final model to be named.  Open this now empty folder.
3. Decide how many speakers you want inside of this beatrice model (as beatrice can be multispeaker) and then create a new folder for each speaker you want.  Then, place audio files of each speaker into their respective folders.
    - For example, let's say from step 2, you want the model to be called ```elden_ring``` and you have audio files for two speakers, ```melina``` and ```ranni```
    - The folder structure would look like this:
    ```
    elden_ring\ranni\<many audio files of ranni>
    elden_ring\melina\<many audio files of ranni>
    ```
4. Now launch the training webui.  In the ```Dataset to Process``` dropdown, select the freshly created dataset from steps 1-3 (if you don't see it, click ```Refresh Datasets Available```)
    - If you run into any errors here, you may have setup the folder structure incorrectly
5. Click ```Begin Process``` and it will start curating a dataset.  The output will be placed in your ```training``` folder
    - Manual install users will incur an additional download for the whisper model that is used to split the datasaet.
6. After some time, you should see something like ```Dataset creation completed successfully``` in the ```Progress Console``` window.
7. Congrats, your first dataset has been completed!

I haven't run into any issues at this step, so if you do, please open an issue in the github tab

### Train
The Create Dataset step should be completed before this proceeding here.  If you don't see anything in the dropdown menu, click ```Refresh Training Datasets Available``` and then choose the dataset to train on.

You could just click ```Start Training``` and use the defaults, but I would adjust some of the settings based on what the webui says.

### Settings
**Dark Mode** - Toggle on/off Dark Mode

**Toggle Custom Theme** - Toggle on/off custom theme

## Acknowledgements
This would not be possible without w-okada and his contributors.  Huge thanks to them for creating this powerful open-source tool: https://github.com/w-okada/voice-changer

## License
Everything I've coded it MIT.  Check w-okada for any licenses involving his tools (the voice changer client and beatrice)

Audio files used here are directly from Libritts-r: https://www.openslr.org/141/ which retains a license of CC BY 4.0.
