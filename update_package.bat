@echo off

set REPO_NAME=beatrice_trainer_webui

if not exist "runtime" (
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    echo This is not the packaged version, if you are trying to update your manual installation, please use git pull instead
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pause
    exit /b
)

if exist "%REPO_NAME%" (
    @echo It looks like you've already cloned the repository for updating before.
    @echo If you want to continue with updating, type y.
    @echo Else, type n.
    choice /M "Do you want to continue?"
    if errorlevel 2 (
        @echo Exiting the script...
        exit /b
    )
    rmdir /S /Q "%REPO_NAME%"
)

portable_git\bin\git.exe clone https://github.com/JarodMica/%REPO_NAME%.git
cd %REPO_NAME%
git submodule init
git submodule update --remote
cd ..

xcopy %REPO_NAME%\update_package.bat update_package.bat /E /I /H 
xcopy %REPO_NAME%\launch_tensorboard.bat launch_tensorboard.bat /E /I /H 
xcopy %REPO_NAME%\requirements.txt requirements.txt /E /I /H 

xcopy %REPO_NAME%\webui.py webui.py /H
xcopy %REPO_NAME%\modules\beatrice_trainer modules\beatrice_trainer /E /I /H
xcopy %REPO_NAME%\modules\gradio_utils modules\gradio_utils /E /I /H

runtime\python.exe -m pip uninstall beatrice_trainer
runtime\python.exe -m pip install modules\beatrice_trainer

runtime\python.exe -m pip uninstall gradio_utils
runtime\python.exe -m pip install modules\gradio_utils
runtime\python.exe -m pip install -r requirements.txt

@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo Finished updating!
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pause