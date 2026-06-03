@echo off
setlocal

set "PROJECT_ROOT=C:\Users\92127\OneDrive - UNSW\project_linpeng\code"
set "PYTHON_EXE=C:\Users\92127\anaconda3\envs\bhsd\python.exe"
set "STDOUT_LOG=%PROJECT_ROOT%\results\run_logs\fold0_train_queue.stdout.log"
set "STDERR_LOG=%PROJECT_ROOT%\results\run_logs\fold0_train_queue.stderr.log"

if not exist "%PROJECT_ROOT%\results\run_logs" mkdir "%PROJECT_ROOT%\results\run_logs"

set "nnUNet_raw=%PROJECT_ROOT%\nnUNet_data\nnUNet_raw"
set "nnUNet_preprocessed=%PROJECT_ROOT%\nnUNet_data\nnUNet_preprocessed"
set "nnUNet_results=%PROJECT_ROOT%\nnUNet_data\nnUNet_results"
set "nnUNet_n_proc_DA=0"

cd /d "%PROJECT_ROOT%"

echo [%date% %time%] Fold-0 training queue started>> "%STDOUT_LOG%"
echo [%date% %time%] Starting training for baseline_2d>> "%STDOUT_LOG%"
"%PYTHON_EXE%" "scripts\run_experiment.py" train --config baseline_2d >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"
if errorlevel 1 goto :fail

echo [%date% %time%] Finished training for baseline_2d>> "%STDOUT_LOG%"
echo [%date% %time%] Starting training for baseline_3d>> "%STDOUT_LOG%"
"%PYTHON_EXE%" "scripts\run_experiment.py" train --config baseline_3d >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"
if errorlevel 1 goto :fail

echo [%date% %time%] Finished training for baseline_3d>> "%STDOUT_LOG%"
echo [%date% %time%] Starting training for naive_25d_3slice>> "%STDOUT_LOG%"
"%PYTHON_EXE%" "scripts\run_experiment.py" train --config naive_25d_3slice >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"
if errorlevel 1 goto :fail

echo [%date% %time%] Finished training for naive_25d_3slice>> "%STDOUT_LOG%"
echo [%date% %time%] Starting training for naive_25d_5slice>> "%STDOUT_LOG%"
"%PYTHON_EXE%" "scripts\run_experiment.py" train --config naive_25d_5slice >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"
if errorlevel 1 goto :fail

echo [%date% %time%] Finished training for naive_25d_5slice>> "%STDOUT_LOG%"
echo [%date% %time%] Fold-0 training queue finished>> "%STDOUT_LOG%"
exit /b 0

:fail
echo [%date% %time%] Fold-0 training queue failed with exit code %errorlevel%>> "%STDERR_LOG%"
exit /b %errorlevel%
