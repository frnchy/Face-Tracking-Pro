@echo off
setlocal EnableDelayedExpansion

if "%PY%"=="" set PY=python
echo.
echo === Stage 0: Python check (%PY%)
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo *** Python "%PY%" not found.
    exit /b 1
)
%PY% --version

echo.
echo === Stage 1: Dependency diagnostic
%PY% -m facetrack.bootstrap --diagnose
set DIAG_RC=%ERRORLEVEL%
if not "%DIAG_RC%"=="0" (
    echo --- Installing missing packages ...
    %PY% -m pip install --upgrade pip
    for /f "delims=" %%P in ('%PY% -m facetrack.bootstrap --list-missing-pip') do (
        echo --- pip install %%P
        %PY% -m pip install "%%P"
        if errorlevel 1 (
            echo *** pip install %%P FAILED
            exit /b 1
        )
    )
    %PY% -m facetrack.bootstrap --diagnose
    if errorlevel 1 (
        echo *** Still missing dependencies.
        exit /b 1
    )
)

echo.
echo === Stage 2: PyInstaller
%PY% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    %PY% -m pip install --upgrade pyinstaller
    if errorlevel 1 ( exit /b 1 )
)
%PY% -m PyInstaller --version

echo.
echo === Stage 3: Verify face landmarker
%PY% -m facetrack.bootstrap --diagnose --with-facemesh
if errorlevel 1 (
    echo *** Face landmarker failed to load.
    exit /b 1
)
set MODEL_PATH=
for /f "delims=" %%M in ('%PY% -c "from facetrack.model import existing_model_path; p = existing_model_path(); print(p if p else '')"') do set MODEL_PATH=%%M
if "%MODEL_PATH%"=="" (
    set MODEL_ADD_ARG=
    echo --- No model file to bundle (Solutions API)
) else (
    set MODEL_ADD_ARG=--add-data "%MODEL_PATH%;."
    echo --- Bundling model: %MODEL_PATH%
)

echo.
echo === Stage 4: Smoke-test imports
%PY% -c "import facetrack.app, facetrack.splash, facetrack.bootstrap, facetrack.tracker, facetrack.analyzer, facetrack.widgets, facetrack.filtering, facetrack.cameras, facetrack.features, facetrack.session; print('imports OK')"
if errorlevel 1 (
    echo *** Source import failed.
    exit /b 1
)

echo.
echo === Stage 5: Clean prior build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist exe rmdir /s /q exe
if exist FaceTrackerPro.spec del /q FaceTrackerPro.spec

echo.
echo === Stage 6: Building FaceTrackerPro.exe

set ICON_ARG=
if exist "assets\brand.ico" set ICON_ARG=--icon assets\brand.ico

set ASSET_ARG=
if exist "assets\brand.png" set ASSET_ARG=--add-data "assets\brand.png;assets"

%PY% -m PyInstaller ^
    --noconfirm ^
    --onefile ^
    --windowed ^
    --name FaceTrackerPro ^
    %ICON_ARG% ^
    %ASSET_ARG% ^
    %MODEL_ADD_ARG% ^
    --collect-all mediapipe ^
    --collect-all customtkinter ^
    --collect-data cv2 ^
    --hidden-import pygrabber ^
    --hidden-import pygrabber.dshow_graph ^
    --hidden-import comtypes ^
    --hidden-import PIL._tkinter_finder ^
    main.py

if errorlevel 1 (
    echo *** PyInstaller failed.
    exit /b 1
)

echo.
echo === Stage 7: Move exe into ./exe/ and cleanup
if not exist "dist\FaceTrackerPro.exe" (
    echo *** Build claimed success but dist\FaceTrackerPro.exe is missing.
    exit /b 1
)
mkdir exe >nul 2>&1
move /Y "dist\FaceTrackerPro.exe" "exe\FaceTrackerPro.exe" >nul
echo --- exe\FaceTrackerPro.exe placed.

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist FaceTrackerPro.spec del /q FaceTrackerPro.spec

for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

for %%I in ("exe\FaceTrackerPro.exe") do set EXE_SIZE=%%~zI
echo.
echo === Stage 8: Looking for Inno Setup to build a real installer

set ISCC=
where iscc >nul 2>&1 && set ISCC=iscc
if "%ISCC%"=="" if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if "%ISCC%"=="" if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
if "%ISCC%"=="" if exist "%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 5\ISCC.exe"

if "%ISCC%"=="" (
    echo --- Inno Setup not found.
    echo --- To build a real Windows installer ^(setup.exe with a path picker,
    echo     Start Menu shortcuts and uninstaller^), install Inno Setup:
    echo         https://jrsoftware.org/isinfo.php
    echo     Then re-run build.bat.  Or just ship exe\FaceTrackerPro.exe.
    goto :done
)
echo --- Found ISCC at %ISCC%
if exist installer rmdir /s /q installer
%ISCC% installer.iss
if errorlevel 1 (
    echo *** Installer build failed.
    goto :done
)

set SETUP_EXE=
for %%F in (installer\FaceTrackerPro_Setup_*.exe) do set SETUP_EXE=%%F
if not "%SETUP_EXE%"=="" (
    for %%I in ("%SETUP_EXE%") do set SETUP_SIZE=%%~zI
    echo --- Installer built: %SETUP_EXE%   ^(%SETUP_SIZE% bytes^)
)

:done
echo.
echo ============================================================
echo   SUCCESS
echo   exe\FaceTrackerPro.exe   ^(%EXE_SIZE% bytes^)
if not "%SETUP_EXE%"=="" echo   %SETUP_EXE%
echo ============================================================
echo.
endlocal
