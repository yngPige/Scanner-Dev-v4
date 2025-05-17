@echo off
REM === Build script for 3lacksScanner PyQt6 application ===

REM Step 1: Activate venv if exists
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Activated virtual environment.
) ELSE (
    echo [INFO] No virtual environment found. Using system Python.
)

REM Step 2: Install dependencies
IF EXIST requirements.txt (
    echo Installing dependencies from requirements.txt ...
    pip install -r requirements.txt
) ELSE (
    echo [WARNING] requirements.txt not found. Skipping dependency installation.
)

REM Step 3: Check for assets/icon.png
REM (Removed icon check as icon is no longer required)

REM Step 4: Build executable with PyInstaller
IF NOT EXIST 3lacksScanner.spec (
    echo [ERROR] 3lacksScanner.spec not found! Please ensure the spec file is present.
    pause
    exit /b 1
)

pyinstaller 3lacksScanner.spec

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

REM Step 5: Notify user of output location
echo.
echo [SUCCESS] Build complete! Look for your executable in the 'dist' folder.
echo To run: dist\3lacksScanner\3lacksScanner.exe
pause 