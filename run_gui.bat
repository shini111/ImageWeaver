@echo off
echo ========================================
echo 🖥️ ImageWeaver - GUI Application
echo ========================================

:: Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check if the main app file exists
if exist "imageweaver_gui.py" (
    echo 🚀 Starting ImageWeaver GUI...
    echo.
    python imageweaver_gui.py
) else (
    echo ❌ GUI app file not found!
    echo Looking for: imageweaver_gui.py
    echo Available Python files:
    dir *.py /b
    echo.
    echo Please make sure the GUI app file is in this directory
    pause
    exit /b 1
)

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ❌ Application encountered an error
    echo Check the error messages above
    pause
)