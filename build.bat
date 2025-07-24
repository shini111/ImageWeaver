@echo off
echo ========================================
echo 🏗️ ImageWeaver - Build Executable
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

:: Check if build script exists
if exist "build.py" (
    echo 🚀 Starting build process...
    echo.
    python build.py
) else (
    echo ❌ Build script not found!
    echo Looking for: build.py
    pause
    exit /b 1
)

:: Keep window open to see results
pause