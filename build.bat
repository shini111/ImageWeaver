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
if exist "build_py.py" (
    echo 🚀 Starting build process...
    echo.
    python build_py.py
) else (
    echo ❌ Build script not found!
    echo Looking for: build_py.py
    pause
    exit /b 1
)

:: Keep window open to see results
pause