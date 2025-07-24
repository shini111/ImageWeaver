@echo off
echo ========================================
echo 🔧 ImageWeaver - Setup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    echo Download from: https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

:: Check if tkinter is available
echo.
echo 🔍 Checking tkinter availability...
python -c "import tkinter; print('✅ Tkinter is available')" 2>nul
if errorlevel 1 (
    echo ❌ Tkinter is not available
    echo Please install Python with tkinter support
    echo On Ubuntu/Debian: sudo apt-get install python3-tk
    echo On other systems, reinstall Python with tkinter
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

:: Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo 📚 Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Installing core dependencies...
    pip install customtkinter>=5.2.2
    pip install beautifulsoup4>=4.12.0
    pip install lxml>=4.9.0
    pip install requests>=2.31.0
    pip install pathlib2>=2.3.7
)

if errorlevel 1 (
    echo ❌ Failed to install some packages
    echo You may need to install them manually
) else (
    echo ✅ All packages installed successfully
)

echo.
echo 🎉 Setup complete!
echo.
echo Available commands:
echo   • run_gui.bat           - Launch GUI application
echo   • run_console.bat       - Launch command-line version  
echo   • activate_env.bat      - Activate environment for development
echo   • build.bat             - Build executable
echo.
pause