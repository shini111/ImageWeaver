@echo off
echo ========================================
echo 🔧 ImageWeaver - Development Environment
echo ========================================

if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo ✅ Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🎯 Development environment activated!
echo Available commands:
echo   • python imageweaver_gui.py          - Run GUI
echo   • python imageweaver_console.py      - Run console
echo   • python build_py.py                 - Build executable
echo   • pip install -r requirements-dev.txt - Install dev dependencies
echo.

cmd /k