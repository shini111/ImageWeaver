@echo off
echo ========================================
echo 🧪 ImageWeaver - Run Tests
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

echo 🔍 Testing basic imports...
python -c "import imageweaver_console; print('✅ Console import OK')"
python -c "import imageweaver_gui; print('✅ GUI import OK')" 2>nul || echo "⚠️ GUI import failed (might need display)"

echo.
echo 🔍 Testing dependencies...
python -c "import customtkinter; print('✅ CustomTkinter OK')"
python -c "import bs4; print('✅ BeautifulSoup OK')"
python -c "import requests; print('✅ Requests OK')"

echo.
echo 🔍 Testing console help...
python imageweaver_console.py --help

echo.
echo ✅ Basic tests completed!
pause