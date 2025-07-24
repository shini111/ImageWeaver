@echo off
echo ========================================
echo 💻 ImageWeaver - Console Application
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

:: Check if the console app file exists
if exist "imageweaver_console.py" (
    echo 🚀 Starting ImageWeaver Console...
    echo.
    echo Usage: imageweaver_console.py --original [folder] --translated [folder] --output [folder]
    echo.
    echo For help: python imageweaver_console.py --help
    echo.
    
    :: If arguments provided, run with them
    if "%1"=="" (
        echo ℹ️  No arguments provided. Run with --help for usage information.
        python imageweaver_console.py --help
    ) else (
        python imageweaver_console.py %*
    )
) else (
    echo ❌ Console app file not found!
    echo Looking for: imageweaver_console.py
    echo Available Python files:
    dir *.py /b
    echo.
    echo Please make sure the console app file is in this directory
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