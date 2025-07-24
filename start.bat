@echo off
echo ========================================
echo 🎯 ImageWeaver - Quick Start
echo ========================================
echo.

:: Check if setup has been run
if not exist "venv" (
    echo 📦 First time setup detected...
    echo Running setup.bat...
    echo.
    call setup.bat
    
    if errorlevel 1 (
        echo ❌ Setup failed!
        pause
        exit /b 1
    )
)

:: Menu for user choice
echo.
echo 🚀 Choose how to start ImageWeaver:
echo.
echo 1. GUI Application (Recommended)
echo 2. Console Application  
echo 3. Development Environment
echo 4. Run Tests
echo 5. Build Executable
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    call run_gui.bat
) else if "%choice%"=="2" (
    call run_console.bat
) else if "%choice%"=="3" (
    call activate_env.bat
) else if "%choice%"=="4" (
    call test.bat
) else if "%choice%"=="5" (
    call build.bat
) else (
    echo ❌ Invalid choice. Running GUI by default...
    call run_gui.bat
)