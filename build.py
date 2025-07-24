#!/usr/bin/env python3
"""
ImageWeaver Build Script
Creates standalone executables for all platforms
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

# Build configuration
APP_NAME = "ImageWeaver"
VERSION = "1.0.0"
DESCRIPTION = "AI-Powered Image Placement for Translated Documents"

# Paths
ROOT_DIR = Path(__file__).parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"
SPEC_DIR = ROOT_DIR

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_path}")
    
    # Remove .spec files
    for spec_file in SPEC_DIR.glob("*.spec"):
        spec_file.unlink()
        print(f"   Removed {spec_file}")

def check_dependencies():
    """Check if build dependencies are installed"""
    print("🔍 Checking build dependencies...")
    
    try:
        import PyInstaller
        print(f"   ✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("   ❌ PyInstaller not found. Install with: pip install pyinstaller")
        return False
    
    try:
        import customtkinter
        print(f"   ✅ CustomTkinter {customtkinter.__version__}")
    except ImportError:
        print("   ❌ CustomTkinter not found. Install with: pip install customtkinter")
        return False
    
    return True

def build_gui():
    """Build GUI executable"""
    print("🏗️ Building GUI executable...")
    
    # PyInstaller command for GUI
    cmd = [
        "pyinstaller",
        "--name", f"{APP_NAME}",
        "--onefile",
        "--windowed",  # No console window
        "--add-data", "requirements.txt;.",
        "--hidden-import", "customtkinter",
        "--hidden-import", "PIL._tkinter_finder",
        "--collect-all", "customtkinter",
        "--clean",
        "imageweaver_gui.py"
    ]
    
    # Only add icon if it exists
    icon_path = Path("icon.ico")
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   ✅ GUI executable built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ GUI build failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

def build_console():
    """Build Console executable"""
    print("🏗️ Building Console executable...")
    
    # PyInstaller command for console
    cmd = [
        "pyinstaller",
        "--name", f"{APP_NAME}-Console",
        "--onefile",
        "--console",  # Keep console window
        "--add-data", "requirements.txt;.",
        "--clean",
        "imageweaver_console.py"
    ]
    
    # Only add icon if it exists
    icon_path = Path("icon.ico")
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   ✅ Console executable built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Console build failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

def create_release_package():
    """Create release package with documentation"""
    print("📦 Creating release package...")
    
    release_dir = DIST_DIR / f"{APP_NAME}-{VERSION}-{platform.system().lower()}"
    release_dir.mkdir(exist_ok=True)
    
    # Copy executables
    for exe_pattern in ["ImageWeaver*", "imageweaver*"]:
        for exe_file in DIST_DIR.glob(exe_pattern):
            if exe_file.is_file() and exe_file.parent == DIST_DIR:
                shutil.copy2(exe_file, release_dir)
                print(f"   Copied {exe_file.name}")
    
    # Copy documentation
    docs = ["README.md", "LICENSE", "requirements.txt"]
    for doc in docs:
        if Path(doc).exists():
            shutil.copy2(doc, release_dir)
            print(f"   Copied {doc}")
    
    # Create usage instructions
    usage_file = release_dir / "USAGE.txt"
    with open(usage_file, 'w') as f:
        f.write(f"""{APP_NAME} {VERSION}
{DESCRIPTION}

QUICK START:
1. Run {APP_NAME}.exe (GUI) or {APP_NAME}-Console.exe (Command Line)
2. For LLM features, install Ollama: https://ollama.ai
3. See README.md for detailed instructions

FILES:
- {APP_NAME}.exe          - Main GUI application
- {APP_NAME}-Console.exe  - Command line version
- README.md              - Full documentation
- LICENSE                - MIT license
- requirements.txt       - Python dependencies (for reference)

No Python installation required!
""")
    
    print(f"   ✅ Release package created: {release_dir}")
    return release_dir

def main():
    """Main build process"""
    print(f"🎯 Building {APP_NAME} {VERSION}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build executables
    gui_success = build_gui()
    console_success = build_console()
    
    if not (gui_success and console_success):
        print("\n❌ Build failed!")
        sys.exit(1)
    
    # Create release package
    release_dir = create_release_package()
    
    print("\n🎉 Build completed successfully!")
    print(f"📦 Release package: {release_dir}")
    print(f"📁 Individual files: {DIST_DIR}")
    
    # Show file sizes
    print("\n📊 File sizes:")
    for exe_file in DIST_DIR.glob("*"):
        if exe_file.is_file():
            size_mb = exe_file.stat().st_size / (1024 * 1024)
            print(f"   {exe_file.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()