@echo off
REM Real-Time AI Scene Description System - Windows Setup Script

echo 🤖 Real-Time AI Scene Description System - Setup
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo ❌ Python 3.8+ is required
    pause
    exit /b 1
)

echo ✅ Python version is compatible

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📈 Upgrading pip...
python -m pip install --upgrade pip

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ NVIDIA GPU detected
    echo 📦 Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo ⚠️ No NVIDIA GPU detected, using CPU version
    echo 📦 Installing PyTorch CPU version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Install other requirements
echo 📦 Installing other dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Test installation
echo 🧪 Testing installation...
python -c "import torch, cv2, transformers, streamlit, fastapi; print('✅ All dependencies working!')"

if %errorlevel% neq 0 (
    echo ❌ Installation test failed
    pause
    exit /b 1
)

REM Create batch files for easy running
echo 🔗 Creating run scripts...

echo @echo off > run_streamlit.bat
echo call venv\Scripts\activate.bat >> run_streamlit.bat
echo python -m src.main --mode streamlit >> run_streamlit.bat
echo pause >> run_streamlit.bat

echo @echo off > run_cli.bat
echo call venv\Scripts\activate.bat >> run_cli.bat
echo python -m src.main --mode cli >> run_cli.bat
echo pause >> run_cli.bat

echo @echo off > run_api.bat
echo call venv\Scripts\activate.bat >> run_api.bat
echo python -m src.main --mode api >> run_api.bat
echo pause >> run_api.bat

echo @echo off > run_legacy.bat
echo call venv\Scripts\activate.bat >> run_legacy.bat
echo python app.py >> run_legacy.bat
echo pause >> run_legacy.bat

echo.
echo 🎉 Installation completed successfully!
echo ==================================================
echo.
echo Quick Start:
echo   run_streamlit.bat   - Launch web interface
echo   run_cli.bat         - Launch CLI version  
echo   run_api.bat         - Launch API server
echo   run_legacy.bat      - Original simple version
echo.
echo Manual Commands:
echo   venv\Scripts\activate.bat
echo   python -m src.main --mode streamlit
echo.
echo Documentation:
echo   README.md - Full documentation
echo   CONTRIBUTING.md - Development guide
echo.
echo Have fun with your AI scene description system! 🚀
echo.
pause
