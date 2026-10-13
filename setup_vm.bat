@echo off
echo ============================================================================
echo   Tata Steel TDA Router - A100 Windows RDP VM Setup
echo ============================================================================
echo.
echo [1/3] Installing PyTorch with CUDA GPU support for NVIDIA A100...
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo.
echo [2/3] Installing Hugging Face Transformers, PEFT, and FastAPI...
pip install -r requirements.txt

echo.
echo [3/3] Verifying NVIDIA A100 CUDA visibility...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo.
echo Setup complete! You can now run:
echo   python router.py       (Interactive CLI)
echo   python api_server.py   (REST API Server on port 8000)
pause
