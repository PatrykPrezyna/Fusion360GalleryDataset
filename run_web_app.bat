@echo off
echo Starting Image Retrieval Web Application...
echo.
echo Make sure you have:
echo   1. Installed all dependencies: pip install -r requirements.txt
echo   2. Prepared image pools in output_data folder
echo   3. (Optional) Trained VAE models in pool directories
echo.
echo Opening browser at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py
