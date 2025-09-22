@echo off
echo ========================================
echo  Menstrual Health Analysis Dashboard
echo ========================================
echo.
echo Starting the analysis application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "data\DATA SHEET.xlsx" (
    echo ERROR: Data file not found
    echo Please ensure you're running this from the project root directory
    echo and that "data\DATA SHEET.xlsx" exists
    pause
    exit /b 1
)

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Run Streamlit app
echo.
echo ========================================
echo  Launching Streamlit Dashboard
echo ========================================
echo.
echo The application will open in your default web browser
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py --server.port 8501 --server.address localhost

echo.
echo Application stopped.
pause
