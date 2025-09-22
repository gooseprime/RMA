#!/usr/bin/env python3
"""
Script to run the Streamlit application for Menstrual Health Analysis
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        # Change to the project directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    print("🩸 Starting Menstrual Health Analysis Dashboard...")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Run Streamlit
    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    run_streamlit()
