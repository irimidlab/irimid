@echo off
cd /d "%~dp0"
python face_analyzer\scripts\download_model.py
streamlit run face_analyzer\main.py
