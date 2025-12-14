@echo off
cd /d "C:\Users\USUARIO\Desktop\machine lerarning proyecto appi\Proyecto_api_ML\Proyecto_api_ML-Yolo"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -q numpy opencv-python face_recognition || python -m pip install -q numpy opencv-python
python "proyecto apI reconocimiento.py"
pause
