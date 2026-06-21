@echo off
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "ultralytics[export]"
echo Training/export dependencies installed.
