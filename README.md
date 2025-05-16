## Prepare Development Environment

Right now, this project requires an nvidia GPU and CUDA drivers.

This project was built using python 3.10.11.

Install the included requirements.txt (pip install -r requirements.txt)

Since it using CUDA, the default torch libraries have to be uninstalled.  To do this run:

pip uninstall -y torch

pip uninstall -y torchvision

pip uninstall -y torchaudio

Then install CUDA versions of these libraries.

pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

pip install torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

Run the app by entering the "python ./src/app.py" from the root project directory.
