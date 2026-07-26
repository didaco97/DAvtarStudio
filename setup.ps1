python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install gdown
python download_models.py
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
cd ..
echo "Setup completed successfully."
