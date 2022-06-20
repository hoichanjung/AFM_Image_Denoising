# Noise Generation

## Requirements
- Python 2.7.18
- pip install -r requirements.txt
- Gwyddion 32 bit Windows (https://sourceforge.net/projects/gwyddion/files/gwyddion/2.60/Gwyddion-2.60.win32.exe/download)
- Pygwy, PyGTK2, PYCairo, PyGObject (https://sourceforge.net/projects/gwyddion/files/pygtk-win32/)


"""python
python main.py --noise_type Line
python main.py --noise_type Scar
python main.py --noise_type Hum
python main.py --noise_type Random
"""