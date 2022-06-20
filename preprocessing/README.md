## Download Data
Organic Electronics Morphology (https://drive.google.com/drive/folders/1ChhN62Z_0zL-dqOZt-5UL1vwYGwMM4-5?usp=sharing)
- raw_data.zip (raw data)
- 0802_Dataset.zip (ground-truth and input data)

## Requirements
- Python 2.7.18
- pip install -r requirements.txt
- Download Gwyddion 32 bit Windows (https://sourceforge.net/projects/gwyddion/files/gwyddion/2.60/Gwyddion-2.60.win32.exe/download)
- Download Pygwy, PyGTK2, PYCairo, PyGObject (https://sourceforge.net/projects/gwyddion/files/pygtk-win32/)


```
python main.py --noise_type Line
python main.py --noise_type Scar
python main.py --noise_type Hum
python main.py --noise_type Random
```

