python main.py train --model UNET       --noise Random  --b 32  --epochs 100
python main.py train --model UNET       --noise Line    --b 32  --epochs 100
python main.py train --model UNET       --noise Scar    --b 32  --epochs 100
python main.py train --model UNET       --noise Hum     --b 32  --epochs 100

python main.py train --model REDNET       --noise Random  --b 32  --epochs 100
python main.py train --model REDNET       --noise Line    --b 32  --epochs 100
python main.py train --model REDNET       --noise Scar    --b 32  --epochs 100
python main.py train --model REDNET       --noise Hum     --b 32  --epochs 100

python main.py train --model UNET_REDNET       --noise Random  --b 32  --epochs 100
python main.py train --model UNET_REDNET       --noise Line    --b 32  --epochs 100
python main.py train --model UNET_REDNET       --noise Scar    --b 32  --epochs 100
python main.py train --model UNET_REDNET       --noise Hum     --b 32  --epochs 100

python main.py train --model VDSR       --noise Random  --b 32  --epochs 100
python main.py train --model VDSR       --noise Line    --b 32  --epochs 100
python main.py train --model VDSR       --noise Scar    --b 32  --epochs 100
python main.py train --model VDSR       --noise Hum     --b 32  --epochs 100

python main.py train --model HINET      --noise Random  --b 16  --epochs 100    --neptune offline
python main.py train --model HINET      --noise Line    --b 16  --epochs 100    --neptune offline
python main.py train --model HINET      --noise Scar    --b 16  --epochs 100    --neptune offline
python main.py train --model HINET      --noise Hum     --b 16  --epochs 100    --neptune offline

python main.py train --model MPRNET     --noise Random  --b 4  --epochs 100    --neptune offline
python main.py train --model MPRNET     --noise Line    --b 4  --epochs 100    --neptune offline
python main.py train --model MPRNET     --noise Scar    --b 4  --epochs 100    --neptune offline
python main.py train --model MPRNET     --noise Hum     --b 4  --epochs 100    --neptune offline

python main.py train --model UFORMER    --noise Random  --b 8  --epochs 100    --neptune offline
python main.py train --model UFORMER    --noise Line    --b 8  --epochs 100    --neptune offline
python main.py train --model UFORMER    --noise Scar    --b 8  --epochs 100    --neptune offline
python main.py train --model UFORMER    --noise Hum     --b 8  --epochs 100    --neptune offline

python main.py train --model RESTORMER  --noise Random  --b 1  --epochs 100    --neptune offline
python main.py train --model RESTORMER  --noise Line    --b 1  --epochs 100    --neptune offline
python main.py train --model RESTORMER  --noise Scar    --b 1  --epochs 100    --neptune offline
python main.py train --model RESTORMER  --noise Hum     --b 1  --epochs 100    --neptune offline


