python main.py test --model HINET      --noise Random  --b 16   --ckpt Denoising_AFM_HINET_Random_exp1_0214 --neptune offline
python main.py test --model HINET      --noise Line    --b 16   --ckpt Denoising_AFM_HINET_Line_exp1_0214 
python main.py test --model HINET      --noise Scar    --b 16   --ckpt Denoising_AFM_HINET_Scar_exp1_0214 
python main.py test --model HINET      --noise Hum     --b 16   --ckpt Denoising_AFM_HINET_Hum_exp1_0215 

python main.py test --model MPRNET     --noise Random  --b 4    --ckpt Denoising_AFM_MPRNET_Random_exp1_0215 
python main.py test --model MPRNET     --noise Line    --b 4    --ckpt Denoising_AFM_MPRNET_Line_exp1_0215 
python main.py test --model MPRNET     --noise Scar    --b 4    --ckpt Denoising_AFM_MPRNET_Scar_exp1_0215 
python main.py test --model MPRNET     --noise Hum     --b 4    --ckpt Denoising_AFM_MPRNET_Hum_exp1_0215 

python main.py test --model UFORMER    --noise Random  --b 8    --ckpt Denoising_AFM_UFORMER_Random_exp1_0216 
python main.py test --model UFORMER    --noise Line    --b 8    --ckpt Denoising_AFM_UFORMER_Line_exp1_0216 
python main.py test --model UFORMER    --noise Scar    --b 8    --ckpt Denoising_AFM_UFORMER_Scar_exp1_0216 
python main.py test --model UFORMER    --noise Hum     --b 8    --ckpt Denoising_AFM_UFORMER_Hum_exp1_0216 

python main.py test --model RESTORMER  --noise Random  --b 1    --ckpt Denoising_AFM_RESTORMER_Random_exp1_0216 
python main.py test --model RESTORMER  --noise Line    --b 1    --ckpt Denoising_AFM_RESTORMER_Line_exp1_0217 
python main.py test --model RESTORMER  --noise Scar    --b 1    --ckpt Denoising_AFM_RESTORMER_Scar_exp1_0217 
python main.py test --model RESTORMER  --noise Hum     --b 1    --ckpt Denoising_AFM_RESTORMER_Hum_exp1_0217 
