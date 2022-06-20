import torch
import argparse
from models import * 

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='UNET', type=str,
                        help='model name(UNET/HINET/MPRNET/RESTORMER/UFORMER)')
args = parser.parse_args()

nnArchitecture = args.model
imgtransResize = 256
# -------------------- SETTINGS: NETWORK ARCHITECTURE
if nnArchitecture == 'UNET':
    model = Denoising_UNet(in_channels=1, n_classes=1, padding=True).cuda()
elif nnArchitecture == 'REDNET':
    model = REDNet20(in_channels=1, num_layers=6).cuda()
elif nnArchitecture == 'UNET_REDNET':
    model = UNet_REDNet().cuda()                      
elif nnArchitecture == 'VDSR':
    model = VDSR().cuda()            
elif nnArchitecture == 'HINET':
    model = HINet(in_chn = 1).cuda()
elif nnArchitecture == 'MPRNET':
    model = MPRNet(in_c = 1, out_c=1).cuda()      
elif nnArchitecture == 'UFORMER':
    model = Uformer(img_size=imgtransResize, in_chans=1).cuda()
elif nnArchitecture == 'RESTORMER':
    model = Restormer(inp_channels=1, out_channels=1, LayerNorm_type='BiasFree').cuda()      

model = torch.nn.DataParallel(model).cuda()
dummy_input = torch.randn(1, 1, 256, 256, dtype=torch.float).cuda()
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 10
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f'{nnArchitecture} : {mean_syn}')