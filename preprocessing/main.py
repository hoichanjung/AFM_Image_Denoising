import sys
sys.path.append(r'C:\Program Files (x86)\Gwyddion\bin')
sys.path.append(r'C:\Program Files (x86)\Gwyddion\share\gwyddion\pygwy')

import os
import gwy
import gwyutils
import argparse
from utils import get_noise_setting

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type=str, default='Line', help='type of noise (Line/Scar/Hum/Random)')
parser.add_argument('--random_seed', type=int, default=777, help='random seed')
args = parser.parse_args()

def main():

    get_noise_setting(args)

    image_path = "E:/SKhynix/raw_data/AFM4005/" # Path of Raw Data
    os.chdir(image_path)
    image_list = os.listdir(image_path)

    filtered_image_path = "E:/SKhynix/0802_Dataset/Original/" # Path of Filtered Data
    filtered_image_list = os.listdir(filtered_image_path)
    image_list = [image for image in image_list if image in filtered_image_list] # Only use image without noise
   
    if args.noise_type == "Line" or args.noise_type == "Scar" or args.noise_type == "Hum":
        noise_syn = 'lno_synth'
    
    elif args.noise_type == "Random":
        noise_syn = 'noise_synth'

    save_path = os.path.join(image_path, args.noise_type)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_id, image_file in enumerate(image_list):
        noise_save_path = os.path.join(save_path, image_file)
        container = gwy.gwy_file_load(image_file, gwy.RUN_NONINTERACTIVE)
        gwy.gwy_app_data_browser_add(container)
        
        for i in gwy.gwy_app_data_browser_get_data_ids(container):
                
            gwy.gwy_app_data_browser_select_data_field(container, i) # Image File Load
            
            if image_id == 0:
                gwy.gwy_process_func_run(noise_syn, container, gwy.RUN_INTERACTIVE) # Initialize the Setting of Synthesis, Press "Like Current Image" and "Instant updates"
            else:
                gwy.gwy_process_func_run(noise_syn, container, gwy.RUN_IMMEDIATE) # Perform the Setting of Synthesis

            gwy.gwy_file_save(container, noise_save_path, gwy.RUN_NONINTERACTIVE) # Save File
            

if __name__ == '__main__':
    print(vars(args))
    main()
    