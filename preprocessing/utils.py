# -*- coding: utf-8 -*- 
# Reference : http://gwyddion.net/documentation/user-guide-en/synthetic.html

import gwy
import gwyutils

def get_noise_setting(args):
    
    settings = gwy.gwy_app_settings_get()

    if args.noise_type == "Line": 
        
        settings['/module/lno_synth/initialize'] = True             
        settings['/module/lno_synth/replace'] = True                
        settings['/module/lno_synth/type'] = 0                      
        
        settings['/module/lno_synth/direction'] = 1     
        settings['/module/lno_synth/distribution'] = 0  
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/cumulative'] = False
        settings['/module/lno_synth/density'] = 0.02    
        settings['/module/lno_synth/lineprob'] = 0      
        settings['/module/lno_synth/sigma'] = 1.0       
            

    elif args.noise_type == "Scar": 
        
        settings['/module/lno_synth/type'] = 1          

        settings['/module/lno_synth/direction'] = 1     
        settings['/module/lno_synth/distribution'] = 0  
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/coverage'] = 0.3    
        settings['/module/lno_synth/length'] = 256      
        settings['/module/lno_synth/length_var'] = 1    
        settings['/module/lno_synth/sigma'] = 0.30000   
        

    elif args.noise_type == "Hum": 
        
        settings['/module/lno_synth/type'] = 4          

        settings['/module/lno_synth/direction'] = 0     
        settings['/module/lno_synth/distribution'] = 0  
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/ncontrip'] = 2         
        settings['/module/lno_synth/spread'] = 1         
        settings['/module/lno_synth/wavelength'] = 10               
        settings['/module/lno_synth/sigma'] = 0.30000   

    
    elif args.noise_type == "Random": 
                
        settings['/module/noise_synth/density'] = 1     
        settings['/module/noise_synth/direction'] = 0   
        settings['/module/noise_synth/distribution'] = 0
        settings['/module/noise_synth/randomize'] = True
        settings['/module/noise_synth/sigma'] = 0.30000 

    settings['/module/pixmap/grayscale'] = True