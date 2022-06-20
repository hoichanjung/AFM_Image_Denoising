# -*- coding: utf-8 -*- 
# 참고 : http://gwyddion.net/documentation/user-guide-en/synthetic.html

import gwy
import gwyutils

def get_noise_setting(args):
    
    settings = gwy.gwy_app_settings_get()

    if args.noise_type == "Line": 
        
        settings['/module/lno_synth/initialize'] = True             # Initialize, Replace : 기존 이미지 정보를 가져와서 수행 항상 True
        settings['/module/lno_synth/replace'] = True
        settings['/module/lno_synth/type'] = 0                      # Line (Step) Noise : type = 0
        
        settings['/module/lno_synth/direction'] = 1                 # Direction : 0 : Symmetrical (Pos/Neg 동시 발생, Default) 1 : One Sided Positive 2: One Sided Negative
        settings['/module/lno_synth/distribution'] = 0              # Distribution : 0 = Gaussian (Default), 1 = Exponential, 2 = Uniform, 3 = Triangular
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/cumulative'] = False            # Cumulative = True (Default), Step Noise 누적 옵션
        settings['/module/lno_synth/density'] = 0.02                # Density = 0.1~100 (Default)
        settings['/module/lno_synth/lineprob'] = 0                  # Lineprob = 0 (Default)
        settings['/module/lno_synth/sigma'] = 1.0                   # Sigma = 이미지 전체의 RMS (Root Mean Square) Error 값을 별도로 추출해서 넣으면 적합합니다
            

    elif args.noise_type == "Scar": 
        
        settings['/module/lno_synth/type'] = 1                      # Scar Noise : type = 1

        settings['/module/lno_synth/direction'] = 1                 # Direction : 0 : Symmetrical (Pos/Neg 동시 발생, Default) 1 : One Sided Positive 2: One Sided Negative
        settings['/module/lno_synth/distribution'] = 0              # Distribution : 0 = Gaussian (Default), 1 = Exponential, 2 = Uniform, 3 = Triangular
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/coverage'] = 0.3                # Coverage = 0.1 (Default), Density의 개념과 동일
        settings['/module/lno_synth/length'] = 256                  # Length = 256 (Default), Scar의 길이 값
        settings['/module/lno_synth/length_var'] = 1                # Length_Var = 1 (Default), Scar 길이의 자유도
        settings['/module/lno_synth/sigma'] = 0.30000               # Sigma = 이미지 전체의 RMS (Root Mean Square) Error 값을 별도로 추출해서 넣으면 적합합니다
        

    elif args.noise_type == "Hum": 
        
        settings['/module/lno_synth/type'] = 4                      # Hum Noise : type = 4

        settings['/module/lno_synth/direction'] = 0                 # Direction : 0 : Symmetrical (Pos/Neg 동시 발생, Default) 1 : One Sided Positive 2: One Sided Negative
        settings['/module/lno_synth/distribution'] = 0              # Distribution : 0 = Gaussian (Default), 1 = Exponential, 2 = Uniform, 3 = Triangular
        settings['/module/lno_synth/randomize'] = True

        settings['/module/lno_synth/ncontrip'] = 2         
        settings['/module/lno_synth/spread'] = 1         
        settings['/module/lno_synth/wavelength'] = 10               # Wavelength = 10 (실제와 가장 유사한 사이즈)
        settings['/module/lno_synth/sigma'] = 0.30000              # Sigma = 이미지 전체의 RMS (Root Mean Square) Error 값을 별도로 추출해서 넣으면 적합합니다

    
    elif args.noise_type == "Random": 
                
        settings['/module/noise_synth/density'] = 1                 # Density = 0.1~100 (Default)
        settings['/module/noise_synth/direction'] = 0               # Direction : 0 : Symmetrical (Pos/Neg 동시 발생, Default) 1 : One Sided Positive 2: One Sided Negative
        settings['/module/noise_synth/distribution'] = 0            # Distribution : 0 = Gaussian (Default), 1 = Exponential, 2 = Uniform, 3 = Triangular, 4 = Salt and Pepper
        settings['/module/noise_synth/randomize'] = True
        settings['/module/noise_synth/sigma'] = 0.30000             # Sigma = 이미지 전체의 RMS (Root Mean Square) Error 값을 별도로 추출해서 넣으면 적합합니다

    settings['/module/pixmap/grayscale'] = True