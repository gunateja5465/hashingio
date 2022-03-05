import imp
from django.shortcuts import render

from django.shortcuts import render
from .models import Pic
import os
import glob
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from django.conf import settings
from .static.datasets.div2k.parameters import Div2kParameters 
from .static.models.srresnet import build_srresnet
from .static.models.pretrained import pretrained_models
from .static.utils.prediction import get_sr_image
from .static.utils.config import config

def index(request):
    MEDIA_ROOT=settings.MEDIA_ROOT
    STATIC_ROOT=settings.STATIC_ROOT
    if request.method=='GET':
        print(MEDIA_ROOT)
        return render(request,'project/index.html')  
    else:
        dataset_key = "bicubic_x4"

        data_path = config.get("data_path", "") 

        div2k_folder = os.path.abspath(os.path.join(data_path, "./static/datasets/div2k"))

        dataset_parameters = Div2kParameters(dataset_key, save_data_directory=div2k_folder)

        print('hello')
        f=request.FILES['blurimage']
        
        obj=Pic(image=f)
        obj.save()
       
        

        def test(model_key,path):
            

            def load_image(path):
                print('loaded weigkljskldjaldkjaskldjsalkjdklsdj')
                img = Image.open(path)
    
                was_grayscale = len(img.getbands()) == 1
    
                if was_grayscale or len(img.getbands()) == 4:
                    img = img.convert('RGB')
                return was_grayscale, np.array(img)
            print(MEDIA_ROOT)
            weights_directory = MEDIA_ROOT+'\weights\srgan_bicubic_x4'

            file_path =  MEDIA_ROOT+"\weights\srgan_bicubic_x4\generator.h5"

            model = build_srresnet(scale=dataset_parameters.scale)

            weights_file = f"{weights_directory}\generator.h5"

            model.load_weights(weights_file)
            print('loaded weights')
            results_path = MEDIA_ROOT+f"/output/{model_key}/"
            image_paths = glob.glob(path)
            print(image_paths)

            was_grayscale, lr = load_image(MEDIA_ROOT+path.replace('/media',''))

            print('getting')
    
            sr = get_sr_image(model, lr)
            print('loa')
            if was_grayscale:
                sr = ImageOps.grayscale(sr)
    
            image_name = path.split("/")[-1]
            sr.save(f"{results_path}{image_name}" )
            

        model_name = "srgan"
        model_key = f"{model_name}_{dataset_key}"
        path=obj.image.url
        test(model_key,path)
        print(MEDIA_ROOT)
        return render(request, 'project/index.html',{'context':str("media/output/srgan_bicubic_x4/"+obj.image.name)})
        
