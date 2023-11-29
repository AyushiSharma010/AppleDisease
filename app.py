from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)
MODEL_PATH ='model.h5'
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The disease in this leaf is Alternaria. And the treatment for alternaria is Treatment for Alternaria requires fungicide to be sprayed directly on infected plants, as well as improvements in sanitation and crop rotation to prevent future outbreaks."
    elif preds==1:
        preds="This leaf is healthy and no treatment is required."
    elif preds==2:
        preds="The disease in this leaf is Marsonina and High value trees can be treated with a fungicide application in early spring when the buds start to swell. Applications may have to be repeated at 10- to 14-day intervals in spring as long as wet weather conditions continue. Products with the active ingredient chlorothalonil are effective in controlling this disease."
    elif preds==3:
        preds="In this leaf there is Powdery Mildew disease and You can try to control powdery mildew by: removing infected buds. modifying the environment so that it's less favourable to infection. spraying to protect buds from infection."
    else:
        preds="Enter a valid image"    
    
    
    return preds



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)