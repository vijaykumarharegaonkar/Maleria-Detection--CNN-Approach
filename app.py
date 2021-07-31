#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 17:11:32 2021

@author : vijayharegaonkar
"""

#from _future_ import division,print_function
import os
import glob
import re
import numpy as np

# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils

from flask import Flask,redirect,url_for,request,render_template
from werkzeug.utils import secure_filename

# define the flask app
app = Flask(__name__)

# model saved with keras model.save()
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

Model_path = 'model_vgg19.h5'

# Load your trained model

model = load_model(Model_path)



def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(244,244))
    
    # preprocessing the image
    
    x = image.img_to_array(img)
    
    # scaling
    
    x= x/255
    
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    
    preds = model.predict(x)
    
    preds = np.argmax(preds,axis=1)
    if preds==0:
        preds = "the Person is Infected with Maleria"
    else:
        preds = "The Person is not infected with Maleria"       
    
    return preds
    
    
    
@app.route('/', methods=['GET'])

def index():
    # main Page
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def upload():
    
    if request.method == 'POST':
        # get the file from post request
        f = request.files['file']
        
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        
        #make prediction
        
        preds = model_predict(file_path,model)
        result = preds
        return result
    
    return None


if app == '__main__':
    app.run(debug=True)
        

    
    



