#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import json
from keras.models import Model , load_model
from keras.layers import *
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.resnet50 import ResNet50 , preprocess_input


model = load_model("models/model_weights_25.h5")
model_resnet = ResNet50(weights = 'imagenet' , input_shape = (224, 224, 3))

model_final = Model(model_resnet.input , model_resnet.layers[-2].output)


def preprocess_image(img):
    img = img.convert("RGB")  
    img = img.resize((224, 224))  
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img



def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_final.predict(img)  
    return feature_vector


with open("models/word_to_idx.pkl" , "rb") as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("models/idx_to_word.pkl" , "rb") as i2w:
    idx_to_word = pickle.load(i2w)


def predict_caption_using_greedySearch(photo):   
    inp_text = 'startseq'
    max_len = 33
    
    for i in range(max_len):
        sequence = [word_to_idx[word] for word in inp_text.split() if word in word_to_idx]
        sequence = pad_sequences([sequence] , maxlen = max_len , padding = 'post')        
        
        pred_label = model.predict([photo , sequence])
        pred_label = pred_label.argmax()       
        pred_word = idx_to_word[pred_label]    
    
        inp_text += " " + pred_word    
        if pred_word == "endseq":
            break

    final_caption = inp_text.split(' ')[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


def generate_caption(image):  
    enc = encode_image(image) 
    caption = predict_caption_using_greedySearch(enc)  
    return caption
