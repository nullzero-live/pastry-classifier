'''PyTorch Food Classifier - FastAI 2022- Mostly Good For Pastries
and trained on ResNet 34'''

import streamlit as st
import os
from PIL import Image
import time
from fastai.vision.all import *
from fastai.learner import load_learner

def GetLabel(img):
    return img.split('-')[0]
    

#Load the Learner (Exported from ipnyb file with learn.export() )
learn = load_learner('export.pkl')


#Classify image
def classify_image(cl_img):
    img = Image.open(cl_img)
    st.image(img)
    pred, _ , _ = learn.predict(img)
    return pred


    
    
st.set_page_config(page_title="PyTorch Food Classifier - FastAI 2022", page_icon=":robot:")
st.header("PyTorch Food Classifier")

file_up = st.file_uploader("Upload Your Food Image Below", type=["jpg","png"])

if st.button('Run Model'):
    st.write("Button Pressed")
    cl_done = classify_image(file_up)
    
    st.write(f"Your food is:  {cl_done}")

st.write('This classifier is trained on Resnet-34 and works primarily for three classes. Donuts, CheeseCake and Panna Cotta (HOTDOGNOTHOTDOG).\n\n Thankyou to FastAI for the exercise.')
