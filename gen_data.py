import tensorflow as tf
import os
import cv2 
import numpy as np 
import math 
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

path = '/media/dl/DL/aashish/upscaling/images/bill/'
files = os.listdir(path)
for f in files:

	img = Image.open(path + f)
	img.load
	#print(img.size)
	img16 = img.resize((24*8, 24*8), Image.ANTIALIAS)

	img16.save('/media/dl/DL/aashish/upscaling/images/hr/' + f)
