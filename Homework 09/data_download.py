import os
import numpy as np

import urllib
categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
print(categories[:10])
category = 'candle'

####### Set Path ######
path = "C:\Users\henni\Documents\GitHub\ANN_group_5\Homework 09\data"
#######################

if not os.path.isdir(path):
    os.mkdir(path)
    
url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'  
urllib.request.urlretrieve(url, f'{path}/{category}.npy')

images = np.load(f'{path}/{category}.npy')
print(f'{len(images)} images to train on')