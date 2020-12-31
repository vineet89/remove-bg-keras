#%%
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from os import listdir, path
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt #To run as a notebook
import png
import numpy as np

basepath = path.dirname(__file__)
folder_path = path.abspath(path.join(basepath, "..", "sample_images"))

images = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

model = pspnet_50_ADE_20K()

for i in images:

    segmented_path = path.join(folder_path, 'output' ,path.basename(i))


    model.predict_segmentation(
    inp=i,
    out_fname=segmented_path
    )

    image_data = cv2.imread(i)
    segmented_image_data = cv2.imread(segmented_path)

    segmented_image_data = cv2.cvtColor(segmented_image_data, cv2.COLOR_BGR2RGB)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    

    # Colors in segmented image we will use to create a mask
    # These colors correspond to cars/vehicles from ADE250 dataset segmentation
    r1, g1, b1 = 96, 94, 148 
    r2, g2, b2 = 255, 165, 253 

    # Split channel data for segmented image
    red, green, blue = segmented_image_data[:,:,0], segmented_image_data[:,:,1], segmented_image_data[:,:,2]
    ms = (red == r1) & (green == g1) & (blue == b1)
    ms2 = (red == r2) & (green == g2) & (blue == b2)

    # Create a mask image
    mask = np.zeros(segmented_image_data.shape, dtype=np.uint8)

    # Set desired pixels to white
    mask[:,:,:3][ms] = [255, 255, 255]  
    mask[:,:,:3][ms2] = [255, 255, 255]

    result = cv2.bitwise_and(image_data,mask)
    
    cv2.imwrite(segmented_path, result)
# %%
