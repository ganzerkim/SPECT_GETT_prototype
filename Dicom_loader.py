# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:20:40 2024

@author: User
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
%matplotlib ipympl
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

from mpl_interactions import image_segmenter


import pydicom
from os import listdir
from os.path import isfile, join
import os
import cv2 as cv
#import SimpleITK as sitk
import pydicom._storage_sopclass_uids

# load a sample image
#import urllib

import PIL

#url = "https://github.com/matplotlib/matplotlib/raw/v3.3.0/lib/matplotlib/mpl-data/sample_data/ada.png"
#image = np.array(PIL.Image.open(urllib.request.urlopen(url)))

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

images_path = "D:\\dicom_data\\SPECT\\GALLBLADDER_EF_20060510_090334_437000\\30_MIN_P_I__0001"

path_tmp = []
name_tmp = []
img_tmp = []

for (path, dir, files) in os.walk(images_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        
        if ext == '.dcm' or '.IMA':
            print("%s/%s" % (path, filename))
            path_tmp.append(path)
            name_tmp.append(filename)

dcm_tmp = []

for i in range(len(path_tmp)):
    dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
    dcm_tmp.append(dcm_p)
    ccc = dcm_p.pixel_array
    img_tmp.append(ccc)
    

img_list = []
dcm_list = []

if len(np.shape(img_tmp[0])) == 3:

    for ii in range(len(img_tmp)):
        for itr in range(len(img_tmp[0])):
            img_list.append(img_tmp[0][itr])
            dcm_list.append(dcm_tmp[ii])
    
else:
    img_list = img_tmp[0]
    dcm_list = dcm_tmp[0]
    

################################################################################################


    
