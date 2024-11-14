# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:20:40 2024

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np

import pydicom
from os import listdir
from os.path import isfile, join
import os
#import cv2 as cv
#import SimpleITK as sitk
import pydicom._storage_sopclass_uids

# load a sample image
#import urllib

import PIL
import cv2 as cv

#url = "https://github.com/matplotlib/matplotlib/raw/v3.3.0/lib/matplotlib/mpl-data/sample_data/ada.png"
#image = np.array(PIL.Image.open(urllib.request.urlopen(url)))

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# A, B detector mean//

images_path = "D:\\dicom_data\\SPECT\\GALLBLADDER_EF_20060510_090334_437000\\Static\\"

path_tmp = []
name_tmp = []


for (path, dir, files) in os.walk(images_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        
        if ext == '.dcm' or '.IMA':
            print("%s/%s" % (path, filename))
            path_tmp.append(path)
            name_tmp.append(filename)
            
#sorted_path = []
#for sorting in sorted(path_tmp, key=int):
#    sorted_path.append(sorting)
    
    
dcm_tmp = []
dcm_tmp.append(pydicom.dcmread(path_tmp[0] + '/' + name_tmp[0], force = True))

for i in range(len(path_tmp)):
    
    dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
    if dcm_p.AcquisitionTime == dcm_tmp[i-1].AcquisitionTime:
        pass
    elif dcm_p.AcquisitionTime > dcm_tmp[i-1].AcquisitionTime:
        dcm_tmp.append(dcm_p)
    elif dcm_p.AcquisitionTime < dcm_tmp[i-1].AcquisitionTime:
        dcm_tmp.insert(i-1, dcm_p)



img_tmp = []

for i in range(len(dcm_tmp)):
    ccc = dcm_tmp[i].pixel_array
    img_tmp.append(ccc)
    

#Static or Dynamic??

img_list = []
dcm_list = []

if len(np.shape(img_tmp[0])) == 3:

    for ii in range(len(img_tmp)):
        for itr in range(len(img_tmp[0])):
            img_list.append(img_tmp[0][itr])
            dcm_list.append(dcm_tmp[ii])
    
else:
    img_list = img_tmp
    dcm_list = dcm_tmp
    

#Static image 합치기 (평균)
def img_mean(img_list):
    img_tmp = 0
    for iii in range(len(img_list)):
        print(iii)
        img_tmp = img_tmp + img_list[iii]
    final_img = img_tmp/len(img_list)
    
    return final_img
    
def img_sum(img_list):
    img_tmp = 0
    for iii in range(len(img_list)):
        print(iii)
        img_tmp = img_tmp + img_list[iii]
    final_img = img_tmp
    
    return final_img

final_img = img_sum(img_list)
#final_img = img_sum(img_list)
#plt.hist(final_img.ravel(), bins=20, range = [0, 20])
#img_scaled = cv.normalize(final_img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)

################################################################################################

import logging
import numpy as np
#import tkinter
import matplotlib
#matplotlib.use("tkAgg")

from matplotlib import pyplot as plt
from roipoly import MultiRoi
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Create image
img = final_img
vmin=0
vmax= 80

# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="gist_yarg", vmin=vmin, vmax=vmax)
plt.title("Click on the button to add a new ROI")

# Draw multiple ROIs
multiroi_named = MultiRoi(roi_names=['My first ROI', 'My second ROI'])

# Draw all ROIs
plt.imshow(img, interpolation='nearest', cmap="hot", vmin=vmin, vmax=vmax)
roi_names = []
roi_tmp = []
for name, roi in multiroi_named.rois.items():
    roi.display_roi()
    roi.display_mean(img)
    roi_tmp.append(roi)
    roi_names.append(name)
plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
plt.show()

################################################################################
def get_roi_img(img, roi_tmp):
    roi_order = []
    for abc in range(len(roi_tmp)):
        a = roi_tmp[abc].get_mask(img)
        roi_order.append(roi_tmp[abc].get_mask(img))
        
    return roi_order

def get_msk_img(img_list, roi_order):
    msk_img = []
    for xyz in range(len(img_list)):
        b = img_list[xyz] * roi_order
        msk_img.append(b)
    return msk_img

        
roi_order = get_roi_img(img, roi_tmp)
msk_img = get_msk_img(img_list, roi_order)

###############################################################################
vmin=0
vmax= 70
plt.figure()
plt.imshow(msk_img[3][0], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)
plt.show()
plt.imshow(msk_img[3][1], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)

############################################################################################
import re
#numbers = re.findall(r'\d+', text)

x = []
y = []
for i in range(len(dcm_list)):
    tt = dcm_list[i].AcquisitionTime
    numbers = re.findall(r'\d+', tt)
    x.append(float(numbers[0]))
    y.append(np.mean(msk_img[i][0])) #0번째 ROI
    
plt.scatter(x, y, cmap = 'black', s = 100)    

x = np.array(x)
y = np.array(y)
fit1 = np.polyfit(x, y, 1, full=True)
print(fit1)
fit2 = np.polyfit(x, y, 2, full=True)
print(fit2)
fit3 = np.polyfit(x, y, 3, full=True)
print(fit3)


num = len(x)
for i in range(num):
    fit = fit1[0][0]*x + fit1[0][1]

plt.yticks(np.arange(0, 4, 0.1))
plt.scatter(x, y, label = 'origin')
plt.plot(x, fit, c ='r', label = 'fitted')
plt.axhline(np.max(y)/2, 0, 1, color = 'lightgray', linestyle = '--', label ="50%", linewidth=1)
plt.legend()
plt.show()

#########################################################
#Exponential or gaussian or logarithm based curve fitting 추가 필요.

