# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:57:59 2024

@author: User
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:20:40 2024

@author: User
"""

#####Gastric emptying time tool (GETT)

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


#url = "https://github.com/matplotlib/matplotlib/raw/v3.3.0/lib/matplotlib/mpl-data/sample_data/ada.png"
#image = np.array(PIL.Image.open(urllib.request.urlopen(url)))

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# A, B detector mean//

images_path = "D:\\dicom_data\\SPECT\\goshin_dataset\\"

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
    
    
'''
dcm_tmp = []
dcm_tmp.append(pydicom.dcmread(path_tmp[0] + '/' + name_tmp[0], force = True))

for i in range(len(path_tmp)-1):
    
    dcm_p = pydicom.dcmread(path_tmp[i+1] + '/' + name_tmp[i+1], force = True)
    if dcm_p.AcquisitionTime == dcm_tmp[i].AcquisitionTime:
        dcm_tmp.append(dcm_p)
    elif dcm_p.AcquisitionTime > dcm_tmp[i].AcquisitionTime:
        dcm_tmp.append(dcm_p)
    elif dcm_p.AcquisitionTime < dcm_tmp[i].AcquisitionTime:
        dcm_tmp.insert(i-1, dcm_p)
'''

### Dicom List indexing according to AcquisitionTime, return to dcm_tmp_2
dcm_tmp = []

for i in range(len(path_tmp)):
    dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
    dcm_tmp.append(dcm_p)

#AT = dcm_tmp[3].AcquisitionTime
def indexfinder(AT, dcm_tmp):
    for ii in range(len(dcm_tmp)):
        if AT < dcm_tmp[ii].AcquisitionTime:
            print(ii)
            return ii
        
dcm_tmp_2 = []
dcm_tmp_2.append(dcm_tmp[0])
for i in range(len(dcm_tmp)-1):
    if dcm_tmp[i+1].AcquisitionTime == dcm_tmp_2[i].AcquisitionTime:
        pass
    elif dcm_tmp[i+1].AcquisitionTime > dcm_tmp_2[i].AcquisitionTime:
        dcm_tmp_2.append(dcm_tmp[i+1])
    else:
        AT = dcm_tmp[i+1].AcquisitionTime
        dcm_tmp_2.insert(indexfinder(AT, dcm_tmp_2), dcm_tmp[i+1])

############################################################
# image load
img_tmp = []

for i in range(len(dcm_tmp_2)):
    ccc = dcm_tmp_2[i].pixel_array
    img_tmp.append(ccc)
    

#Static or Dynamic??
#ANT or POS??
'''
img_list = []
dcm_list = []

if len(np.shape(img_tmp[0])) == 3:

    for ii in range(len(img_tmp)):
        for itr in range(len(img_tmp[0])):
            img_list.append(img_tmp[0][itr])
            dcm_list.append(dcm_tmp[ii])
    
else:
    img_list = img_tmp
    dcm_list = dcm_tmp_2
    
'''
#detector = "ANT"
img_list = []
def detector_selection(dcm_tmp_2, detector):
    if detector == "ANT":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append(ccc[0])
            elif position_1 == 'POST':
                img_list.append(ccc[1])
                            
    elif detector == "POST":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append(ccc[1])
            elif position_1 == 'POST':
                img_list.append(ccc[0])
        
    elif detector == "SUM_ANT":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append((ccc[0] + np.flip(ccc[1], 1))/2)
            elif position_1 == 'POST':
                img_list.append((ccc[1] + np.flip(ccc[0], 1))/2)
                
    elif detector == "SUM_POST":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append((np.flip(ccc[0], 1) + ccc[1])/2)
            elif position_1 == 'POST':
                img_list.append((np.flip(ccc[1], 1) + ccc[0])/2)
                
    return img_list

# np.flip(array,0) -> 좌우대칭 RL
# np.flip(array,1) -> 앞뒤대칭 AP 
# np.flip(array,2) -> 상하대칭 IS

#sqrt(Ant * POST)

img_list = detector_selection(dcm_tmp_2, "SUM_ANT")
##ANT, POST, SUM_ANT, SUM_POST


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

#final_img = img_mean(img_list)
final_img = img_sum(img_list)
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
vmax= 900

# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="gist_yarg", vmin=vmin, vmax=vmax)
plt.title("Click on the button to add a new ROI")

# Draw multiple ROIs
multiroi_named = MultiRoi(roi_names=['First ROI', 'Second ROI', 'Background ROI'])

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

# Dicom -----pixel value = counts per minute => CPS convert

###############################################################################
vmin=0
vmax= 400
plt.figure()
plt.imshow(msk_img[3][0], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)
plt.show()
plt.imshow(msk_img[3][1], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)

############################################################################################
import re
#numbers = re.findall(r'\d+', text)
#ddd = [start_time[0][i:i+2] for i in range(0, len(start_time[0]), 2)]
def mean_cps(msk_img, roi_order):
    m_cps = np.sum(msk_img)/np.sum(roi_order)
    
    return m_cps

def decaycorrection(x, y, halflife):
    x = np.array(x)
    y = np.array(y)
    halflife = np.float64(halflife)
    time_passed_hours = x
    decay_constant = 0.693/halflife
    correcty = y * np.exp(-decay_constant * time_passed_hours)
        
    return correcty

roi_number = 0  #0번째 ROI     dual ROI

x = []
y = []
empty = []
background = []
for i in range(len(dcm_tmp_2)):
    start = dcm_tmp_2[0].AcquisitionTime
    start_time = re.findall(r'\d+', start)
    aaa = [start_time[0][i:i+2] for i in range(0, len(start_time[0]), 2)]
    inittime = int(aaa[0]) * 60 + int(aaa[1])
    tt = dcm_tmp_2[i].AcquisitionTime
    numbers = re.findall(r'\d+', tt)
    bbb = [numbers[0][i:i+2] for i in range(0, len(numbers[0]), 2)]
    actime = int(bbb[0]) * 60 + int(bbb[1])
    x.append(actime - inittime)
    m_cps = mean_cps(msk_img[i][roi_number], roi_order[roi_number])
    f_cps = mean_cps(msk_img[0][roi_number], roi_order[roi_number])
    y.append(m_cps)
    
    empty.append((f_cps - m_cps)/f_cps * 100)
    background.append(np.mean(msk_img[i][2]))

halflife = 131
correcty = decaycorrection(x, y, halflife)

plt.scatter(x, correcty, cmap = 'black', s = 100)

y = correcty

#polynomial fit
x = np.array(x)
y = np.array(y)
fit1 = np.polyfit(x, y, 1, full=True)
print(fit1)
fit2 = np.polyfit(x, y, 2, full=True)
print(fit2)
fit3 = np.polyfit(x, y, 3, full=True)
print(fit3)
fitlog = np.polyfit(x, np.log(y), 1)
print(fitlog)
# y ≈ 8.46 log(x) + 6.62

num = len(x)
for i in range(num):
    fit = fit3[0][0]*x*x*x + fit3[0][1]*x*x + fit3[0][2]*x + fit3[0][3]
'''
num = len(x)
for i in range(num):
    fit = fitlog[0]*x + fitlog[1]
'''

#plt.yticks(np.arange(0, 4, 0.1))
plt.scatter(x, y, cmap = 'black', s = 100)
plt.plot(x, y, label = 'Origin', c = 'Blue')
plt.plot(x, fit, c ='r', label = 'Fitted')
plt.plot(x, background, c = 'yellowgreen', label = 'Background')
plt.axhline(np.max(y)/2, color = 'lightgray', linestyle = '--', label ="50%", linewidth = 1)
plt.axvline(halflife, color = 'red', linestyle = ':', label ="99m-Tc T(1/2)", linewidth = 1)
plt.legend()
plt.show()

#########################################################
#Exponential or gaussian or logarithm based curve fitting 추가 필요.
from scipy.optimize import curve_fit
# Inverse Logistic Function 
# https://en.wikipedia.org/wiki/Logistic_function
def func(x, L ,x0, k, b):
    y = 1/(L / (1 + np.exp(-k*(x-x0)))+b)
    return y

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

#############logarithm###################################################
# FIT DATA
p0 = [max(y), np.median(x), 1, min(y)] # this is an mandatory initial guess

popt, pcov = curve_fit(func, x, y, p0, method='trf', maxfev=10000, sigma=None, absolute_sigma=False)
#method = trf, dogbox, lm
# PERFORMANCE

modelPredictions = func(x, *popt)
absError = modelPredictions - y
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(y))

print('Parameters:', popt)
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

#PLOT

plt.figure()
#plt.plot(x, y, 'ko', label="Original Noised Data")
plt.scatter(x, y, cmap = 'skyblue', s = 100)
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve", c = 'red')
#plt.yscale('log')
#plt.xscale('log')
plt.plot(x, y, label = 'origin', c = 'Blue')
plt.plot(x, background, c = 'yellowgreen', label = 'Background')
plt.axhline(np.max(y)/2, color = 'lightgray', linestyle = '--', label ="50%", linewidth = 1)
plt.axvline(131, color = 'red', linestyle = ':', label ="99m-Tc T(1/2)", linewidth = 1)
plt.legend()
plt.show()


#############Sigmoid###################################################
# FIT DATA
p0 = [max(y), np.median(x), 0.5, min(y)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, x, y, p0, method='trf', maxfev=10000, sigma=None, absolute_sigma=False)
#method = trf, dogbox, lm
# PERFORMANCE

modelPredictions = sigmoid(x, *popt)
absError = modelPredictions - y
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(y))

print('Parameters:', popt)
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

#PLOT

plt.figure()
#plt.plot(x, y, 'ko', label="Original Noised Data")
plt.scatter(x, y, cmap = 'skyblue', s = 100)
plt.plot(x, sigmoid(x, *popt), 'r-', label="Fitted Curve", c = 'red')
#plt.yscale('log')
#plt.xscale('log')
plt.plot(x, y, label = 'origin', c = 'Blue')
plt.plot(x, background, c = 'yellowgreen', label = 'Background')
plt.axhline(np.max(y)/2, color = 'lightgray', linestyle = '--', label ="50%", linewidth = 1)
plt.axvline(131, color = 'red', linestyle = ':', label ="99m-Tc T(1/2)", linewidth = 1)
plt.legend()
plt.show()

########################################################################
import pandas as pd

data_sheet1 = {'Patient Name' : dcm_p.PatientName,
               'Patient ID' : dcm_p.PatientID,
               'Study Date' : dcm_p.StudyDate,
               'Study Time' : dcm_p.StudyTime,
               'Study Name' : dcm_p.StudyDescription,
               'Energy Window Name' : dcm_p.EnergyWindowInformationSequence[0].EnergyWindowName,
               'Begin Time' : x[0], 'End Time' : x[-1], 'T 1/2' : halflife}
data_sheet2 = {'time (min)' : x,'cps' : y,
               '%empty' : empty,
               '%retention' : 100-np.array(empty),
               'background' : background, 
               'decay corrected' : correcty,
               'polynomial fit' : fit,
               'Exponential fit' : func(x, *popt),
               'sigmoid fit' : sigmoid(x, *popt),
               'RMSE': RMSE, 'R-squared' : Rsquared
               } #리스트 자료형으로 생성 
excel1 = pd.DataFrame(data_sheet1) #데이터 프레임으로 전환 및 생성
excel2 = pd.DataFrame(data_sheet2)
xlxs_dir='D:\\dicom_data\\SPECT\\example.xlsx' #경로 및 파일명 설정
with pd.ExcelWriter(xlxs_dir) as writer:
    excel1.to_excel(writer, sheet_name = 'Header') #raw_data1 시트에 저장
    excel2.to_excel(writer, sheet_name = 'Analysis')