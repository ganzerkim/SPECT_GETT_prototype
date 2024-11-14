# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:57:59 2024

@author: User
"""

#####Gastric emptying time tool (GETT)

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from os import listdir
from os.path import isfile, join
import os
import pydicom._storage_sopclass_uids
import re
import logging
from roipoly import MultiRoi
from scipy.optimize import curve_fit
import pandas as pd

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# A, B detector mean//
images_path = "D:\\dicom_data\\SPECT\\MotionCorrected\\MotionCorrected\\"

path_tmp = []
name_tmp = []

for (path, dir, files) in os.walk(images_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        
        if ext == '.dcm' or '.IMA':
            print("%s/%s" % (path, filename))
            path_tmp.append(path)
            name_tmp.append(filename)
            

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

#detector = "ANT"

def detector_selection(dcm_tmp_2, detector):
    img_list = []
    img_geo =[]
    if detector == "ANT":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append(ccc[0])
                img_geo.append(ccc)
            elif position_1 == 'POST':
                img_list.append(ccc[1])
                img_geo.append(ccc)
                            
    elif detector == "POST":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append(ccc[1])
                img_geo.append(ccc)
            elif position_1 == 'POST':
                img_list.append(ccc[0])
                img_geo.append(ccc)
        
    elif detector == "SUM_ANT":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append((ccc[0] + np.flip(ccc[1], 1))/2)
                ccc[1] = np.flip(ccc[1], 1)
                img_geo.append(ccc)
                
            elif position_1 == 'POST':
                img_list.append((ccc[1] + np.flip(ccc[0], 1))/2)
                ccc[0] = np.flip(ccc[0], 1)
                img_geo.append(ccc)
                
    elif detector == "SUM_POST":
        for i in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[i].pixel_array
            position_1 = dcm_tmp_2[i].DetectorInformationSequence[0].ViewCodeSequence[0].CodeValue.split('-')[0]
            position_2 = dcm_tmp_2[i].DetectorInformationSequence[1].ViewCodeSequence[0].CodeValue.split('-')[0]
        
            if position_1 == 'ANT':
                img_list.append((np.flip(ccc[0], 1) + ccc[1])/2)
                ccc[0] = np.flip(ccc[0], 1)
                img_geo.append(ccc)
                
            elif position_1 == 'POST':
                img_list.append((np.flip(ccc[1], 1) + ccc[0])/2)
                ccc[1] = np.flip(ccc[1], 1)
                img_geo.append(ccc)
    return img_list, img_geo

# np.flip(array,0) -> 좌우대칭 RL
# np.flip(array,1) -> 앞뒤대칭 AP 
# np.flip(array,2) -> 상하대칭 IS

#sqrt(Ant * POST)

img_list, img_geo = detector_selection(dcm_tmp_2, "SUM_ANT")
##ANT, POST, SUM_ANT, SUM_POST
#img_geo는 A, B detector image set.

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
plt.imshow(img, interpolation='lanczos', cmap="hot", vmin=vmin, vmax=vmax)
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
        if len(img_list[xyz]) == 1:
            b = img_list[xyz] * roi_order
            msk_img.append(b)
        elif len(img_list[xyz]) == 2:
            ccc = []
            ccc.append(img_list[xyz][0] * roi_order)
            ccc.append(img_list[xyz][1] * roi_order)
            msk_img.append(ccc)
    return msk_img

        
roi_order = get_roi_img(img, roi_tmp)
msk_img = get_msk_img(img_geo, roi_order) #msk_img = A detecor / B detector 's 3 ROI set
# [Frame number][Detector Number][ROI number]

# Dicom -----pixel value = counts per minute => CPS convert

###############################################################################
'''
vmin=0
vmax= 400
plt.figure()
plt.imshow(msk_img[3][0], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)
plt.show()
plt.imshow(msk_img[3][1], interpolation='nearest', cmap="inferno", vmin=vmin, vmax=vmax)
'''
############################################################################################

#numbers = re.findall(r'\d+', text)
#ddd = [start_time[0][i:i+2] for i in range(0, len(start_time[0]), 2)]
def roi_cps(msk_img, roi_order):
    m_cpm = np.sum(msk_img)
    m_cps = m_cpm / 60
    
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

def countcal(roi_number, dcm_tmp_2):
    x = []
    a = []
    b = []
    background_a = []
    background_b = []
    
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
        
        
        m_cps_a = roi_cps(msk_img[i][0][roi_number], roi_order[roi_number])
        m_cps_b = roi_cps(msk_img[i][1][roi_number], roi_order[roi_number])
        a.append(m_cps_a)
        b.append(m_cps_b)
    
        background_a.append((np.sum(msk_img[i][0][2])/60))
        background_b.append((np.sum(msk_img[i][1][2])/60))
        
    return x, a, b, background_a, background_b

x, a, b, background_a, background_b = countcal(0, dcm_tmp_2)
x2, a2, b2, background_a2, background_b2 = countcal(1, dcm_tmp_2)

#if decay correction is on
y = np.sqrt(np.array(a) * np.array(b))
background = np.sqrt(np.array(background_a) * np.array(background_b))
#if decay correction is off
#y = (np.array(a) + np.array(b)) / 2
#background = (np.array(background_a) + np.array(background_b)) / 2
y2 = np.sqrt(np.array(a2) * np.array(b2))
background2 = np.sqrt(np.array(background_a2) * np.array(background_b2))

halflife = 360.1

correcty = decaycorrection(x, y, halflife) #if decy correction is on
correcty2 = decaycorrection(x2, y2, halflife) #if decay correction is on

correcty = correcty - background #if background correction is on
correcty2 = correcty2 - background2 #if background correction is on

#plt.figure()
#plt.scatter(x, correcty, c = 'blue', s = 100)
#plt.figure()
#plt.scatter(x, correcty2, c = 'red', s = 100)

y = correcty
y2 = correcty2

def polynominalfit(x, y, x2, y2):
    #polynomial fit
    x = np.array(x)
    y = np.array(y)
    x2 = np.array(x2)
    y2 = np.array(y2)
    
    #fit1 = np.polyfit(x, y, 1, full=True)
    #print(fit1)
    #fit2 = np.polyfit(x, y, 2, full=True)
    #print(fit2)
    fit3 = np.polyfit(x, y, 3, full=True)
    print(fit3)
    fit3_2 = np.polyfit(x2, y2, 3, full=True)
    print(fit3_2)
    #fitlog = np.polyfit(x, np.log(y), 1)
    #print(fitlog)
    # y ≈ 8.46 log(x) + 6.62
    
    num = len(x)
    for i in range(num):
        fit = fit3[0][0]*x*x*x + fit3[0][1]*x*x + fit3[0][2]*x + fit3[0][3]
    
    num = len(x)
    for i in range(num):
        fit2 = fit3_2[0][0]*x*x*x + fit3_2[0][1]*x*x + fit3_2[0][2]*x + fit3_2[0][3]
        
        
        
    modelPredictions = fit
    absError = modelPredictions - y
    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y))
    
    modelPredictions2 = fit2
    absError2 = modelPredictions2 - y2
    SE2 = np.square(absError2) # squared errors
    MSE2 = np.mean(SE2) # mean squared errors
    RMSE2 = np.sqrt(MSE2) # Root Mean Squared Error, RMSE
    Rsquared2 = 1.0 - (np.var(absError2) / np.var(y2))
   
    #plt.yticks(np.arange(0, 4, 0.1))
    plt.figure()
    plt.scatter(x, y, c = 'blue', s = 100)
    plt.plot(x, y, label = 'ROI_1', c = 'Skyblue', linestyle = '--')
    plt.plot(x, fit, c ='Blue', label = 'Fitted curve_1 (Polynominal)')
    
    plt.scatter(x2, y2, c = 'Green', s = 100)
    plt.plot(x2, y2, label = 'ROI_2', c = 'Yellowgreen', linestyle = '--')
    plt.plot(x2, fit2, c ='Green', label = 'Fitted curve_2 (Polynominal)')
    
    plt.plot(x, background, c = 'Red', label = 'Background', linestyle = '--')
    
    base1 = np.max(np.where( y > y[0]/2))
    halfT1 = x[base1] + (y[0]/2 - y[base1 + 1]) / (y[base1] - y[base1 + 1]) * (x[base1 + 1] - x[base1])
    base2 = np.max(np.where( y > y2[0]/2))
    halfT2 = x[base1] + (y2[0]/2 - y[base1 + 1]) / (y[base1] - y[base1 + 1]) * (x[base1 + 1] - x[base1])
    
    
    #plt.axline(np.max(y)/2, color = 'lightgray', linestyle = '--', label ="50%", linewidth = 1)
    plt.axvline(halfT1, color = 'Skyblue', linestyle = ':', label ="T(1/2)_roi1", linewidth = 1)
    plt.axvline(halfT2, color = 'yellowGreen', linestyle = ':', label ="T(1/2)_roi2", linewidth = 1)
    plt.legend()
    plt.title('Empty curve')

    plt.xlabel('Time (min)')
    plt.ylabel('cps')
    plt.show()
    
    return fit, fit2, MSE, RMSE, Rsquared, MSE2, RMSE2, Rsquared2, halfT1, halfT2
    
fit, fit2, MSE, RMSE, Rsquared, MSE2, RMSE2, Rsquared2, halfT1, halfT2 = polynominalfit(x, y, x2, y2)

def empty_cal(fit):
    empty = []
    for i in range(len(fit)):
        e = (fit[0] - fit[i]) / fit[0]
        empty.append(e * 100)
    
    return empty

empty = empty_cal(fit)
empty2 = empty_cal(fit2)

def savedataset(dcm_p, x, y, x2, y2, halflife, empty, empty2, background, background2, correcty, correcty2, fit, fit2, RMSE, Rsquared, RMSE2, Rsquared2, halfT1, halfT2):
    data_sheet1 = {'Patient Name' : dcm_p.PatientName,
                   'Patient ID' : dcm_p.PatientID,
                   'Study Date' : dcm_p.StudyDate,
                   'Study Time' : dcm_p.StudyTime,
                   'Study Name' : dcm_p.StudyDescription,
                   'Energy Window Name' : dcm_p.EnergyWindowInformationSequence[0].EnergyWindowName,
                   'Begin Time' : x[0], 'End Time' : x[-1], 'T 1/2_roi_1' : halfT1, 'T 1/2_roi_2' : halfT2}
    data_sheet2 = {'time (min)' : x,'cps_1' : y, 'cps_2' : y2,
                   '%empty_1' : empty, '%empty_2' : empty2,
                   '%retention_1' : 100-np.array(empty), '%retention_2' : 100-np.array(empty2),
                   'background_1' : background, 'background_2' : background2, 
                   'decay corrected_1' : correcty, 'decay corrected_2' : correcty2,
                   'polynomial fit_1' : fit, 'polynomial fit_2' : fit2,
                   
                   'RMSE_1': RMSE, 'R-squared_1' : Rsquared, 'RMSE_2': RMSE2, 'R-squared_2' : Rsquared2
                   } #리스트 자료형으로 생성 
    excel1 = pd.DataFrame(data_sheet1) #데이터 프레임으로 전환 및 생성
    excel2 = pd.DataFrame(data_sheet2)
    xlxs_dir='D:\\dicom_data\\SPECT\\' + str(dcm_p.StudyID) #경로 및 파일명 설정
    excel1.to_csv(xlxs_dir + 'Header.csv', index=False, mode='w', encoding='utf-8-sig') #raw_data1 시트에 저장
    excel2.to_csv(xlxs_dir + 'Analysis.csv', index=False, mode='w', encoding='utf-8-sig')
    
savedataset(dcm_p, x, y, x2, y2, halflife, empty, empty2, background, background2, correcty, correcty2, fit, fit2, RMSE, Rsquared, RMSE2, Rsquared2, halfT1, halfT2)


##############################################################################################################3