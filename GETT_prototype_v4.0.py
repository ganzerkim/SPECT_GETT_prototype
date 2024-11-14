# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:32:00 2024

@author: Siemens Healthineers_Mingeon Kim
"""
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

import os
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import * # __all__
from tkinter import filedialog
#from PIL import Image
import shutil

root = Tk()
root.title("SIEMENS MI Gastric emptying time tool")
root.geometry("500x850")#가로 세로
def add_file():
    files = filedialog.askdirectory(title="추가할 파일경로를 선택하세요", \
        initialdir=r".\Desktop")
        # 최초에 사용자가 지정한 경로를 보여줌

    # 사용자가 선택한 파일 목록
    list_file.insert(END, files)

# 선택 삭제
def del_file():
    #print(list_file.curselection())
    for index in reversed(list_file.curselection()):
        list_file.delete(index)


# 추가 경로 (폴더)
def browse_dest_loadpath():
    folder_selected = filedialog.askdirectory()
    if folder_selected == "": # 사용자가 취소를 누를 때
        # print("폴더 선택 취소")
        return
    #print(folder_selected)
    txt_dest_loadpath.delete(0, END)
    txt_dest_loadpath.insert(0, folder_selected)


# 저장 경로 (폴더)
def browse_dest_savepath():
    folder_selected = filedialog.askdirectory()
    if folder_selected == "": # 사용자가 취소를 누를 때
        # print("폴더 선택 취소")
        return
    #print(folder_selected)
    txt_dest_savepath.delete(0, END)
    txt_dest_savepath.insert(0, folder_selected)
    
    
def hash_acc(num, length, sideID):
   try:
       siteID = str.encode(sideID)
       num = str.encode(num)
                              # hash
       m = hmac.new(siteID, num, hashlib.sha256).digest()
                              #convert to dec
       m = str(int(binascii.hexlify(m),16))
                              #split till length
       m=m[:length]
       return m
   except Exception as e:
          print("Something went wrong hashing a value :(")
          return
      
def indexfinder(AT, dcm_tmp):
    for ii in range(len(dcm_tmp)):
        if AT < dcm_tmp[ii].AcquisitionTime:
            print(ii)
            return ii
                
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

def roi_cps(msk_img, roi_order):
    m_cpm = np.sum(msk_img)
    m_cps = m_cpm / 60
    
    return m_cps

def decaycorrection(x, y, halflife):
    x = np.array(x)
    y = np.array(y)
    halflife = np.around(halflife)
    time_passed_hours = x
    decay_constant = 0.693/halflife
    correcty = y * np.exp(-decay_constant * time_passed_hours)
        
    return correcty

def countcal(roi_number, dcm_tmp_2, msk_img, roi_order):
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

def empty_cal(fit):
    empty = []
    for i in range(len(fit)):
        e = (fit[0] - fit[i]) / fit[0]
        empty.append(e * 100)
    
    return empty

def polynominalfit(x, y, x2, y2, background, background2, halflife):
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
    plt.plot(x, background2, c = 'Orange', label = 'Background2', linestyle = '--')
    
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


def savedataset(dcm_p, x, y, x2, y2, halflife, empty, empty2, background, background2, correcty, correcty2, fit, fit2, RMSE, Rsquared, RMSE2, Rsquared2, halfT1, halfT2, save_path):
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
    xlxs_dir= txt_dest_savepath.get() + '/' + str(dcm_p.StudyID) #경로 및 파일명 설정 #경로 및 파일명 설정
    excel1.to_csv(xlxs_dir + 'Header.csv', index=False, mode='w', encoding='utf-8-sig') #raw_data1 시트에 저장
    excel2.to_csv(xlxs_dir + 'Analysis.csv', index=False, mode='w', encoding='utf-8-sig')

def gett():
    try:
        images_path = list_file.get(0)
        
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
        i = 0
        ii = 0
        iii = 0

        for i in range(len(path_tmp)):
            dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
            dcm_tmp.append(dcm_p)
            progress = (i+1 + ii + iii) / (len(path_tmp) * 3)  * 100
            p_var.set(progress)
            progress_bar.update()
            

        #AT = dcm_tmp[3].AcquisitionTime
        
                
        dcm_tmp_2 = []
        dcm_tmp_2.append(dcm_tmp[0])
        for ii in range(len(dcm_tmp)-1):
            if dcm_tmp[ii+1].AcquisitionTime == dcm_tmp_2[ii].AcquisitionTime:
                pass
            elif dcm_tmp[ii+1].AcquisitionTime > dcm_tmp_2[ii].AcquisitionTime:
                dcm_tmp_2.append(dcm_tmp[ii+1])
            else:
                AT = dcm_tmp[ii+1].AcquisitionTime
                dcm_tmp_2.insert(indexfinder(AT, dcm_tmp_2), dcm_tmp[ii+1])
            
            progress = (i+1 + ii+1 + iii) / (len(path_tmp) * 3)  * 100
            p_var.set(progress)
            progress_bar.update()
        
        # image load
        img_tmp = []

        for iii in range(len(dcm_tmp_2)):
            ccc = dcm_tmp_2[iii].pixel_array
            img_tmp.append(ccc)
            progress = (i+1 + ii+1 + iii+1) / (len(path_tmp) * 3)  * 100 # 실제 percent 정보를 계산
            p_var.set(progress)
            progress_bar.update()
            
        
        option_type = cmb_width.get()
        img_list, img_geo = detector_selection(dcm_tmp_2, option_type)
        final_img = img_sum(img_list)
        
        #ROI_Drawing part

        logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                                   '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                            level=logging.INFO)

        # Create image
        img = final_img
        vmin = int(txt_vmin.get())
        vmax = int(txt_vmax.get())

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
        
        ####################################################################################
        
        roi_order = get_roi_img(img, roi_tmp)
        
        msk_img = get_msk_img(img_geo, roi_order)

        
        ###################################################################################
        x, a, b, background_a, background_b = countcal(0, dcm_tmp_2, msk_img, roi_order)
        x2, a2, b2, background_a2, background_b2 = countcal(1, dcm_tmp_2, msk_img, roi_order)

        halflife = np.int16(txt_Halflife.get())
        
        if geo_var.get():
            y = np.sqrt(np.array(a) * np.array(b))
            background = np.sqrt(np.array(background_a) * np.array(background_b))
            y2 = np.sqrt(np.array(a2) * np.array(b2))
            background2 = np.sqrt(np.array(background_a2) * np.array(background_b2))
        else:
            y = (np.array(a) + np.array(b)) / 2
            background = (np.array(background_a) + np.array(background_b)) / 2
            y2 = (np.array(a2) + np.array(b2)) / 2
            background2 = (np.array(background_a2) + np.array(background_b2)) / 2
        
        if decay_var.get():
            correcty = decaycorrection(x, y, halflife) #if decy correction is on
            correcty2 = decaycorrection(x2, y2, halflife) #if decay correction is on
        else:
            correcty = np.array(y)
            correcty2 = np.array(y2)
            
        if background_var.get():
            correcty = correcty - background #if background correction is on
            correcty2 = correcty2 - background2 #if background correction is on
        else:
            correcty = correcty
            correcty2 = correcty2
            
            
        fit, fit2, MSE, RMSE, Rsquared, MSE2, RMSE2, Rsquared2, halfT1, halfT2 = polynominalfit(x, correcty, x2, correcty2, background, background2, halflife)
        empty = empty_cal(fit)
        empty2 = empty_cal(fit2)

        savedataset(dcm_p, x, y, x2, y2, halflife, empty, empty2, background, background2, correcty, correcty2, fit, fit2, RMSE, Rsquared, RMSE2, Rsquared2, halfT1, halfT2, txt_dest_savepath.get())
        
        msgbox.showinfo("알림", "교수님~! 분석이 완료되었습니다. 저장폴더의"+ str(dcm_p.StudyID) + ".xlsx 확인해주세요~")

    except Exception as err: # 예외처리
        msgbox.showerror("에러", str(err) + ", Siemens Research Collaboration Scientist에게 문의해주세요!")    
        

# 시작
def start():
   
    # 파일 목록 확인
    if list_file.size() == 0:
        msgbox.showwarning("경고", "폴더 경로를 추가해주세요")
        return

    # 저장 경로 확인
    if len(txt_dest_savepath.get()) == 0:
        msgbox.showwarning("경고", "저장 경로를 선택해주세요")
        return

    # 이미지 통합 작업
    gett()

photo = PhotoImage(file="./pics/SHSKRRCKMG.png")
label2 = Label(root, image=photo)
label2.pack()

# 파일 프레임 (파일 추가, 선택 삭제)
file_frame = Frame(root)
file_frame.pack(fill="x", padx=5, pady=5) # 간격 띄우기

btn_add_file = Button(file_frame, padx=5, pady=5, width=10, text="폴더추가", command=add_file)
btn_add_file.pack(side="left")

btn_del_file = Button(file_frame, padx=5, pady=5, width=10, text="선택삭제", command=del_file)
btn_del_file.pack(side="right")

# 리스트 프레임
list_frame = Frame(root)
list_frame.pack(fill="both", padx=5, pady=5)

scrollbar = Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")

list_file = Listbox(list_frame, selectmode="extended", height=5, yscrollcommand=scrollbar.set)
list_file.pack(side="left", fill="both", expand=True)
scrollbar.config(command=list_file.yview)

# 저장 경로 프레임
savepath_frame = LabelFrame(root, text="저장경로")
savepath_frame.pack(fill="x", padx=5, pady=5, ipady=5)

txt_dest_savepath = Entry(savepath_frame)
txt_dest_savepath.pack(side="left", fill="x", expand=True, padx=5, pady=5, ipady=4) # 높이 변경

btn_dest_savepath = Button(savepath_frame, text="찾아보기", width=10, command=browse_dest_savepath)
btn_dest_savepath.pack(side="right", padx=5, pady=5)

# 옵션 프레임
frame_option = LabelFrame(root, text="*이미지 처리시 필요한 정보들을 기입해주세요*")
frame_option.pack(padx=15, pady=15, ipady=1)
################################################################

# 실행할 옵션 선택
lbl_option = Label(frame_option, text="Summation Method", width=15)
lbl_option.pack(side="left", padx=5, pady=5)

# 실행 옵션 콤보
opt_width = ["ANT", "POST", "SUM_ANT", "SUM_POST"]
cmb_width = ttk.Combobox(frame_option, state="readonly", values=opt_width, width=10)
cmb_width.current(0)
cmb_width.pack(side="left", padx=5, pady=5)

# Halflife 입력 옵션
lbl_Halflife = Label(frame_option, text="Halflife", width = 10)
lbl_Halflife.pack(side="top", padx = 5, pady = 0, fill="both", expand=True)

txt_Halflife = Entry(frame_option, width=5)
txt_Halflife.pack(pady = 5)
txt_Halflife.insert(END, "360")

# vmax vmin 입력
lbl_vmin = Label(frame_option, text="vmin", width = 10)
lbl_vmin.pack(side="top", padx = 5, pady = 0, ipadx = 5, fill="both", expand=True)

txt_vmin = Entry(frame_option, width=5)
txt_vmin.pack(pady = 5)
txt_vmin.insert(END, "0")

lbl_vmax = Label(frame_option, text="vmax", width = 10)
lbl_vmax.pack(side="top", padx = 5, pady = 0, ipadx = 5, fill="both", expand=True)

txt_vmax = Entry(frame_option, width=5)
txt_vmax.pack(pady = 5)
txt_vmax.insert(END, "4000")

bool_frame = Frame(root)
list_frame.pack(fill="both", padx=5, pady=5)

scrollbar = Scrollbar(list_frame)
scrollbar.pack(side="right", fill="y")

geo_var = BooleanVar()
decay_var = BooleanVar()
background_var = BooleanVar()

geo_var.set(True)
decay_var.set(True)
background_var.set(True)

correction_frame = Frame(root)
correction_frame.pack(side="top", fill="both", expand=True, padx = 20)
geo_checkbutton = Checkbutton(correction_frame, text="Geometry Correction", variable=geo_var)
geo_checkbutton.pack(side=LEFT)
decay_checkbutton = Checkbutton(correction_frame, text="Decay Correction", variable=decay_var)
decay_checkbutton.pack(side=LEFT)
background_checkbutton = Checkbutton(correction_frame, text="Background Correction", variable=background_var)
background_checkbutton.pack(side=LEFT)

############################################################################
##################################################################
# 진행 상황 Progress Bar
frame_progress = LabelFrame(root, text="진행상황")
frame_progress.pack(fill="x", padx=5, pady=5, ipady=5)

p_var = DoubleVar()
progress_bar = ttk.Progressbar(frame_progress, maximum=100, variable=p_var)
progress_bar.pack(fill="x", padx=5, pady=5)

# 실행 프레임
frame_run = Frame(root)
frame_run.pack(fill="x", padx=5, pady=5)

btn_close = Button(frame_run, padx=5, pady=5, text="닫기", width=12, command=root.quit)
btn_close.pack(side="right", padx=5, pady=5)

btn_start = Button(frame_run, padx=5, pady=5, text="시작", width=12, command=start)
btn_start.pack(side="right", padx=5, pady=5)

root.resizable(True, True)
root.mainloop()
