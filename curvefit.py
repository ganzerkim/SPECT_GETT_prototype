# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:51:58 2024

@author: User
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def func(x,a,b):
	return a * x + b 
# func은 피팅을 진행할 일종의 모델이라고 생각할 수 있습니다. 

x = np.linspace(0,100) # x축을 만들어줍니다
y = func(x, 1, 2) # 함수를 이용해 원본의 데이터를 만들어줍니다. 기울기는 1 y절편은 2인 함수입니다. 

yn = y + 0.9 * np.random.normal(size = len(x)) #원본 데이터에 노이즈를 섞어 줍니다.

popt, pcov = curve_fit(func, x, yn) 
# 선형모델(func은 1차 함수여서 선형모델이라고 했습니다)을 이용해 피팅을 진행합니다.
#popt에는 피팅의 결과로 계산된 a, b 값이 저장됩니다
#pcov에는 얼마나 잘 피팅되었는지 오류에 관한 부분이 저장됩니다

print(popt)
# a, b 모두 설정한 값과 비슷한 값이 나옵니다

plt.plot(x, y, label='original')
plt.plot(x, func(x, *popt), label = 'fitting')
plt.legend()