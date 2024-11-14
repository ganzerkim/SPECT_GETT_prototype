# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:02:18 2024

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def find_boundaries(images):
    # 각 이미지의 경계를 찾아 ROI를 만듭니다.
    rois = []
    for img in images:
        threshold = 0.25 * np.max(img)
        roi = img > threshold
        labeled_array, num_features = ndimage.label(roi)
        for i in range(1, num_features + 1):
            region = np.argwhere(labeled_array == i)
            min_x, min_y = np.min(region, axis=0)
            max_x, max_y = np.max(region, axis=0)
            rois.append((min_x, min_y, max_x, max_y))
    return rois

def smooth_edges(image):
    # 이미지의 가장자리를 부드럽게 만듭니다.
    blurred = ndimage.gaussian_filter(image, sigma=2)
    threshold = 0.5 * np.max(blurred)
    binary = blurred > threshold
    labeled_array, num_features = ndimage.label(binary)
    largest_region = np.zeros_like(binary)
    max_size = 0
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled_array == i)
        if region_size > max_size:
            max_size = region_size
            largest_region = labeled_array == i
    return largest_region

def calculate_longitudinal_axis(image):
    # 가장자리를 부드럽게 하고 중점을 찾아 두 점을 연결하는 곡선을 계산합니다.
    smooth_contour = smooth_edges(image)
    y, x = np.nonzero(smooth_contour)
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    return p

def separate_regions(image, longitudinal_axis):
    # 두 반으로 이미지를 분리합니다.
    midpoint = image.shape[1] // 2
    longitudinal_axis_values = longitudinal_axis(np.arange(image.shape[1]))
    separation_line = np.argmax(longitudinal_axis_values)
    top_half = image[:, :separation_line]
    bottom_half = image[:, separation_line:]
    return top_half, bottom_half

def calculate_gastric_counts(top_half, bottom_half):
    # 각 영역의 게이트릭 카운트를 계산합니다.
    total_counts = np.sum(top_half) + np.sum(bottom_half)
    proximal_counts = np.sum(top_half)
    distal_counts = np.sum(bottom_half)
    return proximal_counts, distal_counts, total_counts

def process_images(images):
    # 단계 1: 이미지 로드 및 정렬
    # 이미지를 로드하고 정렬하는 코드는 여기에 작성됩니다.

    # 단계 2: ROI 경계 찾기 및 수정
    rois = find_boundaries(images)

    # 단계 3: 가장자리 부드럽게 하고 긴 축 찾기
    longitudinal_axes = [calculate_longitudinal_axis(img) for img in images]

    # 단계 4: 영역 분리 및 게이트릭 카운트 계산
    results = []
    for img, axis in zip(images, longitudinal_axes):
        top_half, bottom_half = separate_regions(img, axis)
        proximal_counts, distal_counts, total_counts = calculate_gastric_counts(top_half, bottom_half)
        results.append((proximal_counts, distal_counts, total_counts))

    return results

# 이미지 데이터를 로드하고 process_images 함수를 호출합니다.
# images = [plt.imread('image{}.jpg'.format(i)) for i in range(1, num_images + 1)]
# results = process_images(images)


#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def find_boundaries(images):
    # 각 이미지의 경계를 찾아 ROI를 만듭니다.
    rois = []
    for img in images:
        threshold = 0.25 * np.max(img)
        roi = img > threshold
        labeled_array, num_features = ndimage.label(roi)
        for i in range(1, num_features + 1):
            region = np.argwhere(labeled_array == i)
            if region.size > 0:  # ROI가 발견되었는지 확인
                min_x, min_y = np.min(region[:, 1]), np.min(region[:, 0])
                max_x, max_y = np.max(region[:, 1]), np.max(region[:, 0])
                # ROI를 직사각형으로 만듭니다.
                roi = np.zeros_like(img, dtype=bool)
                roi[min_y:max_y, min_x:max_x] = True
                rois.append(roi)
    return rois

def smooth_edges(image):
    # 이미지의 가장자리를 부드럽게 만듭니다.
    blurred = ndimage.gaussian_filter(image, sigma=2)
    threshold = 0.5 * np.max(blurred)
    binary = blurred > threshold
    labeled_array, num_features = ndimage.label(binary)
    largest_region = np.zeros_like(binary)
    max_size = 0
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled_array == i)
        if region_size > max_size:
            max_size = region_size
            largest_region = labeled_array == i
    return largest_region

def calculate_longitudinal_axis(image):
    # 가장자리를 부드럽게 하고 중점을 찾아 두 점을 연결하는 곡선을 계산합니다.
    smooth_contour = smooth_edges(image)
    points = np.transpose(np.nonzero(smooth_contour))
    x, y = points[:, 1], points[:, 0]
    return x, y

def separate_regions(image, longitudinal_axis):
    # 두 반으로 이미지를 분리합니다.
    midpoint = image.shape[1] // 2
    longitudinal_axis_values = longitudinal_axis[0]  # x 좌표만 사용
    separation_line = np.argmax(longitudinal_axis_values)
    top_half = image[:, :separation_line]
    bottom_half = image[:, separation_line:]
    return top_half, bottom_half

def calculate_gastric_counts(top_half, bottom_half):
    # 각 영역의 게이트릭 카운트를 계산합니다.
    total_counts = np.sum(top_half) + np.sum(bottom_half)
    proximal_counts = np.sum(top_half)
    distal_counts = np.sum(bottom_half)
    return proximal_counts, distal_counts, total_counts

def process_image(image):
    # 이미지 처리 및 결과 반환
    roi = find_boundaries([image])[0]
    longitudinal_axis = calculate_longitudinal_axis(image)
    top_half, bottom_half = separate_regions(image, longitudinal_axis)
    proximal_counts, distal_counts, total_counts = calculate_gastric_counts(top_half, bottom_half)
    return roi, longitudinal_axis, proximal_counts, distal_counts, total_counts

# 이미지 로드
image = plt.imread('C:\\Users\\User\\Desktop\\Paik\\test.png')

# 이미지 처리
roi, longitudinal_axis, proximal_counts, distal_counts, total_counts = process_image(image)

# 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(roi, cmap='gray')
axes[0, 1].set_title('ROI')
axes[0, 1].axis('off')

x_values = np.arange(image.shape[1])
y_values = longitudinal_axis(x_values)
axes[1, 0].plot(x_values, y_values, color='red')
axes[1, 0].set_title('Longitudinal Axis')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')

axes[1, 1].bar(['Proximal', 'Distal'], [proximal_counts, distal_counts], color=['blue', 'green'])
axes[1, 1].set_title('Gastric Counts')
axes[1, 1].set_ylabel('Counts')

plt.tight_layout()
plt.show()