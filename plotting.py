# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:00:55 2024

@author: User
"""
import logging

import numpy as np
from matplotlib import pyplot as plt

from roipoly import RoiPoly

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Create image
img = np.ones((100, 100)) * range(0, 100)

# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("left click: line segment         right click or double click: close region")
plt.show(block=False)

# Let user draw first ROI
roi1 = RoiPoly(color='r', fig=fig)

# Show the image with the first ROI
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
roi1.display_roi()
plt.title('draw second ROI')
plt.show(block=False)

# Let user draw second ROI
roi2 = RoiPoly(color='b', fig=fig)

# Show the image with both ROIs and their mean values
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
for roi in [roi1, roi2]:
    roi.display_roi()
    roi.display_mean(img)
plt.title('The two ROIs')
plt.show()

# Show ROI masks
plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
           interpolation='nearest', cmap="Greys")
plt.title('ROI masks of the two ROIs')
plt.show()
######################################################################################
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
img = np.ones((100, 100)) * range(0, 100)

# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.title("Click on the button to add a new ROI")

# Draw multiple ROIs
multiroi_named = MultiRoi(roi_names=['My first ROI', 'My second ROI'])

# Draw all ROIs
plt.imshow(img, interpolation='nearest', cmap="Greys")
roi_names = []
roi_tmp = []
for name, roi in multiroi_named.rois.items():
    roi.display_roi()
    roi.display_mean(img)
    roi_tmp.append(roi)
    roi_names.append(name)
plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
plt.show()

###################################################################################

import pylab as pl
from roipoly import roipoly

# create image
img = pl.ones((100, 100)) * range(0, 100)

# show the image
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
pl.title("left click: line segment         right click: close region")

# let user draw first ROI
ROI1 = roipoly(roicolor='r')  # let user draw first ROI

# show the image with the first ROI
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
ROI1.displayROI()
pl.title('draw second ROI')

# let user draw second ROI
ROI2 = roipoly(roicolor='b')  # let user draw ROI

# show the image with both ROIs and their mean values
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
[x.displayROI() for x in [ROI1, ROI2]]
[x.displayMean(img) for x in [ROI1, ROI2]]
pl.title('The two ROIs')
pl.show()

# show ROI masks
pl.imshow(ROI1.getMask(img) + ROI2.getMask(img),
          interpolation='nearest', cmap="Greys")
pl.title('ROI masks of the two ROIs')
pl.show()