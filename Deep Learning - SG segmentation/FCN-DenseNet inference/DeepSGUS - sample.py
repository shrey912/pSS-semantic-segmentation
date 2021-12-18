# Script for the automatic semantic segmentation of SGUS images

from DeepSGUS import DeepSGUS_CNN
import matplotlib.pyplot as plt
import cv2 as cv # Version 3.7
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__) # 1.14.0
print(cv.__version__) # 4.1.0

#LOAD PRETRAINED MODEL
DeepSGUS = DeepSGUS_CNN('2 Deep Learning - SG segmentation/FCN-DenseNet inference/frozen_graph.pb')
DeepSGUS.print_layerNames()

#INPUTS 
# inputImg    = 'IMG-0001-00008.jpg'
inputImg    = 'TIONI_0001_img.jpg'
# inputImg    = 'STELLIN_0001_img.jpg'

#RUN SEGMENTATION
rez = DeepSGUS.segmentImage('2 Deep Learning - SG segmentation/FCN-DenseNet inference/in/' + inputImg)
output_PerPixelPredictions = rez[0] # 0-background, 1-salivary gland (image)
output_BlackAndWhiteSG     = rez[1] # black-background, white-salivary gland (imge)
output_ContourOverInput    = rez[2] # resulting contour is drawn over the input image (image)
output_contourSG_points    = rez[3] # contour points (array)

#SAVE
cv.imwrite('2 Deep Learning - SG segmentation/FCN-DenseNet inference/out/' + inputImg + '_SG_predictions.jpg'   , output_PerPixelPredictions) 
cv.imwrite('2 Deep Learning - SG segmentation/FCN-DenseNet inference/out/' + inputImg + '_SG_Black&White.jpg'   , output_BlackAndWhiteSG) 
cv.imwrite('2 Deep Learning - SG segmentation/FCN-DenseNet inference/out/' + inputImg + '_SG_Contour.jpg'       , output_ContourOverInput) 
np.savetxt('2 Deep Learning - SG segmentation/FCN-DenseNet inference/out/' + inputImg + '_SG_Contour_Points.txt', output_contourSG_points) 

#SHOW
img = cv.imread('2 Deep Learning - SG segmentation/FCN-DenseNet inference/in/' + inputImg)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.plot(output_contourSG_points[:,0], output_contourSG_points[:,1])
plt.show()