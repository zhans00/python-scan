import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from four_point_transform import four_point_transform, resize
from PIL import Image
from fpdf import FPDF


img = cv2.imread('sample.jpg',0)

thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

edges = np.array([
		[340, 1198],
		[2625, 1184],
		[2956, 4331],
		[81, 4403]], dtype = "float32")

newimg = four_point_transform(thresh, edges)


savedImage = "scanned.jpg"
cv2.imwrite(savedImage, newimg) 

im1 = Image.open(savedImage)

pdfname="scanned.pdf"
im1 = resize(im1, 2480, 3508)
im1.save(pdfname, "PDF", resolution=300.0, save_all=True)







plt.subplot(121),plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(newimg,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()	


