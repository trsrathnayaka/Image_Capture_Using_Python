import cv2 as cv
import numpy as np

# Read the image in grayscale
img = cv.imread('assets/bg2.jpeg', 0)

# Resize the image to reduce its size
scale_percent = 50  # Adjust this percentage to change the size
width = int(img.shape[1] * scale_percent / 140)
height = int(img.shape[0] * scale_percent / 140)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Apply threshold
ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

# 3x3 square structural element
kernel = np.ones((3, 3), np.uint8)

# Inner boundary extraction
eroded = cv.erode(img, kernel, iterations=1)
inner_edge = img - eroded

# Outer boundary extraction
dilated = cv.dilate(img, kernel, iterations=1)
outer_edge = dilated - img

# Stack images for comparison
res = np.hstack((img, eroded, inner_edge, dilated, outer_edge))

# Display the images
cv.imshow("Boundary", res)
cv.moveWindow("Boundary", 1000, 0)  # Move window position

# Wait for a key press and close the window
cv.waitKey(0)
cv.destroyAllWindows()
