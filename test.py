import cv2 as cv
import numpy as np

# Read the noisy image
img = cv.imread('assets/bg2.jpeg', cv.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if img is None:
    print("Error: Could not read the image.")
    exit()

# Resize the image to a smaller size
scale_percent = 50  # Adjust this percentage to change the size
width = int(img.shape[1] * scale_percent / 140)
height = int(img.shape[0] * scale_percent / 140)
dim = (width, height)

# Apply resizing
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

# Apply different denoising filters
mean = cv.blur(img, (5, 5))
median = cv.medianBlur(img, 5)
gaussian = cv.GaussianBlur(img, (5, 5), 0)

# Stack images side by side
res = np.hstack((img, mean, median, gaussian))

# Display the images
cv.imshow("Denoising Comparison", res)
cv.moveWindow("Denoising Comparison", 1000, 0)  # Move window position

# Wait for a key press and close the window
cv.waitKey(0)
cv.destroyAllWindows()
