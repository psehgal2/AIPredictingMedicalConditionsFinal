# import numpy as np
# import cv2
# import argparse
# import matplotlib.pyplot as plt
# from Computer_Vision.Sobel_Edge_Detection.convolution import convolution
# from Computer_Vision.Sobel_Edge_Detection.gaussian_smoothing import gaussian_blur
 
 
# def sobel_edge_detection(image, filter, verbose=False):
#     new_image_x = convolution(image, filter, verbose)
 
#     if verbose:
#         plt.imshow(new_image_x, cmap='gray')
#         plt.title("Horizontal Edge")
#         plt.show()
 
#     new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
 
#     if verbose:
#         plt.imshow(new_image_y, cmap='gray')
#         plt.title("Vertical Edge")
#         plt.show()
 
#     gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
 
#     gradient_magnitude *= 255.0 / gradient_magnitude.max()
 
#     if verbose:
#         plt.imshow(gradient_magnitude, cmap='gray')
#         plt.title("Gradient Magnitude")
#         plt.show()
 
#     return gradient_magnitude
 
 
# if __name__ == '__main__':
#     filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
 
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", required=True, help="Path to the image")
#     args = vars(ap.parse_args())
 
#     image = cv2.imread(args["image"])
#     image = gaussian_blur(image, 9, verbose=True)
#     sobel_edge_detection(image, filter, verbose=True)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # loading image
# #img0 = cv2.imread('SanFrancisco.jpg',)
# img0 = cv2.imread('windows.jpg',)

# # converting to gray scale
# gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# # remove noise
# img = cv2.GaussianBlur(gray,(3,3),0)

# # convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plt.show()


# import cv2
# import numpy as np
  
# #Capture livestream video content from camera 0
# cap = cv2.VideoCapture(0)
  
# while(1):
  
#     # Take each frame
#     _, frame = cap.read()
      
#     # Convert to HSV for simpler calculations
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      
#     # Calculation of Sobelx
#     sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
      
#     # Calculation of Sobely
#     sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
      
#     # Calculation of Laplacian
#     laplacian = cv2.Laplacian(frame,cv2.CV_64F)
      
#     cv2.imshow('sobelx',sobelx)
#     cv2.imshow('sobely',sobely)
#     cv2.imshow('laplacian',laplacian)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
  
# cv2.destroyAllWindows()
  
# #release the frame
# cap.release()


import cv2

import sys

def sobel(imgpath):
    sys.path.append('/usr/local/lib/python3/site-packages')
 
    # Read the original image
    img = cv2.imread(imgpath) 
    # Display original image
    cv2.imwrite(f'{imgpath}.jpg', img)
    cv2.waitKey(0)
    
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imwrite(f'Sobel X{imgpath}.jpg', sobelx)
    cv2.waitKey(0)
    cv2.imwrite(f'Sobel Y{imgpath}.jpg', sobely)
    cv2.waitKey(0)
    cv2.imwrite(f'Sobel X Y using Sobel() function{imgpath}.jpg', sobelxy)
    cv2.waitKey(0)
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imwrite(f'Canny Edge Detection{imgpath}.jpg', edges)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img

sys.path.append('/usr/local/lib/python3/site-packages')
 
# Read the original image
img = cv2.imread('/groups/CS156b/2023/yasers_beavers/pneumonia.jpg') 
# Display original image
cv2.imwrite('Original.jpg', img)
cv2.waitKey(0)
 
# Convert to graycsale
# inc 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# frame = increase_brightness(img_gray, value=20)

 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imwrite('Sobel X.jpg', sobelx)
cv2.waitKey(0)
cv2.imwrite('Sobel Y.jpg', sobely)
cv2.waitKey(0)
cv2.imwrite('Sobel X Y using Sobel() function.jpg', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_gray, threshold1=50, threshold2=100) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imwrite('Canny Edge Detection5.png', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()


background = cv2.imread('/groups/CS156b/2023/yasers_beavers/test.jpg')
overlay = cv2.imread('Canny Edge Detection5.png')

added_image = cv2.addWeighted(background,1.0,overlay,0.3,0)

cv2.imwrite('pneumoniacombined.png', added_image)