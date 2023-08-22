import cv2
import sys
import os

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

path = '/groups/CS156b/2023/yasers_beavers/data/preprocessed'
dir_list = os.listdir(path)
sobelpath = '/groups/CS156b/2023/yasers_beavers/data/sobel/'

for i in range(len(dir_list)):
    im_file = dir_list[i]
    img = cv2.imread(im_file)
    cv2.waitKey(0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1) # Combined X and Y Sobel Edge Detection
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_gray, threshold1=50, threshold2=100) # Canny Edge Detection
    cv2.imwrite(sobelpath + f'canny_edge/edge{i}.png', edges)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    #overlay edges on X-ray image
    background = cv2.imread(im_file)
    overlay = cv2.imread(sobelpath + f'canny_edge/edge{i}.png')

    added_image = cv2.addWeighted(background,1.0,overlay,0.4,0)

    cv2.imwrite(sobelpath + f'overlay/combined{i}.png', added_image)