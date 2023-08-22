import cv2, sys
import pandas as pd
import matplotlib.pyplot as plt

DATA = '/groups/CS156b/data/student_labels/train2023.csv'
CONDITIONS =    ['No Finding',
                 'Enlarged Cardiomediastinum',
                 'Cardiomegaly',
                 'Lung Opacity',
                 'Pneumonia',
                 'Pleural Effusion',
                 'Fracture',
                 'Support Devices',
                 'Support Devices']

# CONDITIONS = ['Fracture']

def sobel(imgpath, name): 
    sys.path.append('/usr/local/lib/python3/site-packages')
    img = cv2.imread('/groups/CS156b/data/' + imgpath) 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21,21), 0) 

    # Sobel
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=11, threshold2=11) # Canny Edge Detection

    # Overlay
    added_image = cv2.addWeighted(img_gray,1.0,edges,1.0,0)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(10,5)
    axs[0].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Grayscale')
    axs[1].imshow(edges, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('Canny edge')
    axs[2].imshow(added_image, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title('Overlayed')
    # fig.suptitle(name)
    plt.set_title(name)
    fig.tight_layout()
    plt.savefig(f'/central/groups/CS156b/2023/yasers_beavers/scratch/data_stats/figures/{name}_sobel.jpg', dpi=500)


df = pd.read_csv(DATA).drop('Unnamed: 0.1', axis=1).drop('Unnamed: 0', axis=1)



for idx, row in df.iterrows():
    if CONDITIONS == []:
        break
    for c in CONDITIONS:
        if row[c] == 1:
            sobel(row['Path'], c)
            CONDITIONS.remove(c)
            print(1 - len(CONDITIONS) / 9)
            break

print('Done')