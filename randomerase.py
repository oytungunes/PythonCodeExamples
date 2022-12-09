import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

import numpy as np

from PIL import Image
from numpy import asarray

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.2, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


# Import Module
import os

# Directory
directory ="random_erase_augmented"
# Parent Directory path
parent_dir = "C:/Users/oytun.gunes/Desktop/"
# Path
path = os.path.join(parent_dir, directory)
# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
os.mkdir(path)

# Folder Path
#path = r"C:\Users\oytun.gunes\Desktop\task_atok-2022_11_22_12_57_25-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_depo-2022_11_22_14_00_15-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_kucuk_oda_1-2022_11_22_15_01_25-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_kucuk_oda_2-2022_11_22_10_49_42-yolo 1.1\obj_train_data"
#path  = r"C:\Users\oytun.gunes\Desktop\task_ofis-2022_11_22_12_27_29-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_otopark_1-2022_11_22_15_42_23-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_otopark_2-2022_11_22_15_15_53-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_toplanti_4kat_1-2022_11_22_15_47_30-yolo 1.1\obj_train_data"
#path =r"C:\Users\oytun.gunes\Desktop\task_toplanti_kat4_2-2022_11_22_15_09_27-yolo 1.1\obj_train_data"
#path = r"C:\Users\oytun.gunes\Desktop\task_toplanti_kat5_1-2022_11_22_14_39_50-yolo 1.1\obj_train_data"
path = r"C:\Users\oytun.gunes\Desktop\task_toplanti_kat5_2-2022_11_22_15_21_31-yolo 1.1\obj_train_data"
# Change the directory
os.chdir(path)




# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".jpg"):

        file_path = f"{path}\{file}"
        image = Image.open(file_path)
        data = np.array(image)
        eraser = get_random_eraser()
        dataerased = eraser(data) # construct erased image
        imageerased = Image.fromarray(dataerased) #convert nparray to image
        basename = os.path.basename(file_path) #find filename
        #foldername= 'augmented'
        imageerased.save('C:/Users/oytun.gunes/Desktop/'+directory+'/'+basename+'.jpg', 'JPEG')
        #plt2.imshow(eraser(data))
        #plt2.show()
        # call read text file function
        # read_text_file(file_path)

# load the image
#image = Image.open('0_20221116_000329_HoloLens_2022_11_16_16_43_38.jpg')

# convert image to numpy array
#data = np.array(image)
#x = np.zeros(( 64, 64, 3), dtype=np.uint8)

#eraser = get_random_eraser()
#plt2.imshow(eraser(data))
#plt2.show()
