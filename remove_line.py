import cv2 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join


# %%
def remove_line(img): 
    '''
    Function which removes the answer line and, if necessary, reconnects the numbers. 
    '''   
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    line_idx_list = []
    for idx, x_array in enumerate(thresh_img): 
        black_pixel_counter = 0 
        for pixel in x_array: 
            if pixel == 0 : 
                black_pixel_counter += 1
        if black_pixel_counter > 0.5 * thresh_img.shape[0]: 
            line_idx_list.append(idx)

    filling_array = np.full(thresh_img.shape[1], fill_value = 255)
    for idx, line_idx in enumerate(line_idx_list): 
        img_gray[line_idx] = filling_array
    ret, thresh_img = cv2.threshold(img_gray, 175, 255, cv2.THRESH_BINARY)

    top_line = min(line_idx_list)
    bottom_line = max(line_idx_list)

    for idx in range(thresh_img.shape[1]): 
        for i in range(top_line, bottom_line + 1):
            # Same Line. 
            if thresh_img[(top_line -1)][idx] == 0 and thresh_img[(bottom_line +1)][idx] == 0:
                thresh_img[i][idx] = 0 
            # +1 clockwise. 
            try: 
                if thresh_img[(top_line -1)][idx + 1] == 0 and thresh_img[(bottom_line +1)][idx - 1] == 0:
                    thresh_img[i][idx] = 0 
            except:
                pass
            # +2 clockwise. 
            try: 
                if thresh_img[(top_line -1)][idx + 2] == 0 and thresh_img[(bottom_line +1)][idx - 2] == 0:
                    thresh_img[i][idx] = 0 
            except:
                pass
            # -1 Clockwise. 
            try: 
                if thresh_img[(top_line -1)][idx - 1] == 0 and thresh_img[(bottom_line +1)][idx + 1] == 0:
                    thresh_img[i][idx] = 0 
            except:
                pass
            # -2 Clockwise. 
            try: 
                if thresh_img[(top_line -1)][idx - 2] == 0 and thresh_img[(bottom_line +1)][idx + 2] == 0:
                    thresh_img[i][idx] = 0 
            except:
                pass
                    

    plt.imshow(thresh_img, cmap = 'gray')
    
    return thresh_img

img = cv2.imread("C://Users//tklop//ThesisADS//EquationImages//answers//tms_14.jpg")
remove_line(img)

#%%