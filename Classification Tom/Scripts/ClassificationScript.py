#%%
import cv2 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join


# %%
def remove_line(thresh_img): 
    '''
    Function which removes the answer line and, if necessary, reconnects the numbers. 
    '''   

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
    org_without_line = thresh_img.copy()

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
                    

    # plt.imshow(thresh_img, cmap = 'gray')
    
    return thresh_img, org_without_line

# %%
def CreateImageBatch(thresh_img): 
    '''
    Function which creates a list of square images, with the size of the training data.     
    '''
    contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    nh, nw = thresh_img.shape[:2]
    original = thresh_img.copy()
    image_batch = []
    idx_list = []

    for idx, cnt in enumerate(contours):
        x,y,w,h = bbox = cv2.boundingRect(cnt)
        if w > 0.05 * nw or h > 0.05*nh: # Should be done cleaner. (Gaussian) blurring? 
            
            # Save rectangles as images
            ROI = original[y-1:y+h + 1, x-1:x+w+1] 
            
            # Create border next to the most narrow side to create a square. 
            max_border_1 = (max(ROI.shape)-min(ROI.shape)) //2
            max_border_2 = (max(ROI.shape)-min(ROI.shape)) - max_border_1
            if h > w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = 0,
                                                    bottom = 0,
                                                    left = max_border_1,
                                                    right = max_border_2,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            if h < w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = max_border_1,
                                                    bottom = max_border_2,
                                                    left = 0,
                                                    right = 0,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            
            # Width and height of the trained on data. 
            width = 40
            height = 40
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(bordered_image, dim, interpolation = cv2.INTER_AREA)
            #cv2.imwrite(f"Resized_{idx}.png", resized)
            image_batch.append(resized)
            idx_list.append(idx)

    return image_batch



#%%
def LoadModel(): 
    '''
    Goal: To load a previously trained classification model. 
    Return type: keras sequential engine (?)
    Improvements: 
                    - Check return type
                    - Check if it is more efficient to load model and classify the batch in 1. 
                    - Test model --> Check if it needs arguments. 
    '''
    # load json and create model
    # "C://Users//tklop//ThesisADS//Model//model_data_augmentation.h5"
    json_file = open('C://Users//tklop//ThesisADS//Models//model_data_augmentation.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C://Users//tklop//ThesisADS//Models//model_data_augmentation.h5")
    print("Succesfully loaded the model")
        
    # evaluate loaded model on test data (remove?)
    return loaded_model 


 
#%%
# RUN THE MODEL. 
#----------------
mypath = "C://Users//tklop//ThesisADS//EquationImages//answers"
img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
loaded_model = LoadModel()

df = pd.DataFrame(columns= ['file_name', 'class_dict', 'answer_student', 'correct_answer', 'points'])

for name_index, name in enumerate(img_names): 
    img = cv2.imread(f"C://Users//tklop//ThesisADS//EquationImages//answers//{name}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_image_plot = img_gray.copy() # ORIGINAL IMAGE. 
    ret, thresh_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    thresh_img, org_without_line = remove_line(thresh_img)
    org_without_line_connected = thresh_img.copy()
    batch = CreateImageBatch(thresh_img)
    classified_list = []
    prob_list = []
    idx_list = []
    img_number = 1
    print(f"File: {name}")   
    for idx, value in enumerate(batch):  
        if idx == 0: 
            continue
        batch_img = batch[idx]
        batch_img = batch_img.reshape(1,40,40,1)
        y_prob = loaded_model.predict(batch_img) 
            
        print(f"Object number {img_number} is classfied as: {y_prob.argmax(axis=-1)[0]}, with a prob of: {np.max(y_prob)}")
        prob_list.append(np.max(y_prob))
        classified_list.append(y_prob.argmax(axis=-1)[0])
        idx_list.append(str(idx))
        
        img_number += 1

    classified_dict = dict(zip(range(1,len(classified_list) + 1), classified_list ))
    
    classified_list_str = [str(i) for i in classified_list]
    answer = ''.join(classified_list_str)
    if answer == name[4:-4]: 
        points = 1
    else: 
        points = 0
    df = df.append({'file_name': name,
                    'class_dict': classified_dict,
                    'answer_student': answer,
                    'correct_answer': name[4:-4], 
                    'points': points},
                     ignore_index=True)
    
    print("\n")
# %% 
# Plot the figure
fig = plt.figure(figsize=(40, 40))
columns = len(batch)

ax1 = fig.add_subplot(2, columns, 1)
ax1.title.set_text('Original Image')
plt.imshow(original_image_plot, cmap = 'gray')

ax2 = fig.add_subplot(2, columns, 2)
ax2.title.set_text('Binary Image Without Line')
plt.imshow(org_without_line, cmap = 'gray')

ax3 = fig.add_subplot(2, columns,3)
ax3.title.set_text('Binary Image Connected')
plt.imshow(org_without_line_connected , cmap = 'gray')

for i in range(1, columns):
    img = batch[i]
    ax = fig.add_subplot(1, columns, i)
    ax.title.set_text(f"{classified_list[i-1]} (prob: {prob_list[i-1]})") # Probability is still incorrect.
    plt.imshow(img, cmap= 'gray')

plt.show()


#%%
print(df.head())
# %%

# %%
