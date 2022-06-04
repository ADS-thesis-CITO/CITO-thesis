'''
:Author: Tom Klopper. 
:Goal: To classify all open answers of a given test and grade it. 
:Last update: 04-06-2022.
'''
#%%
# Import libraries. 
import cv2 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join

def RemoveLine(img_gray, thresh_img): 
    '''
    :Goal:              Function which removes the "answer line" and, if necessary, reconnects the numbers. 
    :Param img_gray:    Grayscaled image as an array.
    :Param thresh_img:  Threshold image as an array.
    :Returns:           Returns threshhold image as an array. 
    '''   
    # Find the answer line by iterating through pixels and find lines of black (value = 0) pixels
    line_idx_list = []
    for idx, x_array in enumerate(thresh_img): 
        black_pixel_counter = 0 
        for pixel in x_array: 
            if pixel == 0: 
                black_pixel_counter += 1
        if black_pixel_counter > 0.9 * thresh_img.shape[0]: # Threshold of 90% should be black to be the answer line. 
            line_idx_list.append(idx)

    # Replace the answer line with white (value = 255) arrays. 
    filling_array = np.full(thresh_img.shape[1], fill_value = 255)
    for idx, line_idx in enumerate(line_idx_list): 
        img_gray[line_idx] = filling_array
    ret, thresh_img = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    
    # Copy image for process visualization. 
    org_without_line = thresh_img.copy()
    
    # Find location of the lines. 
    # Try except for cases in which the line may not be found at all. 
    try: 
        top_line = min(line_idx_list)
        bottom_line = max(line_idx_list)
    except ValueError: 
        top_line = 0 
        bottom_line = 0

    # Connect numbers with black lines from which the pixels above and under the removed line are black. 
    for idx in range(thresh_img.shape[1]): 
        for i in range(top_line, bottom_line + 1):
            # Same Line. 
            try: 
                if thresh_img[(top_line -1)][idx] == 0 and thresh_img[(bottom_line +1)][idx] == 0:
                    thresh_img[i][idx] = 0 
            except:
                pass
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
                    
    return thresh_img, org_without_line

def GetAllRectangles(img): 
    '''
    :Goal:          Function which returns a nested dictionairy including all boxes with max and min values for x and y. 
    :param img:     Threshold image as an array. 
    :Returns:       Dictionairy with the keys 'xmin', 'xmax', 'ymin', 'ymax', 'surface'
    '''
    # Define variables. 
    nested_dict_keys = ['xmin', 'xmax', 'ymin', 'ymax', 'surface']
    all_rectangles = {}
    image_number = 1
    nh, nw = img.shape[:2]
    surface_list =[]
    size_sorted_rectangles = {}

    # Iterate through countours and add them to the dictionairy. 
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    for idx, cnt in enumerate(contours):
        x,y,w,h = bbox = cv2.boundingRect(cnt)
        if h < 0.1 * nh or w*h > 0.8 * img.shape[0]* img.shape[1]: # Skip unusually small and large boxes.  
            continue
        all_rectangles[f"Crop_{str(image_number)}"] = dict(zip(nested_dict_keys, [x, x + w, y, y + h, w*h]))
        image_number += 1

    # Create list of surfaces sorted in descending order.   
    sorted_surfaces = [surface_list.append(all_rectangles[crops]['surface']) for crops in all_rectangles] 
    sorted_surfaces = sorted(surface_list, reverse = True) 

    # Sort the all_rectangles dictionary based on the surface of the crop out.
    for sort_idx, size_values in enumerate(sorted_surfaces): 
        for idx, crop_name in enumerate(all_rectangles.keys()):
            if all_rectangles[crop_name]['surface'] == sorted_surfaces[sort_idx]: 
                size_sorted_rectangles[crop_name] = all_rectangles[crop_name]
    return size_sorted_rectangles

def OverlappingRectangles(all_rectangles):
    '''
    :Goal:                  Function which remove boxes from which the centroid is within a larger box. 
    :Param all_rectangles:  Dictionairy containing information about all boxes.
    :Returns:               Dictionairy sorted on surface size.   
    '''
    # Define variables. 
    centroid_dict = {}
    overlap = []
    rect_dict = all_rectangles.copy()
    dict_keys = list(rect_dict.keys())

    # Create dictionairy with the x and y coördinates which form the centroid of the image. 
    for i in rect_dict: 
        ctr_x = rect_dict[i]['xmin'] + (rect_dict[i]['xmax'] - rect_dict[i]['xmin'])/2
        ctr_y = rect_dict[i]['ymin'] + (rect_dict[i]['ymax'] - rect_dict[i]['ymin'])/2
        centroid_dict[i] = [ctr_x, ctr_y]
    
    # Iterate through all combinations of images and check if the centroid is in an another box.       
    for rectangle in dict_keys: 
        for centroid in dict_keys: 
            if rectangle == centroid: # Skip if the image is the same. 
                continue
                        
            if (int(centroid_dict[rectangle][0]) in range(rect_dict[centroid]['xmin'],rect_dict[centroid]['xmax']) and
                int(centroid_dict[rectangle][1]) in range(rect_dict[centroid]['ymin'],rect_dict[centroid]['ymax'])): 

                # Add image with the smallest surface to overlap list. 
                if all_rectangles[rectangle]['surface'] > all_rectangles[centroid]['surface']: 
                    overlap.append(centroid)
                if all_rectangles[rectangle]['surface'] < all_rectangles[centroid]['surface']: 
                    overlap.append(rectangle)
    
    # Remove smallest images which overlap with other images.  
    for overlapped in set(overlap): 
        all_rectangles.pop(overlapped)

    return all_rectangles

def LoadModel(model_name): 
    '''
    :Goal:              To load a previously trained classification model. 
    :Param model_name:  String with the name of the used model. 
    :Returns:           Full loaded classification model. 
    '''
    # Load json file and create model.
    json_file = open(f"C://Users//tklop//ThesisADS//Models//{model_name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load saved weights into the created model.
    loaded_model.load_weights(f"C://Users//tklop//ThesisADS//Models//{model_name}.h5")
    print("Succesfully loaded the model")
    return loaded_model 

def CreateImageBatch(thresh_img, all_rectangles): 
    '''
    Function which creates a list of square images, with the size of the training data.     
    '''
    
    nh, nw = thresh_img.shape[:2]
    original = thresh_img.copy()
    image_batch = []
    idx_list = []

    
    testlist =[]
    x_sorted_rectangles = {}

    sorted_x = [testlist.append(all_rectangles[crops]['xmin']) for crops in all_rectangles] 
    sorted_x = sorted(testlist, reverse = False)

    for sort_idx, size_values in enumerate(sorted_x): 
        for idx, crop_name in enumerate(all_rectangles.keys()):
            if all_rectangles[crop_name]['xmin'] == sorted_x[sort_idx]: 
                x_sorted_rectangles[crop_name] = all_rectangles[crop_name]
    all_rectangles = x_sorted_rectangles
            

    for idx, rect_key in enumerate(all_rectangles.keys()):
        x, w = all_rectangles[rect_key]['xmin'], all_rectangles[rect_key]['xmax'] - all_rectangles[rect_key]['xmin']
        y, h = all_rectangles[rect_key]['ymin'], all_rectangles[rect_key]['ymax'] - all_rectangles[rect_key]['ymin']
        #print(f"{idx}: {x}, {y}, {w}, {h} ")
        
        if w > 0.15 * nw or h > 0.15*nh: # Should be done cleaner. (Gaussian) blurring? 
            
            # Save rectangles as images
            ROI = original[y-4:y+h + 4, x-4:x+w+4] 
            
            # Create border next to the most narrow side to create a square. 
            max_border_1 = (max(ROI.shape)-min(ROI.shape)) //2
            max_border_2 = (max(ROI.shape)-min(ROI.shape)) - max_border_1
            if h > w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = 20,
                                                    bottom = 20,
                                                    left = max_border_1 + 20,
                                                    right = max_border_2 + 20,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            if h < w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = max_border_1 + 20,
                                                    bottom = max_border_2 + 20,
                                                    left = 20,
                                                    right = 20,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            
            # Width and height of the trained on data. 
            width = 40
            height = 40
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(bordered_image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(f"Resized_{idx}.png", resized)
            image_batch.append(resized)
            idx_list.append(idx)

    return image_batch


def CreateImageCharacters(thresh_img, all_rectangles,img_size): 
    '''
    :Goal:                  Function which resizes and saves the images of single numbers from answers. 
    :Param thresh_img:      Threshold images as an array. 
    :Param all_rectangles:  Dictionairy with information about the coördinates of the characters. 
    :Param img_size:        Integer which which indicates what the size of the final square should be. 
    :Returns:               List of all found numbers stored as arrays. 
    '''
    # Define variables. 
    nh, nw = thresh_img.shape[:2]
    original = thresh_img.copy()
    x_coord_list =[]
    x_sorted_rectangles = {}
    image_batch = []
    idx_list = []

    # Sort boxes in normal order, based on ascending x-coördinates. 
    sorted_x = [x_coord_list.append(all_rectangles[crops]['xmin']) for crops in all_rectangles] 
    sorted_x = sorted(x_coord_list, reverse = False)

    # Sort the dictionairy based on the sorted list. 
    for sort_idx, size_values in enumerate(sorted_x): 
        for idx, crop_name in enumerate(all_rectangles.keys()):
            if all_rectangles[crop_name]['xmin'] == sorted_x[sort_idx]: 
                x_sorted_rectangles[crop_name] = all_rectangles[crop_name]
    all_rectangles = x_sorted_rectangles
            
    # Iterates through all images. 
    for idx, rect_key in enumerate(all_rectangles.keys()):
        # Get image specifications from dictionairy. 
        x, w = all_rectangles[rect_key]['xmin'], all_rectangles[rect_key]['xmax'] - all_rectangles[rect_key]['xmin']
        y, h = all_rectangles[rect_key]['ymin'], all_rectangles[rect_key]['ymax'] - all_rectangles[rect_key]['ymin']
                
        if w > 0.15 * nw or h > 0.15*nh: # Filter out noise. 
            
            # Cutout box. 
            ROI = original[y-4:y+h + 4, x-4:x+w+4] 
            
            # Create border next to the most narrow side to create a square. 
            max_border_1 = (max(ROI.shape)-min(ROI.shape)) //2
            max_border_2 = (max(ROI.shape)-min(ROI.shape)) - max_border_1
            if h > w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = 20,
                                                    bottom = 20,
                                                    left = max_border_1 + 20,
                                                    right = max_border_2 + 20,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            if h < w: 
                bordered_image = cv2.copyMakeBorder(ROI,
                                                    top = max_border_1 + 20,
                                                    bottom = max_border_2 + 20,
                                                    left = 20,
                                                    right = 20,
                                                    borderType = cv2.BORDER_CONSTANT,
                                                    value= (255,255,255)
                                                    )
            if h == w: 
                bordered_image = ROI

            # Width and height of the trained on data. 
            width = int(img_size)
            height = int(img_size)
            dim = (width, height)
            
            # Resize and save image to local location. 
            resized = cv2.resize(bordered_image, dim, interpolation = cv2.INTER_AREA)
            image_batch.append(resized)
            idx_list.append(idx)

    return image_batch

def PrintResults(df): 
    '''
    :Goal:      To print the results of a single test. 
    :Param df:  Pandas dataframe containing all information about a single test. 
    :Return:    No return specified. 
    '''
    
    # Count wrong answers. 
    score_0 = df['points'][df['points']== 0].count()
    # Count correct answers. 
    score_1 = df['points'][df['points']== 1].count()
    print(f"Total score of this student is {score_1} out of {score_0 + score_1}")
    # Calculate the score of all the open answers. 
    grade = (score_1 / (score_0 + score_1))*10
    print(f"This suggests a grade of: {grade:.2f}")


if __name__ == "__main__": 
    # Define Variables. 
    mypath = "C://Users//tklop//ThesisADS//EquationImages//answers"
    img_size = int(40)
    model_name = "augmented_kaggle"
    df = pd.DataFrame(columns= ['file_name', 'class_dict', 'answer_student', 'correct_answer', 'points'])
    
    # Load model before the for loop. 
    loaded_model = LoadModel(model_name)

    # Loop through all images in a folder. 
    img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for name_index, name in enumerate(img_names): 
        # Load image. 
        img = cv2.imread(f"C://Users//tklop//ThesisADS//EquationImages//answers//{name}")
        # Preprocess image. 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_image_plot = img_gray.copy()
        ret, thresh_img = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
        thresh_img, org_without_line = RemoveLine(img_gray, thresh_img)
        org_without_line_connected = thresh_img.copy()
        contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        all_rectangles = GetAllRectangles(thresh_img)
        nested_rectangles = OverlappingRectangles(all_rectangles)
        batch = CreateImageBatch(thresh_img, nested_rectangles)
        
        # Define (or reset) variables.
        classified_list = []
        prob_list = []
        img_number = 1

        # Classify each character in the batch. 
        # (Most) Print statements are hidden to improve readability of the output. 
        # print(f"File: {name}")   
        for idx, value in enumerate(batch):  
            batch_img = batch[idx]
            batch_img = batch_img.reshape(1,40,40,1)
            y_prob = loaded_model.predict(batch_img) 
                
            # print(f"Object number {img_number} is classfied as: {y_prob.argmax(axis=-1)[0]}, with a prob of: {np.max(y_prob)}")
            # Append results.
            prob_list.append(np.max(y_prob))
            classified_list.append(y_prob.argmax(axis=-1)[0])
           
            img_number += 1
        
        # Create dictionairy of classified numbers. 
        classified_dict = dict(zip(range(1,len(classified_list) + 1), classified_list ))
        
        # Join classified numbers together as string. 
        classified_list_str = [str(i) for i in classified_list]
        answer = ''.join(classified_list_str)

        # Check if answer is correct. 
        if answer == name[4:-4]: # Based on image labels in which the correct answer was in the image name. e.g. 'tms_48.jpg'
            points = 1
        else: 
            points = 0

        # Fill dataframe with data. 
        df = df.append({'file_name': name,
                        'class_dict': classified_dict,
                        'answer_student': answer,
                        'correct_answer': name[4:-4], 
                        'points': points},
                        ignore_index=True)
        
    # Print results. 
    PrintResults(df)
    print("\n")
 
    # Plot the (last) figure showing the cleaning and classification process.
    fig = plt.figure(figsize=(40, 40))
    columns = len(batch)

    # Plot original image. 
    ax1 = fig.add_subplot(2, columns, 1)
    ax1.title.set_text('Original Image')
    plt.imshow(original_image_plot, cmap = 'gray')

    # Plot image with the 'answer line' removed. 
    ax2 = fig.add_subplot(2, columns, 2)
    ax2.title.set_text('Binary Image Without Line')
    plt.imshow(org_without_line, cmap = 'gray')

    # Plot the image in which the numbers are reconnected. 
    ax3 = fig.add_subplot(2, columns,3)
    ax3.title.set_text('Binary Image Connected')
    plt.imshow(org_without_line_connected , cmap = 'gray')

    # Plot the individual characters which are classified. 
    for i in range(1, columns + 1):
        img = batch[i-1]
        ax = fig.add_subplot(1, columns, i)
        ax.title.set_text(f"{classified_list[i-1]} (prob: {prob_list[i-1]})") # Add classification answer and corresponding probability as title. 
        plt.imshow(img, cmap= "gray")
        
    plt.show()

