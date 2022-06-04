'''
:Author: Tom Klopper. 
:Goal: To transform unprocessed answers into cleaned single characters for further analysis. 
:Last update: 04-06-2022.
'''

# Import Libraries.
import cv2 
import numpy as np
from os import listdir
from os.path import isfile, join

# Define Functions. 
def RemoveLine(img_gray, thresh_img): 
    '''
    :Goal:              Function which removes the "answer line" and, if necessary, reconnects the numbers. 
    :Param img_gray:    Grayscaled image as an array.
    :Param thresh_img:  Threshold image as an array.
    :Returns:            Returns threshhold image as an array. 
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
                    
    return thresh_img


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


def CreateImageCharacters(thresh_img, all_rectangles, name, img_size, test_name): 
    '''
    :Goal:                  Function which resizes and saves the images of single numbers from answers. 
    :Param thresh_img:      Threshold images as an array. 
    :Param all_rectangles:  Dictionairy with information about the coördinates of the characters. 
    :Param name:            String which contains the name of the given answer. 
    :Param img_size:        Integer which which indicates what the size of the final square should be. 
    :Param test_name:       String which contains the name of the test. 
    :Returns:               No return specified. 
    '''
    # Define variables. 
    nh, nw = thresh_img.shape[:2]
    original = thresh_img.copy()
    x_coord_list =[]
    x_sorted_rectangles = {}

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
            cv2.imwrite(f"C://Users//tklop//ThesisADS//CharacterImages//{img_size}x{img_size}//{test_name}_{name[:-4]}_{idx}.jpg", resized) # Make name more general, i.e.: answer1_1 
            


# Run the code. 
if __name__ == "__main__":
    # Iterate through all test files. 
    test_path =  "C://Users//tklop//ThesisADS//AllAnswers"
    test_names = [f for f in listdir(test_path)]
    for idx_test, test_name in enumerate(test_names): 
        # Get path from single test and iterate through all images in a test. 
        mypath = f"C://Users//tklop//ThesisADS//AllAnswers//{test_name}//crops//handwritten"
        img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        img_sizes = ['28', '40']
        for idx_names, name in enumerate(img_names): 
            # Open image. 
            img = cv2.imread(f"C://Users//tklop//ThesisADS//AllAnswers//{test_name}//crops//handwritten//{name}")
            # Preproces image. 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
            img = RemoveLine(img_gray, thresh_img)
            all_rectangles = GetAllRectangles(img)
            nested_rectangles = OverlappingRectangles(all_rectangles)
            # Save all characters from image in all necessary sizes. 
            for img_size in img_sizes: 
                CreateImageCharacters(img, nested_rectangles, name, img_size, test_name)


