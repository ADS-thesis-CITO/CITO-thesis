# import libraries
import os.path

import numpy as np
import pandas as pd
from scipy.signal import convolve2d

import cv2

import matplotlib.pyplot as plt

from skimage import io

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from pathlib import Path

import re

from os import listdir, mkdir
from os.path import isfile, join, isdir, basename

from zipfile38 import ZipFile

import fitz
from PIL import Image

import pytesseract

# indicate Tesseract path, adapt this when running at a different computer
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from yolov5.detect import find_cropped_images, run


def pdf_to_image(path):
    """
    Takes the path of a pdf and returns a list of cv2 images, one for every pdf page.
    :param path: path of pdf
    :return: list of images of pdf pages
    """

    pages = fitz.open(path)

    img_list = []

    for page in pages:
        zoom = 25
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode,
                              [pix.width, pix.height],
                              pix.samples)

        # convert to cv2 format
        img = np.array(img)

        # convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_list.append(img)
    return img_list


def recognize_nums(img, psm=7, threshold=0.5):
    """
    Recognizes numbers using Tesseract.
    :param img: cv2 image to recognise numbers in
    :param psm: int that represents the internal settings of Tesseract
    :param threshold: float that represents how certain Tesseract should be of a number, higher is more certain
    :return: number recognised
    """

    config = r'-c tessedit_char_whitelist=0123456789 --psm ' + str(psm)

    if threshold == 0.5:
        text = pytesseract.image_to_string(img, config=config)
    else:
        text = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

        # convert confidence to float
        conf = [float(item) for item in text['conf']]

        # first item's index that equals the maximum
        max_conf_index = [index for index, item in enumerate(text['text']) if conf[index] == max(conf)][0]

        if conf[max_conf_index] < (1 - threshold) * 100:
            text = text['text'][max_conf_index]
        else:
            text = ''

    return text


def show_image(img, window_height=500):
    """
    Display the cv2 image and close upon closing the window.
    :param img: cv2 image
    :param window_height: window height in pixels
    :return: empty
    """

    height = img.shape[0]
    width = img.shape[1]
    ratio = height / float(width)

    # set the dimensions
    dim = (window_height, int(window_height * ratio))

    img = cv2.resize(img, dim)
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def add_borders(img, width=4961, height=7016):
    """
    Add borders to an image to make it fit a specified size
    :param img: cv2 image to extend
    :param width: desired width of the output image in pixels
    :param height: desired height of the output image in pixels
    :return: cv2 image with borders
    """

    img_height = img.shape[0]
    img_width = img.shape[1]

    border_image = cv2.copyMakeBorder(img,
                                      0,
                                      height - img_height,
                                      0,
                                      width - img_width,
                                      cv2.BORDER_CONSTANT,
                                      None,
                                      (255, 255, 255))

    border_image = cv2.cvtColor(border_image,
                                cv2.COLOR_RGB2BGR)

    return border_image


def extract_coordinates_questions_img(image):
    """
    Based on an image, extract the question numbers and the coordinates that correspond to the question number on
    the input image.
    :param image: cv2 BGR image
    :return: list of question coordinates, list of question numbers
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # extract only the part with the page numbers, which is 17% of the page on the left side
    height = gray_image.shape[0]
    width = gray_image.shape[1]
    gray_image = gray_image[:, :(int(width * .17))]

    # thresholding
    ret2, gray_image = cv2.threshold(gray_image,
                                     100,
                                     255,
                                     cv2.THRESH_BINARY)

    gray_copy = gray_image.copy()

    # set a kernel for eroding the image
    kernel = np.ones((9, 9), np.uint8)

    # eroding the image
    gray_image = cv2.erode(gray_image, kernel, iterations=8)

    # find contours to find the left top corner of the erosion
    # this is the beginning coordinate for the new question
    # the tilde ~ inverts the image for the contours
    contours, hierarchy = cv2.findContours(~gray_image,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # list of the coordinates of the numbers
    num_coords = []

    # list of the question numbers
    question_nums = []

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # skip if too close to upper, lower or side boundary
        if y < height / 40 or y + h > height * 39 / 40 or x < width / 8:
            continue

        # slice out the question number
        question_num_img = gray_copy[y:y + h, x:x + w]

        # fix the image by removing horizontal white stripes with morphologyEx, this is noise
        kernel = np.zeros((3, 3), np.uint8)
        kernel[:, 1] = 1

        question_num_img = cv2.morphologyEx(question_num_img,
                                            cv2.MORPH_OPEN,
                                            kernel=kernel)

        # set a threshold to avoid the misclassification of numbers
        question_num = recognize_nums(question_num_img, threshold=0.8)

        # set unknown if it is not known else make int
        question_num = "unknown" if question_num == "" else int(question_num)

        # add to array
        if question_num != "unknown":
            num_coords.append((x, y, w, h))
            question_nums.append(question_num)

            # show_image(question_num_img)

    # reverse the list to go from top to bottom
    num_coords = num_coords[::-1]
    question_nums = question_nums[::-1]

    # store the coordinates in a list
    coordinate_list = []

    if len(num_coords) == 1:
        # set the right limit to 47% of the page to ignore the scratch area
        coordinate_list = [[0, height, 0, int(width * 0.47)]]

    elif len(num_coords) == 0:

        coordinate_list = []

    else:
        for index, coords in enumerate(num_coords):
            # trim original image and add to array

            # skip first, because we care about the differences
            if index == 0:
                # store coordinates to compare with next
                previous_coords = coords
                continue

            # set the right limit to 47% of the page to ignore the scratch area
            # cut 130 pixels to the top to avoid cutting the answer in half
            coordinate_list.append([(previous_coords[1] - 130), (coords[1] - 130), 0, int(width * 0.47)])

            # at the last step, also do the same to the bottom of the page
            if index == len(num_coords) - 1:
                # set the right limit to 47% of the page to ignore the scratch area
                # cut 130 pixels to the top to avoid cutting the answer in half
                coordinate_list.append([(coords[1] - 130), height, 0, int(width * 0.47)])

            # store coordinates to compare with next
            previous_coords = coords

    return coordinate_list, question_nums


def extract_first_page(img):
    """
    Extract school number and student number from first page of test using Tesseract
    :param img: cv2 image of first page
    :return: school number, student number
    """

    text = pytesseract.image_to_string(img)

    text_list = re.findall("[A-Za-z0-9]*", text)

    # take the empty strings out of the list
    text_list = list(filter(None, text_list))

    # initial value for school and student number
    school_num = "unknown"
    student_num = "unknown"

    for index, item in enumerate(text_list):

        # read the text after the word 'school'
        if item.lower() == 'school':
            school_num = text_list[index + 1]

        # read the text after the word 'nummer'
        if item.lower() == 'nummer':
            student_num = text_list[index + 1]

    return school_num, student_num


def question_coordinate_list(img_dir):
    """
    Based on the path of the directory containing all images for one test, return the coordinates of the questions,
    the corresponding question numbers and the number of questions that are present per page, to keep track of which
    page the questions are on.
    :param img_dir: string, directory where all images are located for one test
    :return: list of test question coordinates, list of question numbers and the number of questions per page
    """

    files = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))]

    for index, f in enumerate(listdir(img_dir)):

        # remove irrelevant pages
        if f in ['0.jpg', '1.jpg', '2.jpg', '10.jpg', '15.jpg']:
            files.remove(join(img_dir, f))

    list_coordinate_lists = []
    list_question_nums = []
    num_questions_on_page = []

    for index, file in enumerate(files):
        # convert to cv2 first

        coordinate_list, question_nums = extract_coordinates_questions_img(cv2.imread(file))

        list_coordinate_lists.append(coordinate_list)
        list_question_nums.append(question_nums)

        # add the number of questions on the page to keep track of which coordinates belong on which page
        num_questions_on_page.append(len(coordinate_list))

    # flatten these lists to obtain lists of results, ordered by what order they come across the page
    coordinate_list = [coordinates for sublist in list_coordinate_lists for coordinates in sublist]
    question_nums = [question_num for sublist in list_question_nums for question_num in sublist]

    return coordinate_list, question_nums, num_questions_on_page


def find_question_list_most_common(coordinate_list, question_nums, num_questions_on_page_list):
    """
    Finds the question number list that occurs most common, which is assumed to be the correct one. Check whether this
    is correct. Do not account for question number lists that do not contain consecutive numbers only, as this is an
    indication that something went wrong.
    :param coordinate_list: list of lists containing the the coordinates to cut off
    :param question_nums: list of lists containing the (unsorted) question numbers
    :param num_questions_on_page_list: list containing the number of questions per page for each test
    :return: mean coordinates belonging to the most common question list that are valid, the corresponding question
    list, the the corresponding frequency in which it occurs and the list with the number of questions per page on the
    test.
    """

    max_freq = 0
    max_freq_question_list = []
    for index, question_list in enumerate(question_nums):

        # first filter the question lists that are not 1:len(question_list) when sorted as these are the only valid ones
        expected_list = []
        for i, _ in enumerate(question_list):
            expected_list.append(i + 1)

        if sorted(question_list) != expected_list:
            # skip if it is not the expected list
            continue

        current_freq = question_nums.count(question_list)
        if current_freq > max_freq:
            max_freq_question_list = question_list
            max_freq = current_freq

    valid_coordinates_list = []

    # max_freq_question_list is the question list occurring most
    for index, question_list in enumerate(question_nums):
        if max_freq_question_list == question_list:
            valid_coordinates_list.append(coordinate_list[index])
            num_questions_on_page = num_questions_on_page_list[index]

    mean_coordinates = np.mean(valid_coordinates_list, axis=0)

    # convert to int
    mean_coordinates = [[int(coord) for coord in sublist] for sublist in mean_coordinates]

    return mean_coordinates, max_freq_question_list, max_freq, num_questions_on_page


def find_coordinates(img_dirs, max_freq_req=5):
    """
    Using the directory containing a separate directory with images for every test, find the coordinates for all
    questions, their question numbers and the number of questions per page to keep track which questions belong to
    which page.
    :param img_dirs: directory containing a separate directory with images for every test
    :param max_freq_req: the frequency a question list should appear, so with the default of 5, a list of question
    numbers should appear at least 5 times
    :return: list of mean coordinates, their question numbers and the number of questions per page
    """

    directories = [join(img_dirs, f) for f in listdir(img_dirs) if isdir(join(img_dirs, f))]

    coordinate_list = []
    question_nums_list = []
    num_questions_on_page_list = []

    for index, direc in enumerate(directories):
        print('Finding coordinates in directory: ', index)

        coordinates, question_nums, num_questions_on_page = question_coordinate_list(direc)

        coordinate_list.append(coordinates)
        question_nums_list.append(question_nums)
        num_questions_on_page_list.append(num_questions_on_page)

        if index > max_freq_req:
            mean_coordinates, correct_question_list, max_freq, num_questions_on_page_final = \
                find_question_list_most_common(coordinate_list, question_nums_list, num_questions_on_page_list)

            # if max_freq > max_freq_req, then we are confident this is the correct question list
            # then break, since we don't have to run it every time.
            if max_freq > max_freq_req:
                break

    return mean_coordinates, correct_question_list, num_questions_on_page_final


def question_list_using_coords(img_dir, mean_coordinates, correct_question_list, num_questions_on_page):
    """
    Finds the images and question numbers for all questions for a test, while inputting the directory containing all
    images for one test. Also returns the school number and question number of a student.
    :param img_dir: path of directory containing the images for all individual questions
    :param mean_coordinates: coordinates of all questions on their respective pages
    :param correct_question_list: list of question numbers, in corresponding order to the coordinates
    :param num_questions_on_page: number of questions per page, to extract the right amount of questions
    :return: list of images (one image per question), list of question numbers, school number and student number
    """

    # make copies
    coordinates = mean_coordinates[:]
    question_list = correct_question_list[:]
    questions_on_page = num_questions_on_page[:]

    # make a list of all paths of the images
    files = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))]

    # initialise school number and student number, in case the first page is not found
    school_num = "unknown"
    student_num = "unknown"

    for index, f in enumerate(listdir(img_dir)):

        # use information of first page
        if f == '0.jpg':
            first_page_path = join(img_dir, f)
            first_page = cv2.imread(first_page_path)

            # extract the information for the first page
            school_num, student_num = extract_first_page(first_page)

        # remove irrelevant pages, all pages that contain no questions
        if f in ['0.jpg', '1.jpg', '2.jpg', '10.jpg', '15.jpg']:
            files.remove(join(img_dir, f))

    list_image_lists = []
    list_question_nums = []

    for index, file in enumerate(files):
        # convert to cv2 first

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # find the number of questions on this page, select those to be the question numbers and coordinates
        num_questions_this_page = questions_on_page[0]
        question_nums = question_list[:num_questions_this_page]
        coords = coordinates[:num_questions_this_page]

        # extract the exercises using the image, coordinates and question numbers
        image_list, question_nums = extract_exercise_using_coords(image, coords, question_nums)

        # add this information to their lists
        list_image_lists.append(image_list)
        list_question_nums.append(question_nums)

        # delete the information from the beginning of the list, so that the new information is at the beginning of
        # the list
        del question_list[:num_questions_this_page]
        del coordinates[:num_questions_this_page]
        del questions_on_page[0]

    # flatten these lists to obtain lists of results
    image_list = [image for sublist in list_image_lists for image in sublist]
    question_nums = [question_num for sublist in list_question_nums for question_num in sublist]

    return image_list, question_nums, school_num, student_num


def extract_exercise_using_coords(image, coords, question_nums):
    """
    Using an image, the coordinates of the individual questions and their respective question numbers, cut out the
    individual questions from the image, add a border to them to maintain the original dimensions and return a list of
    these images, as well as a list of the corresponding question numbers.
    :param image: cv2 BGR image with one or more questions on them
    :param coords: list containing the coordinates for the cutout of every question
    :param question_nums: list containing the corresponding question numbers to the coordinates
    :return: list of question images, list of question numbers
    """

    # store the images in a list
    image_list = []

    for index, coord in enumerate(coords):
        # trim original image and add to array
        y, h, x, w = coord

        sliced_image = image[y:h, x:w]

        image_list.append(sliced_image)

    for index, img in enumerate(image_list):
        # add borders
        border_image = add_borders(img)

        # make grayscale
        image_list[index] = cv2.cvtColor(border_image,
                                         cv2.COLOR_BGR2GRAY)

    return image_list, question_nums


def make_df(directories, limit=None):
    """
    Makes the dataframe with all the information of the tests.
    :param directories: list of all test directories containing the jpg's of every test.
    :param limit: maximum amount of tests to put in the dataframe, useful to avoid memory issues during testing
    :return: dataframe with for every row (representing a test) the school number, student number, list of question
    numbers and list of cropped question images. The question number is set to -1 if an image is not found, hence
    the number and path should simply be skipped.
    """

    #TODO: this function causes memory issues, which is a problem to be solved

    # for checking whether it is a directory
    dirs = [join(directories, f) for f in listdir(directories) if isdir(join(directories, f))]

    # make dataframe for all info

    df = pd.DataFrame(columns=['School number', 'Student number', 'Question numbers',
                               'Cropped coord list', 'Path list'])

    mean_coordinates, correct_question_list, num_questions_on_page = find_coordinates(directories)

    if limit is None:
        limit = len(dirs)

    for index, test_dir in enumerate(dirs):
        if index >= limit:
            break

        print(index)

        image_list, question_nums, school_num, student_num = question_list_using_coords(test_dir, mean_coordinates,
                                                                                        correct_question_list,
                                                                                        num_questions_on_page)

        for item_nr, image in enumerate(image_list):
            # convert to BGR, as YOLO expects a BGR image
            image_list[item_nr] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        cropped_images, question_nums_found =\
            find_cropped_images(image_list, question_nums,
                                weights=r"yolov5\\runs\\train\\exp45\\weights\\best_new.pt",
                                data=r"yolov5\\models\\cito.yaml")

        #TODO: Issue now is that if no coordinates were found for a crop, that the length of question_nums is not equal
        # to the length of cropped_images (the coordinate list). So also take the question numbers from the function
        # and match the paths with that as well.
        # NOTE: that now only holds for the path list, since a path is used multiple times.

        paths = []
        path_full_list = [join(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]

        # add the path num_questions_on_page times for each file
        for i, num in enumerate(num_questions_on_page):
            for j in range(num):
                paths.append(path_full_list[i])

        df2 = pd.DataFrame({'School number': school_num,
                            'Student number': student_num,
                            'Question numbers': [np.array(question_nums_found)],
                            'Cropped coord list': [np.array(cropped_images)],
                            'Path list':[np.array(paths)]})

        df = pd.concat([df, df2],
                       ignore_index=True,
                       axis=0)

    return df


def save_questions_using_coords(directories, subdirectory_name_questions):
    """
    Save the questions within the directory in all directories with the subdirectory name
    :param directories: directory containing a directory with images for each test
    :param subdirectory_name_questions: name of directory within all test directories
    :return: nothing, saves files
    """

    mean_coordinates, correct_question_list, num_questions_on_page = find_coordinates(directories)

    # for checking whether it is a directory
    dirs = [join(directories, f) for f in listdir(directories) if isdir(join(directories, f))]

    for index, test_dir in enumerate(dirs):

        print(index)

        image_list, question_nums, school_num, student_num = question_list_using_coords(test_dir, mean_coordinates,
                                                                                        correct_question_list,
                                                                                        num_questions_on_page)

        subdirectory_question_path = join(test_dir, subdirectory_name_questions)

        if not os.path.exists(subdirectory_question_path):
            mkdir(subdirectory_question_path)

        # use the question number for the file name, since they shouldn't overlap already
        for question, image in zip(question_nums, image_list):
            path = join(subdirectory_question_path, str(question) + '.png')

            io.imsave(path, image)


def make_zip_questions(zip_path, directories):
    """
    Makes a zip file with the directories for every question in there
    :param zip_path: path of the zip file to create
    :param directories: Directory containing all directories for every individual question
    :return:
    """

    dirs = [join(directories, f) for f in listdir(directories) if isdir(join(directories, f))]

    # create zipfile
    with ZipFile(zip_path, 'w') as zip:

        # list of all test directories
        for index, dir_list in enumerate(dirs):
            print(index)

            # subdirectories in dir
            subdirs = [join(dir_list, f) for f in listdir(dir_list) if isdir(join(dir_list, f))]

            # list of both directories containing the images
            for subdir in subdirs:
                files = [join(subdir, f) for f in listdir(subdir) if isfile(join(subdir, f))]

                for file in files:
                    zip.write(file)


def classify_individual_multiple_choice(img, psm=7):
    """
    Using the cutout of the multiple choice answer, find the number found.
    :param img: Image of cutout
    :param psm: internal settings of multiple choice
    :return:
    """

    config = r'-c tessedit_char_whitelist=ABCD --psm ' + str(psm)

    text = pytesseract.image_to_string(img, config=config)

    if text != '':
        text = text[0]

    return text


def classify_multiple_choice(directory, start_mc=21, end_mc=30):
    """
    Assuming that all multiple choice questions are together, classify the multiple choice answers.
    :param directory: Directory containing all subdirectories with, should be the detect folder from YOLO
    :param start_mc: First question number of multiple choice questions
    :param end_mc:
    :return:
    """

    dir_list = [join(directory, f) for f in listdir(directory) if isdir(join(directory, f))]

    image_dirs = [join(subdir, 'crops/handwritten') for subdir in dir_list]

    for image_dir in image_dirs:
        files = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]

        # extract the names only
        file_nums = [int(f.split('.')[0]) for f in listdir(image_dir) if isfile(join(image_dir, f))]

        for index, file_num in enumerate(file_nums):

            # if the file number does not indicate that it's multiple choice, skip this iteration
            if file_num < start_mc or file_num > end_mc:
                continue

            print(file_num)

            # otherwise, extract the image, print the found answer with tesseract and show the image
            img = cv2.imread(files[index])

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thresh_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)

            fixed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
            fixed_img = cv2.morphologyEx(fixed_img, cv2.MORPH_CLOSE, kernel)

            text = classify_mc_cutout(fixed_img)

            print(text)



def classify_multiple_choice_tesseract():
    """
    Assuming that all multiple choice questions are together, classify the multiple choice answers.
    :param directory: Directory containing all subdirectories with, should be the detect folder from YOLO
    :param start_mc: First question number of multiple choice questions
    :param end_mc:
    :return:
    """

    directory = 'Labeled_multiple_choice'

    dir_list = [join(directory, f) for f in ['A', 'B', 'C', 'D']]

    correct_labels = []
    predicted_labels = []

    for index, image_dir in enumerate(dir_list):
        print(index)

        image_paths = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f))]

        for path in image_paths:

            correct_labels.append(['A', 'B', 'C', 'D'][index])

            # otherwise, extract the image, print the found answer with Tesseract and show the image
            img = cv2.imread(path)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thresh_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)

            fixed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
            fixed_img = cv2.morphologyEx(fixed_img, cv2.MORPH_CLOSE, kernel)

            text = classify_individual_multiple_choice(fixed_img)

            if text == '':
                predicted_labels.append('other')
            else:
                predicted_labels.append(text)

    print(confusion_matrix(correct_labels, predicted_labels))
    print(classification_report(correct_labels, predicted_labels))


def find_overlapping_areas(img1, img2, x, y):
    """
    Find overlapping areas of two cv2 images to each other at coordinates x,y for the bigger image img1.
    :param img1: Larger cv2 image
    :param img2: Smaller cv2 image
    :param x: x coordinate in img1 add img2 to, using left top of img2
    :param y: y coordinate in img1 add img2 to, using left top of img2
    :return:
    """

    print(img1.shape)

    height = img2.shape[0]
    width = img2.shape[1]

    # reduce img1
    img1 = img1[y:(y+height), x:(x+width)]

    print(img1.shape)
    print(img2.shape)

    result_image = np.ones((height, width, 3), np.uint8)*255

    # logical operator results
    or_op = np.logical_or(img1 < 200, img2 < 200)
    and_op = np.logical_and(img1 > 200, img2 > 200)

    # add the non-overlapping part as black
    result_image[or_op] = (0,0,0)

    # add the overlapping part as green
    result_image[and_op] = (0,255,0)

    show_image(img1)
    show_image(img2)

    show_image(result_image)

    return result_image


def classify_mc_cutout(mc_image):
    """
    Use the convolve operation to find the minimum multiplication of the inverse image with the examples of A, B, C and
    D. The function tries to see the ideal location on the image. A perfect match would mean that at the right location
    this convolve operation gives a sum of 0. The minimum of the minima for the four images is the best match.

    Note: this function only works for the given resolution, for a new resolution, the images of A, B, C and D should be
    made over.
    :param mc_image: Image to classify
    :return:
    """

    # make copy of image
    mc_image_copy = mc_image.copy()

    # normalise
    mc_image = mc_image/255

    # read images
    a_img = cv2.imread('mc_ideal_pictures/A.jpg')
    b_img = cv2.imread('mc_ideal_pictures/B.jpg')
    c_img = cv2.imread('mc_ideal_pictures/C.jpg')
    d_img = cv2.imread('mc_ideal_pictures/D.jpg')

    # convert to grayscale and normalise
    a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)/255
    b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)/255
    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)/255
    d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)/255

    # find the sums
    a_sum = a_img.sum()
    b_sum = b_img.sum()
    c_sum = c_img.sum()
    d_sum = d_img.sum()

    #print(a_sum, b_sum, c_sum, d_sum)

    # invert
    a_img_inv = 1-a_img
    b_img_inv = 1-b_img
    c_img_inv = 1-c_img
    d_img_inv = 1-d_img

    # these inverses should be as close to 0 as possible, risk is all zero area's
    A_convolve_inv = convolve2d(mc_image, a_img_inv, 'valid')
    B_convolve_inv = convolve2d(mc_image, b_img_inv, 'valid')
    C_convolve_inv = convolve2d(mc_image, c_img_inv, 'valid')
    D_convolve_inv = convolve2d(mc_image, d_img_inv, 'valid')

    # because of the previous risk, also check whether the normal images overlap by checking whether their convolves
    # are close to the mean of the ideal pictures. This indicates good overlapping. Risk here is finding all 1 area's
    A_convolve = convolve2d(mc_image, a_img, 'valid')
    B_convolve = convolve2d(mc_image, b_img, 'valid')
    C_convolve = convolve2d(mc_image, c_img, 'valid')
    D_convolve = convolve2d(mc_image, d_img, 'valid')

    # to ensure both hold at the same time, find the distance to 0 and the distance to the mean from the previous two,
    # respectively. Do this with the squared Euclidean distance, where both matrices are seen as a dimension. Find the
    # minimum of that distance.
    A_dist = ((A_convolve_inv - 0) ** 2 + (A_convolve - a_sum) ** 2)
    B_dist = ((B_convolve_inv - 0) ** 2 + (B_convolve - b_sum) ** 2)
    C_dist = ((C_convolve_inv - 0) ** 2 + (C_convolve - c_sum) ** 2)
    D_dist = ((D_convolve_inv - 0) ** 2 + (D_convolve - d_sum) ** 2)

    A_dist_min = A_dist.min()
    B_dist_min = B_dist.min()
    C_dist_min = C_dist.min()
    D_dist_min = D_dist.min()

    # find minimum index of all, done using
    # https://www.codegrepper.com/code-examples/python/Using+np.unravel_index+on+argmax+output
    A_min_index = np.unravel_index(np.argmin(A_dist), A_dist.shape)
    B_min_index = np.unravel_index(np.argmin(B_dist), B_dist.shape)
    C_min_index = np.unravel_index(np.argmin(C_dist), C_dist.shape)
    D_min_index = np.unravel_index(np.argmin(D_dist), D_dist.shape)

    # print the minimum distances
    print(A_dist_min, B_dist_min, C_dist_min, D_dist_min)

    # minimum index, find the best match
    min_ind = np.array([A_dist_min, B_dist_min, C_dist_min, D_dist_min]).argmin()

    index_best_img = [A_min_index, B_min_index, C_min_index, D_min_index][min_ind]

    # make BGR image
    mc_image_copy = cv2.cvtColor(mc_image_copy, cv2.COLOR_GRAY2BGR)

    # if uncommented, draw circles at these locations to detect location
    # cv2.circle(mc_image_copy, (index_best_img[1], index_best_img[0]), 5, (255, 0, 0), -1)

    print(['A', 'B', 'C', 'D'][min_ind])

    return ['A', 'B', 'C', 'D'][min_ind]


def label_mc_crops():
    """
    Label the cropped images by taking input when showing the picture. Does this by hand by selecting the right number.
    The cropped images are saved in the directory assigned to the label.
    :return:
    """

    output_dir = "C:/Users/ajtis/Documents/Master/Thesis/pythonProject/Labeled_multiple_choice"

    images_path = "C:/Users/ajtis/Documents/Master/Thesis/pythonProject/yolov5/runs/detect"

    file_dirs = [join(images_path, f, 'crops', 'handwritten') for f in listdir(images_path) if isdir(join(images_path,
                                                                                                          f))]

    crop_files = [join(file_dir, file) for file_dir in file_dirs for file in listdir(file_dir) if isfile(join(file_dir,
                                                                                                              file))]

    crop_files = [crop_file for crop_file in crop_files if int(basename(crop_file).split('.')[0]) > 20]

    A_count = 0
    B_count = 0
    C_count = 0
    D_count = 0
    unknown_count = 0

    # iterate over the files
    for index, crop_file in enumerate(crop_files):
        print(index)

        img = cv2.imread(crop_file)

        cv2.imshow('Select a, b, c, d or u', img)

        key = cv2.waitKey()

        # use the keys corresponding to the letters
        if key == 97:
            path = join(output_dir, 'A', str(A_count) + '.jpg')
            saved = cv2.imwrite(path, img)
            A_count += 1

        elif key == 98:
            path = join(output_dir, 'B', str(B_count) + '.jpg')
            saved = cv2.imwrite(path, img)
            B_count += 1

        elif key == 99:
            path = join(output_dir, 'C', str(C_count) + '.jpg')
            saved = cv2.imwrite(path, img)
            C_count += 1

        elif key == 100:
            path = join(output_dir, 'D', str(D_count) + '.jpg')
            saved = cv2.imwrite(path, img)
            D_count += 1

        elif key == 117:
            path = join(output_dir, 'unknown', str(unknown_count) + '.jpg')
            saved = cv2.imwrite(path, img)
            unknown_count += 1

        print(path)

        # if saving the image has failed
        if not saved:
            print('Saving the image has failed.')


def df_cropped_numbers():
    """
    Find the minimum convolving dissimilarity to each ideal number, put it in a df and save the results to a csv file.
    :return: DataFrame containing the minimum dissimilarities, labels and path
    """

    df = pd.DataFrame(columns = ['label', 'min_A', 'min_B', 'min_C', 'min_D', 'path'])

    cropped_dir = "C:/Users/ajtis/Documents/Master/Thesis/pythonProject/Labeled_multiple_choice"

    # read images
    a_img = cv2.imread('mc_ideal_pictures/A.jpg')
    b_img = cv2.imread('mc_ideal_pictures/B.jpg')
    c_img = cv2.imread('mc_ideal_pictures/C.jpg')
    d_img = cv2.imread('mc_ideal_pictures/D.jpg')

    # convert to grayscale and normalise
    a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY) / 255
    b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY) / 255
    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY) / 255
    d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY) / 255

    # find the sums
    a_sum = a_img.sum()
    b_sum = b_img.sum()
    c_sum = c_img.sum()
    d_sum = d_img.sum()

    # invert
    a_img_inv = 1 - a_img
    b_img_inv = 1 - b_img
    c_img_inv = 1 - c_img
    d_img_inv = 1 - d_img

    count = 0

    for label in ['A', 'B', 'C', 'D']:

        cropped_path_list = [join(cropped_dir, label, f) for f in listdir(join(cropped_dir, label)) if
                             isfile(join(cropped_dir, label, f))]

        for path in cropped_path_list:

            print(count)
            count += 1

            mc_image = cv2.imread(path)

            mc_image = cv2.cvtColor(mc_image, cv2.COLOR_BGR2GRAY)

            # threshold
            ret, mc_image = cv2.threshold(mc_image, 60, 255, cv2.THRESH_BINARY)

            # normalise
            mc_image = mc_image / 255

            # these inverses should be as close to 0 as possible, risk is all zero area's
            A_convolve_inv = convolve2d(mc_image, a_img_inv, 'valid')
            B_convolve_inv = convolve2d(mc_image, b_img_inv, 'valid')
            C_convolve_inv = convolve2d(mc_image, c_img_inv, 'valid')
            D_convolve_inv = convolve2d(mc_image, d_img_inv, 'valid')

            # because of the previous risk, also check whether the normal images overlap by checking whether their convolves
            # are close to the mean of the ideal pictures. This indicates good overlapping. Risk here is finding all 1 area's
            A_convolve = convolve2d(mc_image, a_img, 'valid')
            B_convolve = convolve2d(mc_image, b_img, 'valid')
            C_convolve = convolve2d(mc_image, c_img, 'valid')
            D_convolve = convolve2d(mc_image, d_img, 'valid')

            # to ensure both hold at the same time, find the distance to 0 and the distance to the mean from the previous two,
            # respectively. Do this with the squared Euclidean distance, where both matrices are seen as a dimension. Find the
            # minimum of that distance.
            A_dist = ((A_convolve_inv - 0) ** 2 + (A_convolve - a_sum) ** 2)
            B_dist = ((B_convolve_inv - 0) ** 2 + (B_convolve - b_sum) ** 2)
            C_dist = ((C_convolve_inv - 0) ** 2 + (C_convolve - c_sum) ** 2)
            D_dist = ((D_convolve_inv - 0) ** 2 + (D_convolve - d_sum) ** 2)

            A_dist_min = A_dist.min()
            B_dist_min = B_dist.min()
            C_dist_min = C_dist.min()
            D_dist_min = D_dist.min()

            df2 = pd.DataFrame({'label' : [label],
                                'min_A' : [A_dist_min],
                                'min_B' : [B_dist_min],
                                'min_C' : [C_dist_min],
                                'min_D' : [D_dist_min],
                                'path' : [path]})

            df = pd.concat([df, df2],
                           ignore_index=False,
                           axis=0)

    df.to_csv('crops_minima_data.csv')

    return df


def find_answer_using_minima_df(csv_path):
    """
    Using the path that the csv was made with all the minima in there with the assigned labels, this function shows that
    a random forest is well able to predict the minima using these minima. For this, training and testing data is split.
    This is done using a Decision Tree Classifier, Random Forest Classifier and Logistic Regression.
    :param csv_path: path of csv containing the information from the dataframe with all minima and the label.
    :return:
    """

    # open the dataframe containing the minimum values and the correct label
    df = pd.read_csv(csv_path)

    # do repeated K-fold cross-validation
    X = df[['min_A', 'min_B', 'min_C', 'min_D']]
    y = df['label']
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # save the correct values and predictions in two lists
    y_correct = pd.Series()
    y_pred_rf = pd.Series()
    y_pred_dt = pd.Series()
    y_pred_lr = pd.Series()

    # iterate over the splits of training data
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # add correct answer to correct answer list
        y_correct = pd.concat([y_correct, pd.Series(y_test)])

        # normalise in each run using training data
        normalizer = Normalizer().fit(X_train)

        X_train_norm = normalizer.transform(X_train)
        X_test_norm = normalizer.transform(X_test)

        # Random Forest Classifier
        rf = RandomForestClassifier()  # random_state=42)
        rf.fit(X_train_norm, y_train)

        pred_rf_test = rf.predict(X_test_norm)

        # save the random forest prediction
        y_pred_rf = pd.concat([y_pred_rf, pd.Series(pred_rf_test)])

        # decision tree
        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(X_train_norm, y_train)

        pred_dt_test = dt.predict(X_test_norm)

        # save the decision tree prediction
        y_pred_dt = pd.concat([y_pred_dt, pd.Series(pred_dt_test)])

        # logistic regression without interaction terms
        logreg = LogisticRegression()
        logreg.fit(X_train_norm, y_train)

        pred_lr_test = logreg.predict(X_test_norm)

        # save the logistic regression prediction
        y_pred_lr = pd.concat([y_pred_lr, pd.Series(pred_lr_test)])

    # test performance decision tree
    print('Decision tree')
    print(confusion_matrix(y_true=y_correct, y_pred=y_pred_dt))
    print(classification_report(y_true=y_correct, y_pred=y_pred_dt, digits=4))

    # test performance random forest
    print('Random forest')
    print(confusion_matrix(y_true=y_correct, y_pred=y_pred_rf))
    print(classification_report(y_true=y_correct, y_pred=y_pred_rf, digits=4))

    # test performance logistic regression
    print('Logistic regression')
    print(confusion_matrix(y_true=y_correct, y_pred=y_pred_lr))
    print(classification_report(y_true=y_correct, y_pred=y_pred_lr, digits=4))

    plot_tree(dt, feature_names=['min_A', 'min_B', 'min_C', 'min_D'], class_names=['A', 'B', 'C', 'D'],
              filled=True, impurity=False, proportion=True, rounded=True, label='none')

    # if uncommented, save last decision tree
    # plt.savefig('decision_tree.png')


def check_labeled_images(img_dirs):
    """
    For all multiple choice questions not among the YOLO training data, classify the image based on whether the image
    was labeled correctly. So YOLO shows where the handwritten text was found and classify how it went by pressing keys.
    c: correct, w: wrong, m: correctly missing, i: incorrectly missing. Prints the counts of the categories.
    :param img_dirs: directory containing all tests
    :return:
    """

    # initiate counts
    correct_count = 0
    wrong_count = 0
    correct_missing_count = 0
    incorrect_missing_count = 0

    # all directories
    dirs = [join(img_dirs, f, 'Improved questions 130') for f in listdir(img_dirs) if (isdir(join(img_dirs, f)) and
                                                                                       f not in ['toets0',
                                                                                                 'toets1',
                                                                                                 'toets2'])]

    for directory in dirs:
        mc_file_list = [join(directory, f) for f in listdir(directory) if
                        (isfile(join(directory, f)) and int(f.split('.')[0]) > 20)]

        for path in mc_file_list:

            # run YOLO
            img = run(weights=Path(r"yolov5\runs\train\exp45\weights\best_new.pt"),
                      imgsz=[640, 640],
                      max_det=1,
                      # conf_thres=0.3,
                      # is_image_list=True,
                      # question_nums=question_list,
                      iou_thres=0.1,  # threshold to avoid overlapping images
                      data=Path(r"C:\Users\ajtis\Documents\Master\Thesis\pythonProject\yolov5\models\cito.yaml"),
                      source=Path(path),  # Path(file_list),
                      nosave=True,  # do/don't save images
                      save_crop=False,  # do/don't save the cropped images
                      view_img=False,
                      view_crop=False,
                      return_image=True,
                      # return_coordlist=True,
                      classes=1)  # only do the handwritten


            window_height = 500

            height = img.shape[0]
            width = img.shape[1]
            ratio = height / float(width)

            # set the dimensions
            dim = (window_height, int(window_height * ratio))

            img = cv2.resize(img, dim)

            # show the image including label
            cv2.imshow('Select Correct, Wrong (wrong labeled) or Correctly missing', img)

            key = cv2.waitKey()
            #print(key)

            # use the keys corresponding to the letters
            if key == 99: # C(orrect)

                correct_count += 1

            if key == 119: # W(rong)

                wrong_count += 1

            elif key == 109: # M(issing), correctly

                correct_missing_count += 1

            elif key == 105: # I(ncorrect missing)

                incorrect_missing_count += 1

            print('Correct count:', correct_count)
            print('Incorrect count:', wrong_count)
            print('Correctly missing count:', correct_missing_count)
            print('Incorrectly missing count:', incorrect_missing_count)

    # print the final count
    print('Correct count:', correct_count)
    print('Incorrect count:', wrong_count)
    print('Correctly missing count:', correct_missing_count)
    print('Incorrectly missing count:', incorrect_missing_count)


def main():
    find_answer_using_minima_df('crops_minima_data.csv')

if __name__ == '__main__':
    main()
