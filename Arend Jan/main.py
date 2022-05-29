import numpy as np
import pandas as pd

import cv2

import matplotlib.pyplot as plt

import re

from os import listdir
from os.path import isfile,join, isdir

import fitz
from PIL import Image

import pytesseract

# indicate tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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


def recognize_nums(img, psm=7, threshold = 0.5):
    """
    Recognizes numbers using Tesseract.
    :param img: cv2 image to recognise numbers in
    :param psm: int that represents the internal settings of Tesseract
    :param threshold: float that represents how certain Tesseract should be of a number, higher is more certain
    :return: number recognised
    """

    config = r'-c tessedit_char_whitelist=0123456789 --psm '+str(psm)

    if threshold == 0.5:
        text = pytesseract.image_to_string(img, config=config)
    else:
        text = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

        # convert confidence to float
        conf = [float(item) for item in text['conf']]

        # first item's index that equals the maximum
        max_conf_index = [index for index, item in enumerate(text['text']) if conf[index]==max(conf)][0]

        if conf[max_conf_index] < (1-threshold)*100:
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


def add_borders(img, width=4134, height=5847):
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
                                      height-img_height,
                                      0,
                                      width-img_width,
                                      cv2.BORDER_CONSTANT,
                                      None,
                                      (255,255,255))

    border_image = cv2.cvtColor(border_image,
                                cv2.COLOR_RGB2BGR)

    return border_image

def extract_exercise_img(image):
    """
    Take an image and extract the exercises on that page with their exercise numbers.
    :param image: cv2 BGR image
    :return: list of question images, list of question numbers
    """
    # input the cv2 BGR image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # extract only the part with the page numbers
    height = gray_image.shape[0]
    width = gray_image.shape[1]
    gray_image = gray_image[:, :(int(width * .17))]

    # thresholding
    ret2, gray_image = cv2.threshold(gray_image,
                                     100,
                                     255,
                                     cv2.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)

    gray_copy = gray_image.copy()

    # eroding the image
    gray_image = cv2.erode(gray_image, kernel, iterations=5)

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

        question_num_img = gray_copy[y:y + h, x:x + w]
        # show_image(question_num_img)

        # set a threshold to avoid the misclassification of numbers
        question_num = recognize_nums(question_num_img, threshold=0.8)

        # set unknown if it is not known else make int
        question_num = "unknown" if question_num == "" else int(question_num)

        # add to array
        if question_num != "unknown":
            num_coords.append((x, y, w, h))
            question_nums.append(question_num)

            #show_image(question_num_img)

    # reverse the list to go from top to bottom
    num_coords = num_coords[::-1]
    question_nums = question_nums[::-1]

    # store the images in a list
    image_list = []

    if len(num_coords) == 1:

        image_list = [image]

    elif len(num_coords) == 0:

        image_list = []

    else:
        for index, coords in enumerate(num_coords):
            # trim original image and add to array

            # skip first, because we care about the differences
            if index == 0:
                # store coordinates to compare with next
                previous_coords = coords
                continue

            sliced_image = image[(previous_coords[1] - 150):(coords[1] - 150), :]

            image_list.append(sliced_image)

            # at the last step, also do the same to the bottom of the page
            if index == len(num_coords) - 1:
                sliced_image = image[(coords[1] - 150):, :]

                image_list.append(sliced_image)

            # store coordinates to compare with next
            previous_coords = coords

    for index, img in enumerate(image_list):
        # add borders
        border_image = add_borders(img)

        # make grayscale
        image_list[index] = cv2.cvtColor(border_image,
                                         cv2.COLOR_BGR2GRAY)

        # io.imsave(output_folder + str(question_nums[index]) + '.png', gray_image)
        #print(question_nums[index])
        #show_image(gray_image)


    return image_list, question_nums


def extract_exercise(filename):
    """
    Extract list of exercise images and question numbers based on the path
    :param filename: string, path of the file
    :return: list of question images, list of question numbers
    """

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_list, question_nums = extract_exercise_img(image)

    return image_list, question_nums


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

        if item.lower() == 'school':
            school_num = text_list[index+1]

        if item.lower() == 'nummer':
            student_num = text_list[index+1]

    return school_num, student_num


def question_list(img_dir):
    """
    Using the directory with all images of one test, returns all information about the test taken.
    This is the list of images of all test questions, all question numbers, the school number and the student number.
    :param img_dir: string, directory where all images are located
    :return: list of test question images, list of question numbers, school number and student number
    """

    files = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for index, f in enumerate(listdir(img_dir)):

        # use information of first page
        if f == '0.jpg':
            first_page_path = join(img_dir, f)
            first_page = cv2.imread(first_page_path)

            #show_image(first_page)

            school_num, student_num = extract_first_page(first_page)

        # remove irrelevant pages
        if f in ['0.jpg', '1.jpg', '2.jpg', '10.jpg', '15.jpg']:
            files.remove(join(img_dir, f))

    list_image_lists = []
    list_question_nums = []

    for file in files:
        # convert to cv2 first

        image_list, question_nums = extract_exercise(file)

        list_image_lists.append(image_list)
        list_question_nums.append(question_nums)

    # flatten these lists to obtain lists of results
    image_list = [image for sublist in list_image_lists for image in sublist]
    question_nums = [question_num for sublist in list_question_nums for question_num in sublist]

    #print(question_nums)
    #for image in image_list:
    #    show_image(image)

    # sort them by the question numbers
    image_list = [image for question, image in sorted(zip(question_nums, image_list), key=lambda x: x[0])]
    question_nums = sorted(question_nums)

    return image_list, question_nums, school_num, student_num


def check_questions(question_nums, image_list):
    """
    Check whether there are any clear errors among the question numbers.
    In case of 30 test questions for example, check whether question numbers 1 to 30 are present.
    Makes lists with questions to revise, removes question numbers and images from original list if invalid.
    Indicates whether a manual inspection is necessary.
    :param question_nums: list of question numbers
    :param image_list: list of question images
    :return: list of questions to revise, list of images to revise, questions left, images left, manual inspection needed
    """

    question_nums_to_remove = []

    # manual inspection needed
    manual_inspection = 'No'

    # check whether all numbers from 1 to the total length occur only once, go backwards
    for i in range(1, len(question_nums)+1):
        count = question_nums.count(i)

        if count > 1:
            # if count is larger than 1, add it to the list of numbers to remove
            # counts of 0 are also incorrect, but we cannot remove those
            question_nums_to_remove.append(i)
            manual_inspection = 'Yes'

    questions_revise = []
    images_revise = []

    # remove the question numbers and add them to the revise list, same for the images
    # reverse the enumerate object to avoid issues when removing numbers from the list
    for index, num in reversed(list(enumerate(question_nums))):

        if num in question_nums_to_remove:
            # no need to remove all, because it will iterate over those anyway
            questions_revise.append(num)
            images_revise.append(image_list[index])

            question_nums.pop(index)
            image_list.pop(index)

    return questions_revise, images_revise, question_nums, image_list, manual_inspection


def make_df(directories):
    """
    Makes the dataframe with all the information of the tests
    :param directories: list of all test directories containing the jpg's of every test.
    :return: dataframe
    """

    # for checking whether it is a directory
    dirs = [join(directories, f) for f in listdir(directories) if isdir(join(directories, f))]

    # make dataframe for all info

    df = pd.DataFrame(columns=['School number', 'Student number', 'Question numbers',
                               'Image list', 'Question numbers to revise', 'Images to revise',
                                'Manual inspection'])

    for index, test_dir in enumerate(dirs):

        print(index)

        image_list, question_nums, school_num, student_num = question_list(test_dir)

        # Check whether there are issues with the images and question numbers
        questions_revise, images_revise, question_nums, image_list, manual_inspection = check_questions(question_nums, image_list)

        df2 = pd.DataFrame({'School number': school_num,
                            'Student number': student_num,
                            'Question numbers': [np.array(question_nums)],
                            'Image list': [np.array(image_list)],
                            'Question numbers to revise':[np.array(questions_revise)],
                            'Images to revise':[np.array(images_revise)],
                            'Manual inspection':manual_inspection})

        df = pd.concat([df,df2],
                  ignore_index=True,
                  axis=0)


    return df


def main():

    folders_dir = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Toetsen pdf's"

    df = make_df(folders_dir)

    print(df['School number'])
    print(df['Student number'])
    print(df['Question numbers'])
    print(df['Question numbers to revise'])

    for images in df['Images to revise']:
        for image in images:
            show_image(image)

    for text in df['Manual inspection']:
        print(text)



if __name__ == '__main__':
    main()

