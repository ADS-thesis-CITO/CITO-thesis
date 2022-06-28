import fitz
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from pdf2image import convert_from_path

import os

def save_images(path, out_folder):
    images = convert_from_path(path, dpi=600)

    for index, page in enumerate(images):
        #out_path = f"C:/Users/ajtis/Documents/Master/Thesis/Toetsen/images_pdf/out_{str(index)}.jpg"

        page.save(os.path.join(out_folder, str(index)+'.jpg'), 'JPEG')

    return images

def return_images(path):

    pages = fitz.open(path)

    # delete odd pages and last few pages
    #del pages[range(1, len(pages), 2)]
    #del pages[0, 17, 18, 19]

    img_list = []

    for page in pages:

        zoom = 25
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode,
                              [pix.width, pix.height],
                              pix.samples)

        img_list.append(img)
    return img_list


def show_imglist(img_list):

    for img in img_list:
        plt.imshow(img)
        plt.show()

def main():
    #path = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\0363_001.pdf"

    #path2 = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Cito toets ingevuld.pdf"

    parent_path = r"C:\Users\ajtis\Documents\Master\Thesis\Toetsen\Hogere resolutie toetsen\ToetsData"

    for i in range(73):

        print(i)

        out_folder = os.path.join(parent_path, 'toets' + str(i))

        os.mkdir(out_folder)

        path = f"C:/Users/ajtis/Documents/Master/Thesis/Toetsen/Hogere resolutie toetsen/ToetsData/Toetsgroep 5  - 600dpi ({str(i+1)}).pdf"

        save_images(path, out_folder)




    img_list = return_images(path)

    show_imglist(img_list)

if __name__ == '__main__':
    main()

