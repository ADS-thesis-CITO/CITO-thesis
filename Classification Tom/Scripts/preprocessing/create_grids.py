'''
:Author: Tom Klopper. 
:Goal: To create a grid visualisation of identified and unidentified numbers. 
:Last update: 04-06-2022.
'''

# Import libraries. 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import isfile, join
import cv2


def CreateGrid(mypath, grid_name):
    '''
    :Goal:              To create a grid of 9 images from a given folder. 
    :Param mypath:      String which contains the path to the folder. 
    :Param grid_name:   String which contains the name the grid will be saved as. 
    '''
    # Loop through all images in a folder. 
    img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for idx, img_name in enumerate(img_names): 
        # Load images through using dynamic variable names.  
        globals()[f"im{idx + 1}"] = cv2.imread(mypath + "//" + img_name)

    # Create an image grid. 
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, [im1, im2, im3, im4, im5, im6, im7, im8, im9]):
        # Hide the axis labels and ticks. 
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # Plot in grayscale
        ax.imshow(im, cmap = 'gray')

    #Save figure. 
    plt.savefig(f"{grid_name}.jpg")
    plt.show()


if __name__ == "__main__": 
    # Define variables.
    unid_grid_name = "Unidentified_number_grid"
    id_grid_name = "Identified_number_grid"
    unid_path = "C://Users//tklop//ThesisADS//PlotImages//Unidentified Numbers"
    id_path = "C://Users//tklop//ThesisADS//PlotImages//Identified Numbers"
    # Run function twice. 
    CreateGrid(unid_path, unid_grid_name)
    CreateGrid(id_path, id_grid_name)


