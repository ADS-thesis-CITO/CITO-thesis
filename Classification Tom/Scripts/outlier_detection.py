'''
Author: Tom Klopper
Last updated: 03-07-2022
goal: To create a small script which identifies if numbers in an array are outside the inter quartile range. 
'''

#%%
import numpy as np

#MNIST 
mnist_train = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
mnist_val = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009])

# KAGGLE 
kaggle_train = np.array([3380, 3473, 2499, 3011, 3394, 1925,  3045, 3334, 3000, 2696])
kaggle_val = np.array([841, 910, 597, 715, 800, 496, 780, 851, 780, 670])

# CITO 
cito_train = np.array([1029, 981, 883, 1189, 1092, 689, 912, 853, 971, 405])
cito_val = np.array([256, 224, 217, 311, 268, 181, 228, 217, 244, 105])
cito_test = np.array([62, 61, 60, 78, 56, 42, 65, 45, 63, 31])

#%%

def outlier_detection(data): 
    #calculate interquartile range 

    for idx, arraylist in enumerate(data):
        q3, q1 = np.percentile(arraylist, [75 ,25])
        iqr = q3 - q1
        low_bound = q1 - 1.5*iqr 
        high_bound = q3 + 1.5*iqr
        print(f"========={model_name_list[idx]}=========")
        print(f"Low bound = {low_bound}")
        print(f"High bound = {high_bound}")
        print(f"IQR = {iqr}")
        print(f"1.5*IQR = {1.5*iqr}")
        print(f"q1 = {q1}")
        print(f"q3 = {q3}")

        for value in arraylist:

            if (value < low_bound) or (value > high_bound): 
                print(f"The value {value} is out of range. ")
        


model_list = [mnist_train, mnist_val, kaggle_train, kaggle_val, cito_train, cito_val, cito_test]
model_name_list = ["mnist_train", "mnist_val", "kaggle_train", "kaggle_val", "cito_train", "cito_val", "cito_test"]
#%%
outlier_detection(model_list)
# %%
