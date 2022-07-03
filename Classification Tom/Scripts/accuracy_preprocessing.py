'''
:Author:        Tom Klopper. 
:Goal:          To analyse the succesfullness of preprocessing the data. 
:Last updated:  03-07-2022. 
'''
#%%
# Import libraries.
import matplotlib
import pandas as pd
import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# Define functions. 
def CountQuestion(question_name):
    '''
    :Goal:                  Count the number of cutouts from each given answer. 
    :Param question_name:   String value representing the name of each question e.g. 'exp1_14_'
    :Returns:               Integer value which represents the count of a question in the folder. 
    '''
    count = 0 
    for img_name in img_names:
        if question_name in img_name: 
            count +=1  
    return count

def CreateTestList(question_count_keys): 
    '''
    :Goal:                          To create a list with string values unique for every test. 
    :Param question_count_keys:     list of all image names.
    :Returns:                       List of all unique test names.  
    '''
    test_list = []
    for question_key in question_count_keys:
        # Split '.jpg' from question name. 
        split_key = question_key.split('_')
        test_name = f"{split_key[0]}_"
        if test_name in test_list: 
            continue
        else: 
            test_list.append(test_name)
    return test_list

# Define all thresholds. 
threshold_list = [200,205,210,215,220,225,230,235,240,245]

# Create empty dataframe. 
final_df = pd.DataFrame()

for threshold_value in threshold_list : 
    # Define variables. 
    question_count_keys = []
    count_list = []
    n_questions = 20
    n_tests = 73

    # Define path. 
    mypath = f"C://Users//tklop//ThesisADS//CharacterImages//threshold_{threshold_value}//40x40"
    img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Create dictionary which counts crop-outs per question. 
    for test_number in range(1, n_tests + 1): 
        for question_number in range(1, n_questions + 1): 
            # Create question name. 
            question_name = f"exp{test_number}_{question_number}_"
            name_count = CountQuestion(question_name)
            question_count_keys.append(question_name)
            count_list.append(name_count)

    # Create dictionairy. 
    question_count_dict = dict(zip(question_count_keys, count_list))


    # Dictionairy with the correct answers to the test. 
    answer_dict = {"1": 56, "2": 27, "3": 48, "4": 37, "5": 228,
                    "6": 647, "7": 383, "8": 853, "9": 660, "10": 530, 
                    "11": 150, "12": 870, "13": 84, "14": 631, "15": 93,
                    "16": 384, "17": 144, "18": 411, "19": 255, "20": 142 }

    # Create list of unique test names. 
    test_list = CreateTestList(question_count_keys)

    # Iterate through all test files. 
    test_path =  "C://Users//tklop//ThesisADS//AllAnswers"
    test_folder = [f"exp{f}" for f in range(1,74)]

    # Create empty list. 
    expected_dictionaries =[]
    recognition_mistake_list = []
    n_per_folder = []
    character_per_test = []

    # Run through all test names. 
    for test_name in test_folder:
    
        new_path = f"{test_path}//{test_name}//crops//handwritten"
        
        # Find the names of all images within one test folder. 
        img_names = [f for f in listdir(new_path) if isfile(join(new_path, f))]
        n_per_folder.append(len(img_names))
        image_names = img_names.copy()
        recognition_mistake_count = 0
        temp_expected_count = [] 
        
        # Get all file names by splitting of '.jpg' 
        for idx, image_name in enumerate(image_names):
            img_name_split = image_name.split('.')
            img_names[idx] = img_name_split[0]

        temp_key_list = []
        for q_number in range(1,21): 
            temp_key_list.append(str(q_number))
            if str(q_number) in img_names: # Expected problem every string is in. 
                temp_expected_count.append(len(str(answer_dict[str(q_number)])))
            else: 
                recognition_mistake_count += 1
                temp_expected_count.append(0)
        temp_dict = dict(zip(temp_key_list, temp_expected_count))
        expected_dictionaries.append(temp_dict)
        recognition_mistake_list.append(recognition_mistake_count)  

    negative_crops_per_test = []
    positive_crops_per_test = []
    n_mistakes_per_test = []

    for test_idx, test in enumerate(test_list): 
        negative_list = []
        positive_list = []
        temp_mistakes = 0 

        for question_number in range(1, n_questions + 1): 
            q_name = f"{test}{question_number}_" # e.g. q_name = exp1_1_
            #true_answer = answer_dict[str(question_number)]
            true_answer = expected_dictionaries[test_idx][str(question_number)]
            #print(f"{test}_{question_number}: {true_answer} ")

            if true_answer == question_count_dict[q_name]: 
                continue
            else:
                crop_diff = question_count_dict[q_name] - true_answer
                if crop_diff > 0: 
                    positive_list.append(crop_diff)
                elif crop_diff < 0:
                    negative_list.append(crop_diff)
                temp_mistakes += 1
        negative_crops_per_test.append(sum(negative_list))
        positive_crops_per_test.append(sum(positive_list))
        n_mistakes_per_test.append(temp_mistakes)

    # Print results (hidden for readability).
    #print(positive_crops_per_test)
    #print(negative_crops_per_test)
    #print("\n")
    average_mistakes = np.average(n_mistakes_per_test)
    #print(n_mistakes_per_test)
    #print(average_mistakes)

    # Create Dataframe. 
    df = pd.DataFrame(columns= ['Test_name','Negative_crop_count', 'Positive_crop_count', 'Error_rate'])
    df['Test_name'] = test_list
    df['Negative_crop_count'] = negative_crops_per_test
    df['Positive_crop_count'] = positive_crops_per_test
    df['Number_of_found_questions'] = n_per_folder
    df['Number_of_mistakes'] = n_mistakes_per_test
    df['Error_rate'] = (df['Number_of_mistakes']  / df['Number_of_found_questions']) *100

    #print(df.head(10))
        
    error_rate = np.nanmean(df['Error_rate'])
    positive_count = np.nanmean(df['Positive_crop_count'])
    negative_count = np.nanmean(df['Negative_crop_count'])

    key_list = ['Threshold_value', 'Negative_count', 'Positive_count', 'Error_rate']
    value_list = [threshold_value, negative_count, positive_count, error_rate]
    appending_dict = dict(zip(key_list, value_list))
    final_df = final_df.append(appending_dict, ignore_index = True)
    
    

# Print result of final_df dataframe. 
print(final_df.head(10))

# Plot error visualisation across different thresholds. 
plt.plot(final_df['Threshold_value'], final_df['Error_rate'], color = "black")
#plt.title("Error rate for different threshold values")
plt.xlabel("Threshold value")
plt.ylabel("Error rate (in %)")
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.savefig("optimal_threshold.jpg")
plt.show()

# Add negative and positive averages to the threshold plot. 
# ----------------------------------------------------------

labels = [200,205,210,215,220,225,230,235,240,245]

# Define bar plots. 
positive_means = list(final_df['Positive_count'])
negative_means = list(final_df['Negative_count'])
negative_means_conv = [-x for x in negative_means]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, positive_means, width, label='Positive', color = '#a9e3a6' )
rects2 = ax.bar(x + width/2, negative_means_conv, width, label='Negative', color = '#e3a9a6')

# Set grid and plot layout. 
ax.set_ylabel('Average amount of deviation')
ax.set_xlabel('Threshold values')
ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax.set_xticks(x, labels)
ax.set_axisbelow(True)
ax.legend()

# Add other plot with same x axis but a different y. 
ax2 = plt.twinx()
ax2.plot(x, list(final_df['Error_rate']), color = "black", marker='o', linewidth = 1.5)
ax2.set_ylabel('Error rate (in %)')
fig.tight_layout()

# Save and show figure. 
plt.savefig("full_preprocessing_analysis.jpg")
plt.show()
#%%
