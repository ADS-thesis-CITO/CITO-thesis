# Scripts Folder

This folder contains all python scripts which were executed to retrieve the output. 
The folder is split into three subfolders, each representing a different part of the process. 

## Preprocessing
The scripts in this folder are concerned with retrieving and cleaning the data before classification. 
This folder should contain the following scripts: 
* 'extract_all_numbers.py' - A script to extract and save all individual numbers from given answers.
* 'create_grids.py' - A script which creates a grid visualisation of (un)identifiable numbers.

## Modelling 
The scripts in this folder create all models used for classification. 
All models are constructed using the functional keras API [ADD source].

The following models are created: 

### The F_MNIST model 
The Functional MNIST model (hereafter mentioned as: F_MNIST) is trained on MNIST number data [ADD source]. 

Descriptives:
[ADD descriptives about the MNIST data]

### The F_MS model
The Functional Math Symbols model (hereafter mentioned as: F_MS) is trained on the Math symbols dataset [ADD source]. 

Descriptives: 
[ADD descriptives about the Math Symbols dataset]

### The FA_MS model
The Functional Augmented Math Symbols model (hereafter mentioned as: FA_MS) is trained on the Math symbols dataset [ADD source]. Data augmentation was used to enlarge the dataset by copying numbers and using random rotation with the goal to improve model performance on 'real life' data. 

Descriptives: 
[ADD descriptives about the Augmented Math Symbols dataset]

## Classification
The scripts in this folder are concerned with classifying the processed data. 
The folder should contain: 
* 'single_test_classification.py' - A script which classifies all open answers in a test and grades it accordingly. 