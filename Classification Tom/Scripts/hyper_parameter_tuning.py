'''
:Author:        Tom Klopper. 
:Goal:          Hyper Parameter Tuning of the CITO data model. 
:Last update:   01-07-2022.
:note: Print statements were deleted to improve readability of the script. 
'''
#%%
# Import Libraries. 
from cgi import test
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model 

from sklearn.model_selection import train_test_split

from keras import layers 
from keras import models 
from numpy import expand_dims, ndarray
from keras import callbacks

#%%
# Get files from directory. 
directory = "C://Users//tklop//ThesisADS//CharacterImages//40x40"
# Concat the data. 
Name=[]
df2 = pd.read_csv("C://Users//tklop//ThesisADS//labeling_by_hand.csv", sep = ';')
df1 = pd.read_csv("C://Users//tklop//ThesisADS//custom_label_v1.csv", sep = ',')
df = pd.concat([df1, df2], ignore_index= True)

# clean the data. 
crop_name_list = []
for idx, student_code in enumerate(df['Student']): 
    crop_name = f"{student_code}_{df.loc[idx, 'Question']}_{df.loc[idx, 'Subdigit']}"
    crop_name_list.append(crop_name)

df['Crop'] = crop_name_list
df = df.dropna(subset = ['Expected']).reset_index(drop = True)
df = df[df['Expected'] < 10]
df.reset_index()

Name = df['Crop']

# Define variables. 
dataset=[]
testset=[]
count_jpg = 0
count_png = 0
count = 0

# Create and augment the data. 
for idx, name in enumerate(Name):
    im = name + ".jpg"
    try: 
        img = load_img(os.path.join(directory,im), color_mode='grayscale', target_size=(40,40,1))
        img = img_to_array(img)
        img = img/255.0
        imageNew = expand_dims(img, 0)
        #imageNew = imageNew.reshape(40,40,1)
        try: 
            img_label = df['Expected'].loc[df['Crop'] == name ].item()
        except ValueError: 
            continue
        imageDataGen = ImageDataGenerator(rotation_range=20, width_shift_range=0.20, height_shift_range=0.20)
        iterator = imageDataGen.flow(imageNew, batch_size=1)
        for i in range(9):
            batch = iterator.next()
            image = batch[0].astype('float32')
            dataset.append([image, img_label])
            count_jpg += 1
        
    except FileNotFoundError: 
        continue


#%%
data,labels1 =zip(*dataset)     
data=np.array(data)
labels=to_categorical(labels1)

# Train val test split. 
train_images, test_images,train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state=44)
train_images, val_images,train_labels, val_labels = train_test_split(train_images,train_labels,test_size=0.2,random_state=44)

#%%
###################################
# LAYER DEPTH. 
#-------------------------
# define parameters. 
image_size = 40
input_shape = (image_size, image_size, 1)

n_layers = 5
model = [0] * n_layers

for j in range(5):
    model[j] = models.Sequential()
    model[j].add(layers.Conv2D(32,kernel_size=5,padding='same',activation='relu', input_shape=input_shape))

    model[j].add(layers.MaxPool2D())
    if j>0:
        model[j].add(layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(layers.MaxPool2D(padding='same'))
    if j>1:
        model[j].add(layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(layers.MaxPool2D(padding='same'))
    if j>2:
        model[j].add(layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(layers.MaxPool2D(padding='same'))
    if j>3:
        model[j].add(layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(layers.MaxPool2D(padding='same'))
    if j>4:
        model[j].add(layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
        model[j].add(layers.MaxPool2D(padding='same'))
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(256, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#%%
callbacks_list = [
    callbacks.EarlyStopping(
        monitor = 'accuracy',
        patience = 2,
    ),
    callbacks.ModelCheckpoint(
        filepath= f"callback_own_model.h5", 
        monitor = 'val_loss',
        save_best_only= True,
    ), 
    callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', 
        factor = 0.1, 
        patience = 4
    )
]
#%%
history = [0] * n_layers
names = ["1-layers","2-layers","3-layers", "4-layers", "5-layers"]
epochs = 50
for j in range(n_layers):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)

# %%
#########################################
# KERNEL SIZE 
#---------------------------
n_kernels = 3
kernel_list = [2,3,5]
model = [0] * n_kernels
history = [0] * n_kernels

for j in range(n_kernels):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(32,kernel_size=kernel_list[j],padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    # Layer 2 
    model[j].add(layers.Conv2D(64,kernel_size=kernel_list[j],padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 3
    model[j].add(layers.Conv2D(64,kernel_size=kernel_list[j],padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 4
    model[j].add(layers.Conv2D(64,kernel_size=kernel_list[j],padding='same',activation='relu'))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(256, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_kernels):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)


# %%
########################################
# OPTIMIZER 
#-----------------------------------

n_optimizers = 4
history = [0] * n_optimizers
model = [0] * n_optimizers
names = ["SGD", "Adam", "RMSprop", "Nadam"]
optimizers = ["sgd", "adam", "rmsprop", "nadam"]

for j in range(n_optimizers):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    # Layer 2 
    model[j].add(layers.Conv2D(64,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 3
    model[j].add(layers.Conv2D(64,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 4
    model[j].add(layers.Conv2D(64,kernel_size= 5 ,padding='same',activation='relu'))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(256, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer= optimizers[j], loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_optimizers):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)




# %%
############################################
# FEATURE MAPS 
# -----------------------------
n_feature_maps = 4
history = [0] * n_feature_maps
model = [0] * n_feature_maps
names = ["16", "32", "48", "64"]
feature_maps = [1, 2 ,3 ,4]

for j in range(n_feature_maps):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(j * 16 + 16,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    # Layer 2 
    model[j].add(layers.Conv2D((j * 16 + 16)*2 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 3
    model[j].add(layers.Conv2D((j * 16 + 16)*2 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 4
    model[j].add(layers.Conv2D((j * 16 + 16)*2 ,kernel_size= 5 ,padding='same',activation='relu'))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(256, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_feature_maps):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)



# %%
#########################################
# DENSE LAYER SIZE 
# ---------------------------------------
n_layers = 5
history = [0] * n_layers
model = [0] * n_layers
names = ["64", "128", "256", "512", "1024"]
layervalues = [64, 128, 256 , 512 , 1024]

for j in range(n_layers):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    # Layer 2 
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 3
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    # Layer 4
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(layervalues[j], activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_layers):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)


# %%
#######################################
# DROPOUT LAYERS
# ----------------------
n_dropout = 6
history = [0] * n_dropout
model = [0] * n_dropout
names = ["Dropout = 0.0","Dropout = 0.1","Dropout = 0.2","Dropout = 0.3","Dropout = 0.4","Dropout = 0.5"]
dropoutvalues = [0,1,2,3,4,5]

for j in range(n_dropout):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    model[j].add(layers.Dropout(j*0.1))
    # Layer 2 
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    model[j].add(layers.Dropout(j*0.1))
    # Layer 3
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    model[j].add(layers.Dropout(j*0.1))
    # Layer 4
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    model[j].add(layers.Dropout(j*0.1))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(128, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_dropout):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)


#%%
##############################################
# BATCH NORMALIZATION. 
# -----------------------------------
n_batch = 2
history = [0] * n_batch
model = [0] * n_batch
names = ["Without batch normalization", "With batch normalization"]
dropoutvalues = [0,1,2,3,4,5]

for j in range(n_batch):
    model[j] = models.Sequential()
    # Layer 1 
    model[j].add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
    model[j].add(layers.MaxPool2D())
    if j > 0: 
        model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.2))
    # Layer 2 
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    if j > 0: 
        model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.2))
    # Layer 3
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    if j > 0: 
        model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.2))
    # Layer 4
    model[j].add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
    model[j].add(layers.MaxPool2D(padding='same'))
    if j > 0: 
        model[j].add(layers.BatchNormalization())
    model[j].add(layers.Dropout(0.2))

    #Dense layers
    model[j].add(layers.Flatten())
    model[j].add(layers.Dense(128, activation='relu'))
    model[j].add(layers.Dense(10, activation='softmax'))
    model[j].compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 50
for j in range(n_batch):
    history[j] = model[j].fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)


#%%
#######################################
# FINAL MODEL. 
#--------------------------
history = [0] 
model = [0] 

model = models.Sequential()
# Layer 1 
model.add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= input_shape))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))
# Layer 2 
model.add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
model.add(layers.MaxPool2D(padding='same'))
model.add(layers.Dropout(0.2))
# Layer 3
model.add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
model.add(layers.MaxPool2D(padding='same'))
model.add(layers.Dropout(0.2))
# Layer 4
model.add(layers.Conv2D(64 ,kernel_size= 5 ,padding='same',activation='relu'))
model.add(layers.MaxPool2D(padding='same'))
model.add(layers.Dropout(0.2))

#Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_images,train_labels, batch_size=64, epochs = epochs, validation_data = (val_images,val_labels), callbacks= callbacks_list, verbose=1)



