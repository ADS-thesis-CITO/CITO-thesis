'''
Author:         Tom Klopper
Last updated:   03-07-2022
Goal:           To analyse all four models on the CITO test data. 
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
from cmath import nan
import cv2 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join
import pandas as pd 
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#%%
# Get files from directory. 
directory = "C://Users//tklop//ThesisADS//CharacterImages//40x40"
Name=[]
df2 = pd.read_csv("C://Users//tklop//ThesisADS//labeling_by_hand.csv", sep = ';')

df1 = pd.read_csv("C://Users//tklop//ThesisADS//custom_label_v1.csv", sep = ',')

df = pd.concat([df1, df2], ignore_index= True)
print(f"Total: {len(df['Expected'])}")
crop_name_list = []
for idx, student_code in enumerate(df['Student']): 
    crop_name = f"{student_code}_{df.loc[idx, 'Question']}_{df.loc[idx, 'Subdigit']}"
    crop_name_list.append(crop_name)

df['Crop'] = crop_name_list

df = df.dropna(subset = ['Expected']).reset_index(drop = True)
print(f"Total after drop na: {len(df['Expected'])}")
df = df[df['Expected'] < 10]
df.reset_index()

print(f"Total after lower than 10: {len(df['Expected'])}")
Name = df['Crop']

#%%
# Define variables. 
dataset=[]
testset=[]
count_jpg = 0
count_png = 0
count = 0


for idx, name in enumerate(Name):
    im = name + ".jpg"
    try: 
        img = load_img(os.path.join(directory,im), color_mode='grayscale', target_size=(40,40,1))
        img = img_to_array(img)
        img = img/255.0
        imageNew = expand_dims(img, 0)
        imageNew = imageNew.reshape(40,40,1)
        try: 
            img_label = df['Expected'].loc[df['Crop'] == name ].item()
        except ValueError: 
            continue
        dataset.append([imageNew, img_label])
        
    except FileNotFoundError: 
        continue


#%%
data,labels1 =zip(*dataset)     

data=np.array(data)
labels=np.array(labels1)

# Train test split. 
train_images, test_images,train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state=44)
test_labels = to_categorical(test_labels)

# Generate the data. 
temp_list = []
for idx, train_image in enumerate(train_images): 
    imageDataGen = ImageDataGenerator(rotation_range=20, width_shift_range=0.20, height_shift_range=0.20)
    temp_image = train_image.reshape(1,40,40,1)
    iterator = imageDataGen.flow(temp_image, batch_size=1)
    img_label = train_labels[idx]

    for i in range(5): # Applies the datagenerator 5 times. 
        batch = iterator.next()
        image = batch[0].astype('float32')
        temp_list.append([image, img_label])


# Create train val split. 
train_images_a1,train_labels_a1 =zip(*temp_list)     
train_images_augmented =np.array(train_images_a1)
train_labels_augmented= to_categorical(train_labels_a1)
train_images, val_images,train_labels, val_labels = train_test_split(train_images_augmented,train_labels_augmented,test_size=0.2,random_state=44)


#%%
####################################################
# MODEL: TUNED CITO MODEL
#---------------------------
history = [0] 
model = [0] 
model_name = "final_tuned_cito"

model = models.Sequential()
# Layer 1 
model.add(layers.Conv2D(32,kernel_size= 5 ,padding='same',activation='relu', input_shape= (40,40,1)))
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

# Create callback list. 
callbacks_list = [
    callbacks.EarlyStopping(
        monitor = 'accuracy',
        patience = 2,
    ),
    callbacks.ModelCheckpoint(
        filepath= f"callback_{model_name}model.h5", 
        monitor = 'val_loss',
        save_best_only= True,
    ), 
    callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', 
        factor = 0.1, 
        patience = 4
    )
]


epochs = 50

history = model.fit(train_images,train_labels, batch_size=64, epochs = epochs, 
    validation_data = (val_images, val_labels), callbacks= callbacks_list, verbose=1)

print("=======EVALUATION=======")
model.evaluate(test_images,test_labels, verbose= 2)
print("========================")
#%%
acc = history.history['accuracy']
loss = history.history[ 'loss']
val_acc = history.history['val_accuracy']
val_loss = history.history[ 'val_loss']


# Visualise model performance. 
# Accuracy: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, acc, label = 'Training accuracy')
plt.plot(epochs, val_acc, label = 'Validation accuracy', linestyle = '--' )

plt.ylabel('Accuracy')
plt.xlabel('Number of epochs') 
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='lower right')
axes = plt.gca()
axes.set_ylim([0.8,1.01])
plt.savefig(f"Accuracy_{model_name}")
plt.show()

# Loss: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, loss, label = 'Training loss')
plt.plot(epochs, val_loss, label = 'Validation loss', linestyle = '--' )

plt.ylabel('Loss')
plt.xlabel('Number of epochs') 
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='upper right')
plt_ax = plt.gca()
plt_ax.set_ylim([-0.01,0.5])
plt.savefig(f"Loss_{model_name}")
plt.show()


#%%
# Create dataframe to save correct and wrong predictions. 
classified_number = pd.DataFrame(columns= ['Predicted', 'Expected', 'Probability'])
pred_list = []
exp_list = []
prob_list = []
for idx, img_value in enumerate(test_images): 
    pred_img = img_value.reshape(1,40,40,1)
    y_prob = model.predict(pred_img)
    prediction = y_prob.argmax(axis=-1)[0]
    probability = np.max(y_prob)

    pred_list.append(prediction)
    exp_list.append(test_labels[idx].argmax(axis= 0))
    prob_list.append(probability)

classified_number['Predicted'] = pred_list
classified_number['Expected'] = exp_list
classified_number['Probability'] = prob_list

wrong_classified = classified_number.loc[~(classified_number['Predicted'] == classified_number['Expected'])]
correct_classified = classified_number.loc[(classified_number['Predicted'] == classified_number['Expected'])]
# %%
# Create probability distributions. 
counts, edges, bars = plt.hist(correct_classified['Probability'], bins= 10, range=[0, 1], facecolor='#aef2b1', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_correct_{model_name}.jpg")
plt.show()

counts, edges, bars = plt.hist(wrong_classified['Probability'], bins= 10, range=[0, 1], facecolor='#eb7b6e', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_wrong_{model_name}.jpg")
plt.show()

# %%
# Create classification report 
print(classification_report(classified_number['Expected'], classified_number['Predicted'], digits= 4))

# Create Confusion matrix. 
cf_matrix = confusion_matrix(classified_number['Expected'], classified_number['Predicted'] )
print(cf_matrix)

#%%
######################################################
#MODEL: CHOLLET CITO
#-------------------------
image_size = 40
input_shape = (image_size, image_size, 1)
model_name = "final_chollet_cito"
# Create model architecture. 
inputs = layers.Input(shape= input_shape)
x = layers.Conv2D(32, (3,3), activation = 'relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation = 'relu')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)

# Initiate model. 
model = models.Model(inputs, outputs)
model.summary()

# Compile the model. 
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy', 
              metrics  = ['accuracy'])
# Callback. 
callbacks_list = [
    callbacks.EarlyStopping(
        monitor = 'accuracy',
        patience = 2,
    ),
    callbacks.ModelCheckpoint(
        filepath= f"callback_{model_name}model.h5", 
        monitor = 'val_loss',
        save_best_only= True,
    ), 
    callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', 
        factor = 0.1, 
        patience = 4
    )
]

# Fit the model. 
history = model.fit(train_images, train_labels, 
                    epochs = 100, 
                    batch_size = 64, 
                    callbacks = callbacks_list,
                    validation_data= (val_images, val_labels))

#%%
# Create vars for visualisation
acc = history.history['accuracy']
loss = history.history[ 'loss']
val_acc = history.history['val_accuracy']
val_loss = history.history[ 'val_loss']

# Accuracy: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, acc, label = 'Training accuracy')
plt.plot(epochs, val_acc, label = 'Validation accuracy', linestyle = '--' )

plt.ylabel('Accuracy')
plt.xlabel('Number of epochs') 
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='lower right')
plt_ax = plt.gca()
plt_ax.set_ylim([0.8,1.01])
plt.savefig(f"Accuracy_{model_name}")
plt.show()

# Loss: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, loss, label = 'Training loss')
plt.plot(epochs, val_loss, label = 'Validation loss', linestyle = '--' )

plt.ylabel('Loss')
plt.xlabel('Number of epochs') 
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(loc='upper right')
plt_ax = plt.gca()
plt_ax.set_ylim([-0.01,0.7])
plt.savefig(f"Loss_{model_name}")
plt.show()


#%%
# Create datframe to check correct or wrong classifications
classified_number = pd.DataFrame(columns= ['Predicted', 'Expected', 'Probability'])
pred_list = []
exp_list = []
prob_list = []
for idx, img_value in enumerate(test_images): 
    pred_img = img_value.reshape(1,40,40,1)
    y_prob = model.predict(pred_img)
    prediction = y_prob.argmax(axis=-1)[0]
    probability = np.max(y_prob)

    pred_list.append(prediction)
    exp_list.append(test_labels[idx].argmax(axis= 0))
    prob_list.append(probability)

classified_number['Predicted'] = pred_list
classified_number['Expected'] = exp_list
classified_number['Probability'] = prob_list

wrong_classified = classified_number.loc[~(classified_number['Predicted'] == classified_number['Expected'])]
correct_classified = classified_number.loc[(classified_number['Predicted'] == classified_number['Expected'])]
# %%
# Visualise probability distributions. 
counts, edges, bars = plt.hist(correct_classified['Probability'], bins= 10, range=[0, 1], facecolor='#aef2b1', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_correct_{model_name}.jpg")
plt.show()

counts, edges, bars = plt.hist(wrong_classified['Probability'], bins= 10, range=[0, 1], facecolor='#eb7b6e', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_wrong_{model_name}.jpg")
plt.show()

# %%
# Classification report
print(classification_report(classified_number['Expected'], classified_number['Predicted'], digits= 4))

# Confusion matrix
cf_matrix = confusion_matrix(classified_number['Expected'], classified_number['Predicted'] )
print(cf_matrix)


# %%
###################################################
#MODEL:  KAGGLE MODEL 
#--------------------------
def LoadModel(model_name): 
    '''
    Goal: To load a previously trained classification model. 
    Return type: keras sequential engine (?)
    Improvements: 
                    - Check return type
                    - Check if it is more efficient to load model and classify the batch in 1. 
                    - Test model --> Check if it needs arguments. 
    '''
    # load json and create model
    json_file = open(f"C://Users//tklop//ThesisADS//scripts//{model_name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"C://Users//tklop//ThesisADS//scripts//{model_name}.h5")
    loaded_model.compile(optimizer= "rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    #loaded_model.load_weights(f"C://Users//tklop//ThesisADS//scripts//callback_{model_name}model.h5")
    print(f"Succesfully loaded the {model_name} model")
        
    # evaluate loaded model on test data (remove?)
    return loaded_model 

model = LoadModel("final_model_kaggle")
model_name = "final_model_kaggle"

#%%
# Create dataframe to check wrong and correct classifications. 
classified_number = pd.DataFrame(columns= ['Predicted', 'Expected', 'Probability'])
pred_list = []
exp_list = []
prob_list = []
for idx, img_value in enumerate(test_images): 
    pred_img = img_value.reshape(1,40,40,1)
    y_prob = model.predict(pred_img)
    prediction = y_prob.argmax(axis=-1)[0]
    probability = np.max(y_prob)

    pred_list.append(prediction)
    exp_list.append(test_labels[idx].argmax(axis= 0))
    prob_list.append(probability)

classified_number['Predicted'] = pred_list
classified_number['Expected'] = exp_list
classified_number['Probability'] = prob_list

wrong_classified = classified_number.loc[~(classified_number['Predicted'] == classified_number['Expected'])]
correct_classified = classified_number.loc[(classified_number['Predicted'] == classified_number['Expected'])]
# %%
# Create probability distributions. 
counts, edges, bars = plt.hist(correct_classified['Probability'], bins= 10, range=[0, 1], facecolor='#aef2b1', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_correct_{model_name}.jpg")
plt.show()

counts, edges, bars = plt.hist(wrong_classified['Probability'], bins= 10, range=[0, 1], facecolor='#eb7b6e', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_wrong_{model_name}.jpg")
plt.show()

# %%
# Classification report. 
print(classification_report(classified_number['Expected'], classified_number['Predicted'], digits= 4))

# Confusion matrix. 
cf_matrix = confusion_matrix(classified_number['Expected'], classified_number['Predicted'] )
print(cf_matrix)

#%%
#######################################################
# MODEL: MNIST MODEL
#--------------------------------
# Get files from directory. 
directory = "C://Users//tklop//ThesisADS//CharacterImages//28x28"
Name=[]

# Create same data as before with different image size. 
df2 = pd.read_csv("C://Users//tklop//ThesisADS//labeling_by_hand.csv", sep = ';')
df1 = pd.read_csv("C://Users//tklop//ThesisADS//custom_label_v1.csv", sep = ',')
df = pd.concat([df1, df2], ignore_index= True)

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

# Load images and reverse the grayscale tones. 
for idx, name in enumerate(Name):
    im = name + ".jpg"
    try: 
        img = load_img(os.path.join(directory,im), color_mode='grayscale', target_size=(28,28,1))
        img = img_to_array(img)
        img = img/255.0
        imageNew = expand_dims(img, 0)
        imageNew = imageNew.reshape(28,28,1)
        try: 
            img_label = df['Expected'].loc[df['Crop'] == name ].item()
        except ValueError: 
            continue
        # Reverse the color of the new images. 
        ones = np.ones((28,28,1))  
        inv_image = np.subtract(ones, imageNew)  
        dataset.append([inv_image, img_label])
        
    except FileNotFoundError: 
        continue

data,labels1 =zip(*dataset)     

data=np.array(data)
labels=np.array(labels1)

# Train test split. 
train_images, test_images,train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state=44)
test_labels = to_categorical(test_labels)


temp_list = []
for idx, train_image in enumerate(train_images): 
    imageDataGen = ImageDataGenerator(rotation_range=20, width_shift_range=0.20, height_shift_range=0.20)
    temp_image = train_image.reshape(1,28,28,1)
    iterator = imageDataGen.flow(temp_image, batch_size=1)
    img_label = train_labels[idx]

    for i in range(5):
        batch = iterator.next()
        image = batch[0].astype('float32')
        temp_list.append([image, img_label])

train_images_a1,train_labels_a1 =zip(*temp_list)     


train_images_augmented =np.array(train_images_a1)
train_labels_augmented= to_categorical(train_labels_a1)

train_images, val_images,train_labels, val_labels = train_test_split(train_images_augmented,train_labels_augmented,test_size=0.2,random_state=44)

# %%
# Load the model. 
model = LoadModel("final_model_mnist")
model_name = "final_model_mnist"

#%%
# Create classified dataframe. 
classified_number = pd.DataFrame(columns= ['Predicted', 'Expected', 'Probability'])
pred_list = []
exp_list = []
prob_list = []
for idx, img_value in enumerate(test_images): 
    pred_img = img_value.reshape(1,28,28,1)
    y_prob = model.predict(pred_img)
    prediction = y_prob.argmax(axis=-1)[0]
    probability = np.max(y_prob)

    pred_list.append(prediction)
    exp_list.append(test_labels[idx].argmax(axis= 0))
    prob_list.append(probability)

classified_number['Predicted'] = pred_list
classified_number['Expected'] = exp_list
classified_number['Probability'] = prob_list

wrong_classified = classified_number.loc[~(classified_number['Predicted'] == classified_number['Expected'])]
correct_classified = classified_number.loc[(classified_number['Predicted'] == classified_number['Expected'])]
# %%
# Visualise the probability distributions. 
counts, edges, bars = plt.hist(correct_classified['Probability'], bins= 10, range=[0, 1], facecolor='#aef2b1', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_correct_{model_name}.jpg")
plt.show()

counts, edges, bars = plt.hist(wrong_classified['Probability'], bins= 10, range=[0, 1], facecolor='#eb7b6e', align='mid', edgecolor = "black")
plt.bar_label(bars)
plt.ylabel('Count values')
plt.xlabel('Probability scores') 
plt.savefig(f"prob_hist_wrong_{model_name}.jpg")
plt.show()

# %%
# Classification report. 
print(classification_report(classified_number['Expected'], classified_number['Predicted'], digits= 4))
# Confusion Matrix. 
cf_matrix = confusion_matrix(classified_number['Expected'], classified_number['Predicted'] )
print(cf_matrix)

#%%
