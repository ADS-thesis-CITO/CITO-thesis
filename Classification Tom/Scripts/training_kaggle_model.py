'''
:Author: Tom Klopper. 
:Goal: To create a simple CNN using the functional keras API trained on the Augmented Kaggle dataset. 
:Last update: 03-07-2022.
'''
#%%
# Import Libraries. 
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model 

from sklearn.model_selection import train_test_split

from keras import layers 
from keras import models 
from numpy import expand_dims
from keras import callbacks


# Get files from directory. 
directory = "C://Users//tklop//math_data//dataset"
Name=[]
for file in os.listdir(directory):
    if file!='.directory':
        Name+=[file]

# Define variables. 
dataset=[]
testset=[]
count_jpg = 0
count_png = 0
count = 0

for idx, name in enumerate(Name):
    path=os.path.join(directory,name)
    
    for id_idx, im in enumerate(os.listdir(path)):
        # Use .jpg files for training data. 
        if im[-4:]=='.jpg': 
            img = load_img(os.path.join(path,im), color_mode='grayscale', target_size=(40,40,1))
            img = img_to_array(img)
            img = img/255.0
            imageNew = expand_dims(img, 0)
            # Apply random rotation to images. 
            imageDataGen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15, height_shift_range=0.15)
            # Use Flow to apply transformation
            iterator = imageDataGen.flow(imageNew, batch_size=1)
            # Repeat 9 times. 
            for i in range(9):
                batch = iterator.next()
                image = batch[0].astype('float32')
                dataset.append([image, count])
                count_jpg += 1
        
        # Use .png files for testing data. 
        if im[-4:]=='.png':
            img = load_img(os.path.join(path,im), color_mode='grayscale', target_size=(40,40,1))
            img = img_to_array(img)
            img = img/255.0
            imageNew = expand_dims(img, 0)
            # Apply random rotation to images. 
            imageDataGen = ImageDataGenerator(rotation_range=20, width_shift_range=0.20, height_shift_range=0.20)
            # Use Flow to apply transformation
            iterator = imageDataGen.flow(imageNew, batch_size=1)
            # Repeat 9 times. 
            for i in range(9):
                batch = iterator.next()
                image = batch[0].astype('float32')
                testset.append([image, count])
                count_png += 1
    count = count + 1

#%%
# Create Train / Test split. 
data,labels0=zip(*dataset)
test,tlabels0=zip(*testset)

labels1=to_categorical(labels0)
data=np.array(data)
labels=np.array(labels1)

tlabels1=to_categorical(tlabels0)
test=np.array(test)
tlabels=np.array(tlabels1)

train_images, test_images,train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state=44)
print(f"shape train images: {train_images.shape}")
print(f"shape test images: {test_images.shape}")
print(f"shape train labels: {train_labels.shape}")
print(f"shape test labels: {test_labels.shape}")


#%%
# define parameters. 
image_size = 40
input_shape = (image_size, image_size, 1)
model_name = "final_model_kaggle"
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
                    validation_data= (test_images, test_labels))

#%%
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
axes = plt.gca()
axes.set_ylim([-0.01,0.7])
plt.savefig(f"Loss_{model_name}")
plt.show()

# Evaluate Model
model.evaluate(test_images, test_labels)
#%%

# Save the model as JSON and h5. 
model_json = model.to_json()
with open(f"{model_name}.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(f"{model_name}.h5")
print(f"Succesfully saved the {model_name} model.")
#%%
acc = history.history['accuracy']
loss = history.history[ 'loss']
val_acc = history.history['val_accuracy']
val_loss = history.history[ 'val_loss']

#%%
# Accuracy: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, acc, label = 'Training accuracy')
plt.plot(epochs, val_acc, label = 'Validation accuracy', linestyle = '--' )

plt.ylabel('Accuracy')
plt.xlabel('Number of epochs') 
plt.legend(loc='lower right')
axes = plt.gca()
axes.set_ylim([0.7,1.05])
plt.savefig(f"Accuracy_{model_name}")
plt.show()

#%%
# Loss: 
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,5))
plt.plot(epochs, loss, label = 'Training loss')
plt.plot(epochs, val_loss, label = 'Validation loss', linestyle = '--' )

plt.ylabel('Loss')
plt.xlabel('Number of epochs') 
plt.legend(loc='upper right')
axes = plt.gca()
axes.set_ylim([-0.05,0.5])
plt.savefig(f"Loss_{model_name}")
plt.show()
# %%
