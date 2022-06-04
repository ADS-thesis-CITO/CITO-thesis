#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

from keras import layers 
from keras import models 
import cv2 
from numpy import expand_dims

#%% 
directory = "C://Users//tklop//math_data//dataset"

Name=[]
for file in os.listdir(directory):
    if file!='.directory':
        Name+=[file]
print(Name)
print(len(Name))

#%%
dataset=[]
testset=[]
count_jpg = 0
count_png = 0
count = 0 
for idx, name in enumerate(Name):
    path=os.path.join(directory,name)
    
    for id_idx, im in enumerate(os.listdir(path)):
        if im[-4:]=='.jpg': 
            img = load_img(os.path.join(path,im), color_mode='grayscale', target_size=(40,40,1))
            img = img_to_array(img)
            img = img/255.0
            imageNew = expand_dims(img, 0)
            imageDataGen = ImageDataGenerator(rotation_range=45)
            # because as we alreay load image into the memory, so we are using flow() function, to apply transformation
            iterator = imageDataGen.flow(imageNew, batch_size=1)
            for i in range(9):
                # we are below define the subplot
                plt.subplot(330 + 1 + i)
                # generating images of each batch
                batch = iterator.next()
                # again we convert back to the unsigned integers value of the image for viewing
                image = batch[0].astype('float32')
                dataset.append([image, count])
                # we plot here raw pixel data
                plt.imshow(image, cmap = 'gray')
                count_jpg += 1
        if im[-4:]=='.png':
            img = load_img(os.path.join(path,im), color_mode='grayscale', target_size=(40,40,1))
            img = img_to_array(img)
            img = img/255.0
            imageNew = expand_dims(img, 0)
            imageDataGen = ImageDataGenerator(rotation_range=45)
            # because as we alreay load image into the memory, so we are using flow() function, to apply transformation
            iterator = imageDataGen.flow(imageNew, batch_size=1)
            for i in range(9):
                # we are below define the subplot
                plt.subplot(330 + 1 + i)
                # generating images of each batch
                batch = iterator.next()
                # again we convert back to the unsigned integers value of the image for viewing
                image = batch[0].astype('float32')
                testset.append([image, count])
                # we plot here raw pixel data
                plt.imshow(image, cmap = 'gray')
                count_png += 1
    count = count + 1
print(count_png)
print(count_jpg)
# %%
data,labels0=zip(*dataset)
test,tlabels0=zip(*testset)
labels1=to_categorical(labels0)
data=np.array(data)
labels=np.array(labels1)

tlabels1=to_categorical(tlabels0)
test=np.array(test)
tlabels=np.array(tlabels1)
# %%
train_images, test_images,train_labels, test_labels = train_test_split(data,labels,test_size=0.2,random_state=44)
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (40,40,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()



# %%
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy', 
              metrics  = ['accuracy'])

history = model.fit(train_images, train_labels, epochs = 10, batch_size = 64)

# %%
acc = history.history['accuracy']
loss = history.history[ 'loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, label = 'Training Loss' )
plt.plot(epochs, acc, label = 'Training Accuracy')
plt.suptitle('Training Accuracy and Loss') 
plt.title('Used augmented kaggle data')
plt.legend()
plt.show()

#%%
model.evaluate(test_images, test_labels)

#%%
# serialize model to JSON
model_json = model.to_json()
with open("model_data_augmentation.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data_augmentation.h5")
print("Saved model to disk")

