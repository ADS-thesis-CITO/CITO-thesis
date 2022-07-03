'''
:Author: Tom Klopper. 
:Goal: To create a simple CNN using the functional keras API trained on the standard MNIST dataset. 
:Last update: 03-07-2022.
'''
#%%
# Import libraries. 
from tensorflow.keras.utils import to_categorical
from keras import layers 
from keras import models 
from keras.datasets import mnist
from tensorflow.keras.utils import plot_model 
import matplotlib.pyplot as plt
from keras import callbacks




# Create test/train split. 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32') / 255  # Normalize to values between 0 and 1.
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255    # Normalize to values between 0 and 1. 

#%%
import numpy as np
import pandas as pd 
df = pd.DataFrame(columns = ['train', 'val'])
xtrain = train_labels.tolist()
df['train'] = xtrain
print(df['train'].value_counts())
#%%
df = pd.DataFrame(columns = ['train', 'val'])
xval= test_labels.tolist()
df['val'] = xval 
print(df['val'].value_counts())
#%%

#%%

# Make labels categorical. 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# define parameters. 
image_size = 28
input_shape = (image_size, image_size, 1)
model_name = "final_model_mnist"
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
axes.set_ylim([0.925,1.01])
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
axes.set_ylim([-0.01,0.2])
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