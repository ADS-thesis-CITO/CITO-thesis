'''
:Author: Tom Klopper. 
:Goal: To create a simple CNN using the functional keras API trained on the standard MNIST dataset. 
:Last update: 04-06-2022.
'''

# Import libraries. 
from tensorflow.keras.utils import to_categorical
from keras import layers 
from keras import models 
from keras.datasets import mnist
from tensorflow.keras.utils import plot_model 


# define parameters. 
image_size = 28
input_shape = (image_size, image_size, 1)
model_name = "functional_mnist"

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

# Plot the model architecture. 
plot_model(model, show_shapes = True, to_file = f"{model_name}_architecture.jpg" )

# Create test/train split. 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32') / 255  # Normalize to values between 0 and 1.
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255    # Normalize to values between 0 and 1. 

# Make labels categorical. 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile the model. 
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy', 
              metrics  = ['accuracy'])

# Fit the model. 
model.fit(train_images, train_labels, epochs = 10, batch_size = 64)

# Evaluate the model. 
model.evaluate(test_images, test_labels)

# Save the model as JSON and h5. 
model_json = model.to_json()
with open(f"{model_name}.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(f"{model_name}.h5")
print(f"Succesfully saved the {model_name} model.")

