

"""## Explore the Example Data"""


import os
base_dir = './CovidDataset/'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Val')

# Directory with our training covid pictures
train_covid_dir = os.path.join(train_dir, 'Covid')

# Directory with our training normal pictures
train_normal_dir = os.path.join(train_dir, 'Normal')

# Directory with our validation covid pictures
validation_covid_dir = os.path.join(validation_dir, 'Covid')

# Directory with our validation normal pictures
validation_normal_dir = os.path.join(validation_dir, 'Normal')

train_covid_fnames = os.listdir(train_covid_dir)
print(train_covid_fnames[:10])

train_normal_fnames = os.listdir(train_normal_dir)
train_normal_fnames.sort()
print(train_normal_fnames[:10])

"""Let's find out the total number of cat and dog images in the `train` and `validation` directories:"""

print('total training covid images:', len(os.listdir(train_covid_dir)))
print('total training normal images:', len(os.listdir(train_normal_dir)))
print('total validation covid images:', len(os.listdir(validation_covid_dir)))
print('total validation normal images:', len(os.listdir(validation_normal_dir)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_covid_pix = [os.path.join(train_covid_dir, fname) 
                for fname in train_covid_fnames[pic_index-8:pic_index]]
next_normal_pix = [os.path.join(train_normal_dir, fname) 
                for fname in train_normal_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_covid_pix+next_normal_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# Ze model!
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# Our input feature map is 224x224x3: 224x224 for the image pixels, and 3 for
# the three color channels: R, G, and B
model = Sequential()

# First convolution extracts 32 filters that are 3x3 and a second(64 filters)
# Convolution is followed by max-pooling layer with a 2x2 window
model.add(Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Second convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Third convolution extracts 128 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid", name="visualized_layer"))

model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["accuracy"])"""

# On top of it we stick two fully-connected layers. Because we are facing a two-class classification problem, i.e. a *binary classification problem*, we will end our network with a [*sigmoid* activation](https://wikipedia.org/wiki/Sigmoid_function), so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
# Let's summarize the model architecture

#model.summary()

# The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by a bit due to padding, and each pooling layer halves the feature map.

# Data Preprocessing

Let's set up data generators that will read pictures in our source folders, convert them to `float32` tensors, and feed them (with their labels) to our network. We'll have one generator for the training images and one for the validation images. Our generators will yield batches of 20 images of size 150x150 and their labels (binary).

As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the `[0, 1]` range (originally all values are in the `[0, 255]` range).

In Keras this can be done via the `keras.preprocessing.image.ImageDataGenerator` class using the `rescale` parameter. This `ImageDataGenerator` class allows you to instantiate generators of augmented image batches (and their labels) via `.flow(data, labels)` or `.flow_from_directory(directory)`. These generators can then be used with the Keras model methods that accept data generators as inputs: `fit_generator`, `evaluate_generator`, and `predict_generator`."""


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 224x224
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

"""### Training
#Let's train on all 224 images available, for 15 epochs, and validate on all 60 validation images. (This may take a few minutes to run.)
"""

history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,  
      epochs=11,
      validation_data=validation_generator,
      validation_steps=2  
)

model.evaluate_generator(validation_generator)

"""### Visualizing Intermediate Representations

#To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.

#Let's pick a random covid or normal image from the training set, and then generate a figure where each row is the output of a layer, and each image in the row is a specific filter in that output feature map. Rerun this cell to generate intermediate representations for a variety of training images.
"""

import matplotlib.pyplot as plt

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# fig size
#nrows = 4
#ncols = 4
#pic_index = 0

#fig = plt.gcf()
#fig.set_size_inches(ncols * 4, nrows * 4)
#pic_index += 8


# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
#img_input = layers.Input(shape=( 224, 224,3))
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(input=model.inputs, output=successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
covid_img_files = [os.path.join(train_covid_dir, f) for f in train_covid_fnames]
normal_img_files = [os.path.join(train_normal_dir, f) for f in train_normal_fnames]
img_path = random.choice(covid_img_files + normal_img_files)

img = load_img(img_path, target_size=(224, 224))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    #square = 1
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
        #ix = 1
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
      # Display the grid
      #sp = plt.subplot(nrows, ncols, i+1)
      #sp.axis("off")
        #ax = plt.subplot(square, square, ix)
        #ax.set_xticks([])
        ##ax.set_yticks([])
        #plt.imshow(feature_map[0, :, :, ix-1], cmap="cividis")
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect="auto", cmap='viridis')
  plt.show()



from google.colab import files
uploaded = files.upload()

type(validation_generator)

# Saliency map
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
from keras import activations
import scipy.ndimage as ndimage
import matplotlib.image as mpimg

#get the index of the "visualized_layer"
layer_index = utils.find_layer_idx(model, "visualized_layer")

# swap sigmoid with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

#img_path_2 = [os.listdir( "16660_1_1.jpg")]
img2 = mpimg.imread("16660_1_1.jpg")
img3 = img_to_array(img2)

plt.imshow(img2)

for i , modifier in enumerate([None, "guided", "relu"]):
  grads = visualize_cam(model, layer_index, filter_indices=None, seed_input=img3, backprop_modifier=modifier, grad_modifier="absolute")
  if modifier is None:
    modifier = "vanilla"
  ax[i+1].set_title(modifier)
  ax[i+1].imshow(grads, cmap="jet")








### Evaluating Accuracy and Loss for the Model

#Let's plot the training/validation accuracy and loss as collected during training:


# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')