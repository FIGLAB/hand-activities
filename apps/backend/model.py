from numpy import genfromtxt
import scipy
import logging
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model

# GLobal Variables
fft_size = 256
width = 48
num_channels = 3

####################
# Remap Labels
###################
all_classes = [
    "Hands Still - Idle",
	"Scrolling on Trackpad - Phone",
	"Typing on Keyboard",
	"Moving-Clicking Mouse",
	"Tapping Screen",
	"Playing Piano",
	"Brushing Hair",
	"Swaying (while Locomoting)",
	"Writing (with implement)",
	"Using Scissors",
	"Operating Hand Drill",
	"Using Remote - Game Controller",
	"Petting",
	"Clapping",
	"Scratching",
	"Opening Door",
	"Opening Jar",
	"Pouring Drink",
	"Drinking",
	"Grating",
	"Chopping Vegetables",
	"Wiping (while cleaning)",
	"Washing Utensils",
	"Washing Hands",
	"Brushing Teeth"
]

all_classes.append("Other")

classes_to_include = all_classes
num_classes = len(classes_to_include)

# Load Model
model = load_model("../../models/model_main.hdf5")

# Dataset Transformation + Feature Pipeline
def extract_features(img):
    #data_max = 22948992.0
    data_max = 8800*120
    try:
        signal = img.reshape(img.shape[0], fft_size, width, num_channels).astype('>f4')
        signal /= data_max
        feats = signal
        return feats
    except ValueError:
        return False
        
def encode_labels(dataset_Y, dataset_labels, keep_list, one_hot=True):
    ix = np.in1d(dataset_labels.ravel(), keep_list).reshape(dataset_labels.shape)
    Y = dataset_Y[ix]
    L = dataset_labels[ix]
    
    # Perform one-hot encoding
    if (one_hot==True):
        #unique = np.sort(np.unique(L)) # Sorted alphabetically
        unique = np.sort(np.array(keep_list))
        print(unique)
        label_mapper = dict()
        for i in range(len(unique)):
            k = unique[i]
            label_mapper[k] = i
        
        print(label_mapper)
        
        label_indices = np.array([label_mapper[k] for k in L])
        #n_values = np.max(len(unique))
        n_values = np.max(len(keep_list))
        Y = np.eye(n_values)[label_indices]
    else:
        # Remap Indices
        for k in range(len(keep_list)):
            classname = keep_list[k]
            ix = np.in1d(L.ravel(), [classname]).reshape(L.shape)
            Y[ix] = k

    return Y
    
def predict(X):
    res = model.predict(extract_features(X))
    if (type(res) != type(False)):
        res = res.astype('>f4')
    return res