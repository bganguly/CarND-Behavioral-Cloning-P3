
# coding: utf-8

# In[1]:

# Declare module imports

import numpy as np
from random import randint
import matplotlib.pyplot as plt
import csv
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam

from IPython.core.display import HTML
get_ipython().magic('matplotlib inline')


# In[2]:

# Set some global variables for storing path to img_data and the driving_log

data_folder = './data/'
log_path = data_folder +'driving_log.csv'


# In[3]:

# Read in the driving_log
# Print some summary statistics and any random line for some context

logs = []
with open(log_path,'rt') as f:
    reader = csv.reader(f)
    for line in reader:
        logs.append(line)

log_labels = logs.pop(0)
totalLinesFromLog = len(logs) 
# store a random entry for sample printing
randomLineFromLog = randint(0, totalLinesFromLog)

print(' log headers :', log_labels)
print(' total number of log lines :', totalLinesFromLog)
print(' log line # selected at random :',randomLineFromLog)
print(' sample log line :',logs[randomLineFromLog])


# In[4]:

# Some css overrides to make the pictures render closer to their titles

HTML('<style>div.output_subarea.output_png { margin-top: -17px;}</style>')


# In[5]:

# Print a sample image and demonastrate rescaling

img = plt.imread(data_folder + (logs[randomLineFromLog][0]))
print(' sample image from input data, ', 'image shape :',img.shape)
plt.imshow(img)
plt.show()

img_trimmed = img[20:140]
print(' sample image trimmed, ', 'image shape :',img_trimmed.shape)
plt.imshow(img_trimmed)
plt.show()

img_resized = cv2.resize(img,(32,16))
print(' sample image from input data rescaled, ', 'image shape :',img_resized.shape)
plt.imshow(img_resized)
plt.show()

img_trimmed_rescaled = cv2.resize(img_trimmed,(32,16))
print(' sample image trimmed and then rescaled ', 'image shape :',img_trimmed_rescaled.shape)
plt.imshow(img_trimmed_rescaled)
plt.show()


# In[6]:

# Helper function to change color space of input data

def image_resize(img):
    return cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(32,16))


# In[7]:

# Helper function to normalise input data
# Load features 
#   get input images from all three cameras vantages
#   resize each image using helper function above
# Load labels
#   get steerting angles from all three camera vantages
#   for the left camera add a small delta to the steering angle
#   for the right camera subtract a small delta from the steering angle

def load_data(X,y,data_folder,delta=0.08):
    for i in range(len(logs)):
        img_path = logs[i][0]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        X.append(image_resize(img))
        y.append(float(logs[i][3]))
        
        img_path = logs[i][1]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        X.append(image_resize(img))
        y.append(float(logs[i][3]) + delta)
        
        img_path = logs[i][2]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        X.append(image_resize(img))
        y.append(float(logs[i][3]) - delta)


# In[8]:

# Call Helper function to Load nornalised data
# this takes about 45 seconds to complete on the udacity-carnd AWS GPU instance

data={}
data['features'] = []
data['labels'] = []
load_data(data['features'], data['labels'],data_folder,0.3)


# In[9]:

# Populate standard objects for features and labels

X_train = np.array(data['features']).astype('float32')
y_train = np.array(data['labels']).astype('float32')


# In[10]:

# Augment feature and labels via reflection of existing features and labels

X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
y_train = np.append(y_train,-y_train,axis=0)


# In[11]:

# Train and test validation splits and print some summaries

X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)
print(X_train.shape)
print(X_val.shape)


# In[12]:

# Reshape train features and labels

img_rows = 16
img_cols = 32
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)


# In[13]:

# Main Keras Model for the training run
# The lambda model eventually leads to a json with encode spaces and requires
# at least keras 1.2.0 , upon a model save/restore

def normalize_greyscale(image_data):
    return  image_data/127.5 -1.

model = Sequential()
model.add(Lambda(normalize_greyscale ,input_shape=(16,32,1)))
model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
model.add(MaxPooling2D((4,4),(4,4),'valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))


# In[15]:

# Main training runs for the Keras model

model.compile(loss='mse',optimizer='adam')
history = model.fit(X_train, y_train,batch_size=128, nb_epoch=20,verbose=1, validation_data=(X_val, y_val))


# In[16]:

# Print Model predictions

model.predict(X_train[0:10])


# In[17]:

# Save model (as json) and weights to disk

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Wrote model and weights to disk")

