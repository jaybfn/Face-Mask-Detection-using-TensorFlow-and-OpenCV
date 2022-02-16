# importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2


# explore the data folder
base_path = r"../data_face/test"
os.listdir(base_path)   

# define the classes
classes = os.listdir(base_path)

X_list = []
y_list = []
for class_ in classes:
    files = os.listdir(base_path + '/' + class_)
    #print(files)
    for file in files:
        img = load_img(path=base_path + '/' + class_+'/'+f'{file}',target_size=(128,128))
        x = np.array(img)
        X_list.append(x)
        y_list.append(class_)
        
X = np.array(X_list)/255
y = np.array(y_list)

#plot the image with labels

# plt.figure(figsize=(16,16))
# for i in range(51):
#     plt.subplot(8,8,i+1,title=f'Class:{y[i]}')
#     plt.imshow(X = X[i])
#     plt.axis('off')


y_series = pd.Series(y).map({classes[0]:0, classes[1]:1})
y = to_categorical(y_series)

#Splitting the data into train and test
x_train, x_test, y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

K.clear_session()

#defining the CNN2D model
model = Sequential([

    Conv2D(filters= 6, kernel_size=(5,5), strides=(1,1), input_shape=x_train[0].shape,
           activation=keras.activations.relu,
           padding='valid', kernel_regularizer=l2(0.0001)), # valid means no padding

    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'),
    
    Conv2D(filters = 16, kernel_size=(5,5),strides=(1,1),
           activation=keras.activations.relu, 
           padding = 'valid',kernel_regularizer=l2(0.0001)), # valid means no padding
    
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'), # valid means no padding
    
    Flatten(),
    
    # Fully connected dense layer with relu activation function
    Dropout(0.2),
    Dense(units=256, activation=keras.activations.relu),
    BatchNormalization(),
    
    # Fully connected dense layer with relu activation function
    Dropout(0.2),
    Dense(units=128, activation=keras.activations.relu),
    BatchNormalization(),

    # Fully connected dense layer with relu activation function
    Dropout(0.2),
    Dense(units=64, activation=keras.activations.relu),
    BatchNormalization(),

    # Fully connected dense layer with relu activation function
    Dropout(0.2),
    Dense(units=32, activation=keras.activations.relu),
    BatchNormalization(),

    # Fully connected output layer with softmax
    Dense(units=2, activation=keras.activations.softmax)   # How many neurons? we want classify two classes
    
])

# compling the model
model.compile(optimizer=keras.optimizers.Adam(0.5e-4), 
                loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

cb = keras.callbacks.EarlyStopping(monitor="accuracy",
                                    min_delta=0.005,
                                    patience=30,
                                    verbose=1,
                                    mode="min",
                                    baseline=None,
                                    restore_best_weights=False)

history = model.fit(x_train,y_train, 
                    epochs = 50, 
                    batch_size = 30, 
                    verbose = 1, 
                    validation_split=0.3,
                    callbacks=[cb])

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1.1) # set the y range to [0,1]
plt.show()
print('Model Evalution Score')
print(model.evaluate(x_test, y_test))

# To save the model

model.save(r"..\models\model.h5")
