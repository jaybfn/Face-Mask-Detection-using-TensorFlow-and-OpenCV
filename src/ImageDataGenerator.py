""" This code is used for Image augmentation """

# Importing all the necessary libraries for Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, save_img
import os
import numpy as np
import matplotlib.pyplot as plt

# path for loading images for generating new images for Image Augmentation
base_path = r"C:\Users\Asus\Desktop\Spiced_Academy\
    naive-zatar-student-codes\week9\imageclassifier\RawData"

# path for saving images generated from ImageDataGenerator.py
save_path = r"C:\Users\Asus\Desktop\Spiced_Academy\
    naive-zatar-student-codes\week9\imageclassifier\data_face\test"

# listing the directory from the base_path
# each directory has images which corresponds to a particular class
# and we use folder name as a particular class of a image
classes = os.listdir(base_path)
#print(classes)

def generate_new_images(IDG, file_name_prefix):

    """ This function generates images using ImageDataGenerator class from keras
        The function requires 2 input arguments:
        1. Calling the class ImageDataGenerator() class and in the class 
           you can add parameters listed below:
           ['brightness_range=[0.2,1.0]','width_shift_range=0.25,height_shift_range= 0.25',
            'zoom_range=[0.5,1.0]','horizontal_flip=True','vertical_flip=True','rotation_range=90']
        2. Add a stringname which will be used as a filename
            eg: generate_new_images(IDG = ImageDataGenerator(brightness_range=[0.2,1.0]), 'BR')"""

    # To check if any folder exists in the save_path or not
    files_in_path = os.listdir(save_path)  
    # to loop over the input image folder with names as different classes
    for class_ in classes:
        files = os.listdir(base_path + '/' + class_)
        #checking if the saved path contains any folder in it and if it is empty the it
        #create a new folder with class names
        if len(files_in_path) == 0:
            img_path = os.path.join(save_path, class_)
            os.mkdir(img_path)
            # loop over the input images and apply image augmentation from keras
            for file in files:
                img = load_img(path=base_path + '/' + class_+'/'+f'{file}',target_size=(224,224))
                X_ = np.array(img)
                sample = np.expand_dims(X_,axis = 0)
                iterator = IDG.flow(sample, batch_size=1)
                print(iterator)
                # plots all the images batch wise in 3*3 matix
                for i in range(9):
                    plt.subplot(330 +1 +i)
                    batch = iterator.next()
                    img = batch[0].astype('uint8')
                    save_img(img_path+'\\'+ f'{file_name_prefix}_{i}_{file}', img)
                    plt.imshow(img)
                plt.show()
        # if folder exist it loops over and create images
        else:
            img_path = os.path.join(save_path, class_)
            for file in files:
                img = load_img(path=base_path + '/' + class_+'/'+f'{file}',target_size=(224,224))
                X_ = np.array(img)
                sample = np.expand_dims(X_,axis = 0)
                iterator = IDG.flow(sample, batch_size=1)
                print(iterator)
                for i in range(9):
                    plt.subplot(330 +1 +i)
                    batch = iterator.next()
                    img = batch[0].astype('uint8')
                    save_img(img_path+'\\'+ f'{file_name_prefix}_{i}_{file}', img)
                    plt.imshow(img)
                plt.show()
