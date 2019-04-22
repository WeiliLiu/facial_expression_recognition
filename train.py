import tensorflow
import numpy as np
import os
import argparse
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import scipy.misc
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
from glob import glob
import pandas as pd
from sklearn.externals import joblib

'''The following is the implementation of the VGG16 neural network'''

def main():
    '''h, w = 48, 48
    data = preprocess()
    X_train, X_test, y_train, y_test = split_data(data)
    h, w = X_train[0].shape
    num_train_sample = X_train.shape
    print(num_train_sample)
    epochs = 50
    batch_size = 1'''
    # xx = X_train[0] * 255
    # print(xx)
    # print(X_train[0].shape)
    # x_pre = preprocess_input(X_train[0])*255
    # print(x_pre)
    # print(x_pre.shape)
    # scipy.misc.imsave('ori_sample.jpg', xx)
    # scipy.misc.imsave('sample.jpg', x_pre)
    #model = VGG16(include_top=False, input_shape=(48, 48, 3))

    #print(model.summary())
    #plot_model(model, to_file='vgg.png')
    training_dir = 'emotion/training'
    validation_dir = 'emotion/validation'
    testing_dir = 'emotion/shit'

    image_files = glob(training_dir + '/*/*.jp*g')
    valid_image_files = glob(validation_dir + '/*/*.jp*g')

    folders = glob(training_dir + '/*')
    num_classes = len(folders)
    print ('Total Classes = ' + str(num_classes))

    # !mkdir ~/.keras
    # !mkdir ~/.keras/models
    # !cp keras-pretrained-models/*notop* ~/.keras/models/
    # !cp keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

    IMAGE_SIZE = [48, 48]
    vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    #x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
    x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

    model = Model(inputs = vgg.input, outputs = x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

    validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

    training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')


    print('training shape:', training_generator)
    training_generator.class_indices

    # training_images = 37836
    # validation_images = 12709

    history = model.fit_generator(training_generator,
                   steps_per_epoch = 981,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results.
                   epochs = 3,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = 200)  # this should be equal to total number of images in validation set.


    joblib.dump(model, 'shit.pkl')
    print ('Training Accuracy = ' + str(history.history['acc']))
    print ('Validation Accuracy = ' + str(history.history['val_acc']))

    testing_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)
    testing_generator = testing_datagen.flow_from_directory(testing_dir, target_size = IMAGE_SIZE, batch_size = 200)


    num_step = len(testing_generator)
    print('step should be:', num_step)
    #testing_img = cv2.imread('emotion/testing/1.jpg')
    predictions = model.predict_generator(testing_generator, steps=num_step, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    #predicted_classes = convert_to_class(predictions)
    print(predictions)
    num_label = predictions.shape[0]
    labels = []
    for i in range(predictions.shape[0]):
        labels.append(np.argmax(predictions[i]))


    print(labels)
    true_label = predictions.count(2)
    accuracy = true_label / num_label
    print('accuracy is:', accuracy)

if __name__ == "__main__":
    main()
