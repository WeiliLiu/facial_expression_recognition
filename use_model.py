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
import scipy.misc
from detect_face import detect



def main():
    # test_image = cv2.imread('test.jpg')
    # faces = np.array(detect(test_image))
    # count = 0
    # for face in faces:
    #     scipy.misc.imsave('emotion/real_face/shit/{}.jpg'.format(count), face)
    #     count += 1

    testing_dir = 'emotion/shit'
    IMAGE_SIZE = [48, 48]
    testing_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)
    testing_generator = testing_datagen.flow_from_directory(testing_dir, target_size = IMAGE_SIZE, batch_size = 200)

    model = joblib.load('shit.pkl')

    num_step = len(testing_generator)
    print('step should be:', num_step)
    #testing_img = cv2.imread('emotion/testing/1.jpg')
    predictions = model.predict_generator(testing_generator, steps=num_step, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    #predicted_classes = convert_to_class(predictions)
    print(predictions)
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    num_label = predictions.shape[0]
    labels = []
    for i in range(predictions.shape[0]):
        labels.append(emotions[np.argmax(predictions[i])])

    print(labels)
    true_label = labels.count('surprise')
    accuracy = true_label / num_label
    print('accuracy is:', accuracy)

if __name__ == "__main__":
    main()
