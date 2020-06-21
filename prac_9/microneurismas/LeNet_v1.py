#!/usr/bin/python3
## -*- coding: utf-8 -*-
#
#Author: Fernando Rodrigo Aguilar Javier
#Author email: faguilar@comunidad.unam.mx
#
#Author: Sergio Rogelio Tinoco Martinez
#Author email: stinoco@enesmorelia.unam.mx

#Detalles del codigo 
#length of Data set train: 39429
#Positive  Negative
#7848      31581
###############################
#length of Data set test: 3727 
#Positive  Negative
#218       3509  
##############################s
#43156 Instances
##################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%################################
# import the necessary packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import skimage as sk
import skimage.transform
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix#added
from skimage.io import imread, imshow
from scipy import ndarray


#define the convnet 
class LeNet:
	@staticmethod
	def build(input_shape, classes, DROPOUT=.1):
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(8, kernel_size=5, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01), padding="valid",
			input_shape=input_shape))
		model.add(Activation("relu"))
		#model.add(Dropout(.3))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# CONV => RELU => POOL
		#model.add(Conv2D(16, kernel_size=5, padding="same"))
		#model.add(Activation("relu"))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(.5))
 
		# a sigmoid classifier
		model.add(Dense(1))
		model.add(Activation("sigmoid"))
		#model.add(BatchNormalization())
		#model.add(Dropout(.1))

		return model

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def data_augmentation(img, x_train, y_train):    
    ##++Data augmentation++##
    #Flipping
    fliping = horizontal_flip(img)
    x_train.append(fliping)
    y_train.append(1)
    #Rotattion
    rotation = random_rotation(img)
    x_train.append(rotation)
    y_train.append(1)
    #Add Noise
    noise = random_noise(img)
    x_train.append(noise)
    y_train.append(1)
    return 0    


def gen_data():
    #Constantes
    path_train = '../microneurismas/dataset_median/'#++
    path_test = '../microneurismas/dataset_median/'#++
    clas = ['Negative', 'Positive']#++
    #dta_augm = True
    # un 0 significa que no es un Microneurismas
    #0-----> Negativo
    #1-----> Positivo
    #Neg
    i_train_neg = 31581
    i_test_neg = 35090
    #Pos
    i_train_pos = 1962
    i_test_pos = 2180
    #Filepath
    filepath = np.load('filepath.npy')

    
    #df = pd.read_csv('longitud_img.csv', header=None)
    negfilepath = filepath[:35090]
    posfilepath = filepath[35090:]
    np.random.shuffle(negfilepath)
    np.random.shuffle(posfilepath)

    #Cargando los datos
    x_train, y_train, x_test, y_test = [], [], [], []
    #-- path
    counter = 0
    while counter < i_train_neg:
        x_train.append(imread(negfilepath[counter]))
        y_train.append(0)
        counter += 1
        #print(counter)
    while counter < i_test_neg:
        x_test.append(imread(negfilepath[counter]))
        y_test.append(0)
        counter += 1
        #print(counter)
    
    #++ path
    counter=0
    while counter < i_train_pos:
        img = imread(posfilepath[counter])
        x_train.append(img)
        y_train.append(1)
        data_augmentation(img,x_train, y_train)#For data augmentation
        counter += 1
        #print(counter)
    while counter < i_test_pos:
        c = clas[1]
        x_test.append(imread(posfilepath[counter]))
        y_test.append(1)
        counter += 1
        #print(counter)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    np.save('x_train', x_train)
    np.save('y_train', y_train)
    np.save('x_test', x_test)
    np.save('y_test', y_test)
    return 0

if __name__ == '__main__':
    np.random.seed(1671)  # for reproducibility
    # network and training
    NB_EPOCH = 200#20
    BATCH_SIZE = 4096#128
    VERBOSE = 1
    OPTIMIZER = Adam(lr=0.0001)
    VALIDATION_SPLIT = 0.1
    IMG_ROWS, IMG_COLS = 21, 21 # input image dimensions
    NB_CLASSES = 2  # number of outputs 
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
    k=10#++
    k_folds = [[], [], [], [], [], []]#++
    #Index 0 ---> train_loss
    #Index 1 ---> train_acc
    #Index 2 ---> val_loss
    #Index 3 ---> val_acc
    #Index 4 ---> test_loss
    #Index 5 ---> test_acc
    
    for i in range(k):
        #For gen_data in each iteration
        gen_data()
        #Load data
        X_train, y_train = np.load('x_train.npy', allow_pickle=True), np.load('y_train.npy')
        X_test, y_test = np.load('x_test.npy', allow_pickle=True), np.load('y_test.npy')

        # consider them as float and normalize
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255 
        X_test /= 255
        #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        X_train = X_train.reshape((-1, IMG_ROWS, IMG_COLS, 1 ))
        y_train = y_train.reshape((-1, 1 ))
        X_test = X_test.reshape((-1, IMG_ROWS, IMG_COLS, 1 ))
        y_test = y_test.reshape((-1, 1 ))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=['acc'])

        history = model.fit(X_train, y_train, 
            batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
            verbose=VERBOSE, validation_split=VALIDATION_SPLIT, shuffle=True)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        k_folds[0].append(history.history['loss'][-1])
        k_folds[1].append(history.history['acc'][-1])
        k_folds[2].append(history.history['val_loss'][-1])
        k_folds[3].append(history.history['val_acc'][-1])
        ##+++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Predict and show results
        score = model.evaluate(X_test, y_test, verbose=VERBOSE)
        print("\nTest score:", score[0])
        print('Test accuracy:', score[1])
        ##+++++++++++++++++++++++++++++++++++++++++++++++++++++
        k_folds[4].append(score[0])
        k_folds[5].append(score[1])
        ##+++++++++++++++++++++++++++++++++++++++++++++++++++++
        print('iteracion %s' %(i))
        
    #y_pred1 = model.predict(X_test)
    #y_pred = np.argmax(y_pred1, axis=1)
    # Print f1, precision, and recall scores
    #print("P: %s" %(precision_score(y_test, y_pred , average="macro")))
    #print("R: %s" %(recall_score(y_test, y_pred , average="macro")))
    #print("F1: %s" %(f1_score(y_test, y_pred , average="macro")))
    ##++++++++++++++++
    # list all data in history
    print(history.history.keys())
    for i in range(6):
        print(sum(k_folds[i]) / k)
    # summarize history for accuracy
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    # summarize history for loss
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()   
