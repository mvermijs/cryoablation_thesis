"""
Two similar functions, one accepts and returns image-encoded validation data (train_norm_val) and the other simply encodes different train/test splits for cv purposes (train_norm_cv)
"""

import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from orange_lib.Cart2Pixel import Cart2Pixel
from orange_lib.ConvPixel import ConvPixel
import os
import errno


def train_norm_val(param, dataset):

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    np.random.seed(param["seed"])
    y_train = dataset["Ytrain"]
    del dataset["Ytrain"]
    y_val = dataset["Yval"]
    del dataset["Yval"]
    y_test = dataset["Ytest"]
    del dataset["Ytest"]

    q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Method"],
         "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": y_train}

    # generate train images
    x_train, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                mutual_info=param["mutual_info"], params=param, only_model=False)

    del q["data"]

    # generate validation set image
    if param["mutual_info"]:
        dataset["Xval"] = dataset["Xval"].drop(dataset["Xval"].columns[toDelete], axis=1)

    dataset["Xval"] = np.array(dataset["Xval"]).transpose()

    x_val = [ConvPixel(dataset["Xval"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                             image_model["A"], image_model["B"])
                   for i in range(0, dataset["Xval"].shape[1])]
    
    # generate test set image
    if param["mutual_info"]:
        dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

    dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()

    x_test = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                             image_model["A"], image_model["B"])
                   for i in range(0, dataset["Xtest"].shape[1])]

    # GAN
    del dataset["Xtrain"]
    del dataset["Xval"]
    del dataset["Xtest"]
    x_test = np.array(x_test)
    image_size1, image_size2 = x_test[0].shape
    x_test = np.reshape(x_test, [-1, image_size1, image_size2, 1])

    #train and val data resize to pixel format
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, [-1, image_size1, image_size2, 1])
    x_val = np.array(x_val)
    x_val = np.reshape(x_val, [-1, image_size1, image_size2, 1])

    return x_train, x_val, x_test, y_train, y_val, y_test

def train_norm_cv(param, dataset):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    np.random.seed(param["seed"])
    y_train = dataset["Ytrain"]
    del dataset["Ytrain"]
    y_test = dataset["Ytest"]
    del dataset["Ytest"]

    q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Method"],
         "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": y_train}

    # generate train images
    x_train, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                mutual_info=param["mutual_info"], params=param, only_model=False)

    del q["data"]
    
    # generate test set image
    if param["mutual_info"]:
        dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

    dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()

    x_test = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                             image_model["A"], image_model["B"])
                   for i in range(0, dataset["Xtest"].shape[1])]

    # GAN
    del dataset["Xtrain"]
    del dataset["Xtest"]
    x_test = np.array(x_test)
    image_size1, image_size2 = x_test[0].shape
    x_test = np.reshape(x_test, [-1, image_size1, image_size2, 1])

    #train data resize to pixel format
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, [-1, image_size1, image_size2, 1])

    return x_train, x_test, y_train, y_test
