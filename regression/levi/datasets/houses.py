from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_house_attributes(inputPath):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep="", header=None, names=cols)

    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    for (zipcode, count) in zip(zipcodes, counts):
        # start filter zip codes with low counts
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
        # end filter

    return df


def process_house_attribute(df, train, test):
    continuous = ["bedroom", "bathrooms", "area"]

    scaler = MinMaxScaler()
    trainContinuous = scaler.fit_transform(train[continuous])
    testContinuous = scaler.transform(test[continuous])

    binarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = binarizer.transform(train["zipcode"])
    testCategorical = binarizer.transform(test["zipcode"])

    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    return (trainX, testX)
