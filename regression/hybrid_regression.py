from levi.datasets import houses
from levi.models import mlp, cnn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to input dataset of houses")
args = vars(ap.parse_args())

print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "houses.info.txt"])
df = houses.load_house_attributes(inputPath)

print("[INFO] loading house images...")
images = houses.load_house_image(df, args["dataset"])
images = images / 255.0

print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.2)
(trainAttrX, testAttrX, trainImgX, testImgX) = split

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

(trainAttrX, testAttrX) = houses.process_house_attributes(df, trainAttrX, testAttrX)

mlpModel = mlp.create_mlp(trainAttrX.shape[1], regress=False)
cnnModel = cnn.create_cnn(64, 64, 3, regress=False)

hybrid = concatenate([mlpModel.output, cnnModel.output])

x = Dense(4, activation="relu")(hybrid)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlpModel.input, cnnModel.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=[trainAttrX, trainImgX], y=trainY,
          validation_data=([testAttrX, testImgX], testY),
          epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImgX])

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
