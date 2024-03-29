import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from levi.nn.conv import MiniVGGNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib
matplotlib.use("Agg")


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, type=str,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]


learningRate = 0.01
epochs = 40
decay = learningRate / epochs
momentum = 0.9
batchSize = 64

print("[INFO] compiling model...")
opt = SGD(lr=learningRate, decay=decay, momentum=momentum, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=batchSize, epochs=epochs, verbose=1)

print("[INFO] evaluating network...")
preds = model.predict(testX, batch_size=batchSize)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
                            target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args["output"])
