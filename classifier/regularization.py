from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from levi.preprocessors import ResizePreprocessor
from levi.loaders import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

rsp = ResizePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[rsp])

(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

for r in (None, "l1", "l2"):
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=10,
                          learning_rate="constant", tol=1e-3, eta0=0.01)
    model.fit(trainX, trainY)

    acc = model.score(testX, testY)
    print("[INFO] {} penalty accuracy: {:.2f}%".format(r, acc * 100))
