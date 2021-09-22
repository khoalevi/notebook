from levi.neural_network import Perceptron
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", choices=["or", "and", "xor"],
                required=True, type=str, help="type of dataset")
args = vars(ap.parse_args())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

if args["mode"] == "or":
    y = np.array([[0], [1], [1], [1]])
elif args["mode"] == "and":
    y = np.array([[0], [0], [0], [1]])
else:
    y = np.array([[0], [1], [1], [0]])

print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("[INFO] testing perceptron...")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, truth={}, pred={}".format(x, target[0], pred))
