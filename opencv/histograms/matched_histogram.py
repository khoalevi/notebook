from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True,
                help="path the source image")
ap.add_argument("-r", "--ref", required=True,
                help="path the reference image")
args = vars(ap.parse_args())

src = cv2.imread(args["src"])
ref = cv2.imread(args["ref"])

multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)

cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)

cv2.waitKey(0)

(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for (i, image) in enumerate((src, ref, matched)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (j, color) in enumerate(("red", "green", "blue")):
        (hist, bins) = exposure.histogram(image[..., j], source_range="dtype")
        axs[j, i].plot(bins, hist / hist.max())
        (cdf, bins) = exposure.cumulative_distribution(image[..., j])
        axs[j, i].plot(bins, cdf)
        axs[j, 0].set_ylabel(color)

axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

plt.tight_layout()
plt.show()