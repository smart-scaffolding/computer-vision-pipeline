# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
from helper import *
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="path to input image")
args = vars(ap.parse_args())

videoTracking = True
# load the image, clone it for output, and then convert it to grayscale

if videoTracking:
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# loop over frames from the video stream
while True:
    if videoTracking:
        frame = vs.read()

        if frame is None:
            break
    else:
        frame = cv2.imread(args["image"])

    frame = imutils.resize(frame, width=500)
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heat = np.zeros_like(output[:, :, 0]).astype(np.float)
    # gray = cv2.medianBlur(gray, 5)

    # th, im_th = cv2.threshold(gray, 150, 225, cv2.THRESH_BINARY_INV)

    # im_floodfill = im_th.copy()

    # # h, w = im_th.shape[:2]
    # # mask = np.zeros((h+2, w+2), np.uint8)

    # # cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # im_out = im_th | im_floodfill_inv

    # detect circles in the image
    # image, method, dp, minDist, circles

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 0.75, 3.0, param1=100, param2=35, minRadius=0, maxRadius=40)

   # ensure at least some circles were found
    try:
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            totalCirclesFound = "Total circles found: " + str(len(circles))
            cv2.putText(output, totalCirclesFound, (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            heat = add_heat(heat, circles)

            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 10)
                cv2.rectangle(output, (x - 5, y - 5),
                              (x + 5, y + 5), (0, 128, 255), -1)
                msg = "x:" + str(x) + " y:" + str(y) + " r:" + str(r)
                cv2.putText(output, msg, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    except:
        # needed for real time processing
        pass

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 0)

    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)

    cv2.imshow("output", np.hstack([frame, output]))

    totalCirclesFound = "Total circles found: " + str(labels[1])
    cv2.putText(draw_img, totalCirclesFound, (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Heatmap Output", draw_img)

    # plt.imshow(heatmap, cmap='hot')
    # plt.show()
    # cv2.imshow("Thresholded image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    key = cv2.waitKey(1) & 0xFF
