# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

# load the image, clone it for output, and then convert it to grayscale
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:

    frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)

    # th, im_th = cv2.threshold(gray, 150, 225, cv2.THRESH_BINARY_INV)

    # im_floodfill = im_th.copy()

    # # h, w = im_th.shape[:2]
    # # mask = np.zeros((h+2, w+2), np.uint8)

    # # cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # im_out = im_th | im_floodfill_inv

    # detect circles in the image
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 0.03, 100, 0.3, param1=75, param2=40, minRadius=0, maxRadius=40)

   # ensure at least some circles were found
    try:
        if circles is not None:

            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5),
                              (x + 5, y + 5), (0, 128, 255), -1)
                # cv2.circle(im_th, (x, y), r, (0, 255, 0), 4)
                # cv2.rectangle(im_th, (x - 5, y - 5),
                #               (x + 5, y + 5), (0, 128, 255), -1)
    except:
        # needed for real time processing
        pass

    cv2.imshow("output", np.hstack([frame, output]))
    cv2.imshow("gray", gray)
    # cv2.imshow("Thresholded image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    key = cv2.waitKey(1) & 0xFF
