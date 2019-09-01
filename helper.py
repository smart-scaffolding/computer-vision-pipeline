import numpy as np
import cv2


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox

        heatmap[box[1]-box[2]:box[1]+box[2], box[0]-box[2]:box[0]+box[2]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for detected_block in range(1, labels[1]+1):

        # Find pixels with each block label value
        nonzero = (labels[0] == detected_block).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        centerx = int((np.max(nonzerox) + np.min(nonzerox))/2)
        centery = int((np.max(nonzeroy) + np.min(nonzeroy))/2)
        radius = int((np.max(nonzerox) - np.min(nonzerox))/2)

        cv2.circle(img, (centerx, centery), radius, (0, 0, 255), 10)

    return img
