"""
This module contains the helper functions for OpenCV based finger tracking
for use in human truth file generation

hsv_masking(frame, lower_thresh, upper_thresh) - masks an image with HSV thresholds
find_screen_cnts(frame)

Class _Tracker() provides an example implementation of the tracker functions
"""
# pylint: disable = invalid-name
from collections import OrderedDict
import cv2
import numpy as np
from scipy.spatial import distance


def hsv_masking(frame, lower_thresh_hsv, upper_thresh_hsv, lower_threshold_binary=60):
    """
    Applies a HSV mask to isolate magenta colour
    hsv_masking(frame, lower_thresh, upper_thresh) -> masked

    Parameters:
    ----------
    frame : 3D np.array of the image
    lower_tresh_hsv: lower threshold in HSV color space
    upper_tresh_hsv: upper threshold in HSV color space

    Output:
    ------
    thresholded image
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thresh_hsv, upper_thresh_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    ret = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, lower_threshold_binary, 255, cv2.THRESH_BINARY)
    return thresh, ret


def hsv_detection(hsv_mask):
    """
    Detector that outputs position of contours in the HSV mask

    Parameters:
    -----------
    hsv_mask :
    """
    contours, _ = cv2.findContours(
        hsv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    good_pts = []
    if contours:
        contour_areas = np.array([cv2.contourArea(c) for c in contours])
        max_area = max(contour_areas)
        min_area = max_area/3

        for i, cnt in enumerate(contours):
            if contour_areas[i] > min_area:
                # hull = cv2.convexHull(cnt)
                # cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
                # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)
                moment = cv2.moments(cnt)
                centroid_x = int(moment["m10"] / moment["m00"])
                centroid_y = int(moment["m01"] / moment["m00"])
                good_pts.append([centroid_x, centroid_y])
    return good_pts


def show(frame): 
    """Helper function for debugging frames"""
    cv2.imshow("frame", frame)
    cv2.waitKey()


class _Tracker():
    """
    Simple object tracker with centroid euclidian distance
    """
    def __init__(self):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = 20

    def register(self, centroid):
        """Register a new object"""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """deregister an object"""
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, positions):
        """Update the tracker"""
        # If no objects are detected in the current frame, increment disappeared
        # counter for all objects stored
        if not positions:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Initialise centroid array
        inputCentroids = np.array(positions, dtype='int')

        # If we are currently not tracking objects, register each new object
        if not self.objects:
            for cent in inputCentroids:
                self.register(cent)
        # Otherwise, update existing objects based on Euclidian distance
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the Euclidian distance between each pair
            D = distance.cdist(np.array(objectCentroids), inputCentroids, metric='euclidean')

            # Find the smallest value in each row and sort the row indices based on their minimum values
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                # If number of objects greater or equal to the unused centroids, check for disappearance
                if D.shape[0] >= D.shape[1]:
                    for urow in unusedRows:
                        objectID = objectIDs[urow]
                        self.disappeared[objectID] += 1

                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
                else:
                    for ucol in unusedCols:
                        self.register(inputCentroids[ucol])
        return self.objects

