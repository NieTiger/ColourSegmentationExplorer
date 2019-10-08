# Simple object tracking with colour
import datetime
import time
import cv2
import numpy as np
import imutils
from imutils.video import FPS, WebcamVideoStream
from helper import (
    hsv_masking,
    hsv_detection,
    show,
    _Tracker
)

# parameters
THRESHOLD = 60  # BINARY threshold

UPPER_THRESH_MAGENTA = np.array([255, 255, 255])
LOWER_THRESH_MAGENTA = np.array([140, 110, 100])


def nothing(x):
    pass


if __name__ == "__main__":
    height = 300
    stream = WebcamVideoStream(0)
    stream.start()
    tracker = _Tracker()
    fps = FPS().start()

    print("[INFO] Camera warming up . . .")
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    cv2.namedWindow('Explorer')
    cv2.createTrackbar('HueMax', 'Explorer', 255, 255, nothing)
    cv2.createTrackbar('SatMax', 'Explorer', 255, 255, nothing)
    cv2.createTrackbar('ValMax', 'Explorer', 255, 255, nothing)
    cv2.createTrackbar('HueMin', 'Explorer', 0, 255, nothing)
    cv2.createTrackbar('SatMin', 'Explorer', 0, 255, nothing)
    cv2.createTrackbar('ValMin', 'Explorer', 0, 255, nothing)
    cv2.createTrackbar('GrayMin', 'Explorer', 60, 200, nothing)
    upperThresh = np.zeros(3)
    lowerThresh = np.zeros(3)
    cannyLowerThresh = 50

    GRAY = False

    while not stream.stopped:
        # Get new frame
        frame = stream.read()
        frame = imutils.resize(frame, height=height)
        fps.stop()

        # Get track bar values
        upperThresh[0] = cv2.getTrackbarPos('HueMax', 'Explorer')
        upperThresh[1] = cv2.getTrackbarPos('SatMax', 'Explorer')
        upperThresh[2] = cv2.getTrackbarPos('ValMax', 'Explorer')
        lowerThresh[0] = cv2.getTrackbarPos('HueMin', 'Explorer')
        lowerThresh[1] = cv2.getTrackbarPos('SatMin', 'Explorer')
        lowerThresh[2] = cv2.getTrackbarPos('ValMin', 'Explorer')
        cannyLowerThresh = cv2.getTrackbarPos('GrayMin', 'Explorer')

        # Gray scale mode and colour mode
        if GRAY:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            res1 = cv2.Canny(gray, cannyLowerThresh, 200)
        else:
            pass

        # Detect Object
        hsv_mask, res1 = hsv_masking(
            frame, lowerThresh, upperThresh)

        # Track Object
        track_position = hsv_detection(hsv_mask)

        # Calculate node and draw finger
        for centroid in track_position:
            # Draw and label each finger
            cv2.circle(frame, tuple(centroid), 6, (255, 255, 255), -1)
            cv2.putText(frame, "(%d, %d)" % (*centroid,),
                        (abs(centroid[0] - 25), abs(centroid[1] - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(res1, 'Press G to toggle greyscale canny', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # pylint: disable = protected-access
        cv2.putText(frame,
                    datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")+" Frame %d" %
                    fps._numFrames, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Original', frame)
        cv2.imshow('Explorer', res1)

        # Keyboard OP
        k = cv2.waitKey(5) & 0xFF
        if k == 27:               # Esc
            break
        elif k == ord('g'):       # toggle gray
            GRAY = not GRAY


        fps.update()


    # Stops video stream, calculates FPS and does some clean ups
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    stream.stop()
    cv2.destroyAllWindows()
