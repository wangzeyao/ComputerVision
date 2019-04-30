"""
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
"""
from __future__ import print_function
# Python 2/3 compatibility
import numpy as np

import cv2 as cv

from common import clock, draw_str
# local modules
from video import create_capture


# the function which get the rectangle of the detected face
def detect(img, cascade):
    rects = cascade.detectMultiScale(img,
                                     scaleFactor=1.3,
                                     minNeighbors=4,
                                     minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)  # using cascade function to found the
    # face for the image
    if len(rects) == 0:  # if no face detected return rects as a vide list
        return []
    rects[:, 2:] += rects[:, :2]  # if face detected , transfer the value in the format x1,y1,x2,y2
    return rects


# function to draw the rectangle
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


if __name__ == '__main__':
    detected = np.array([0, 0, 0, 0])  # when a face detected store the x and y for the sub rectangle we use
    # in a list inside an array
    threshold = 3000  # we introduce the total time and threshold to control the process,since we only detect the face
    # in the sub rectangle,we add a threshold(in ms) that if the total time is greater than the threshold,
    # we stop detecting face in the sub image(sub rectangle) and use the full image to detect the face
    total_time = 0.0
    import sys
    import getopt

    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '',
                                    ['cascade=', 'nested-cascade='])  # Parses command line options and parameter list
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)  # make args as dictionary
    cascade_fn = args.get('--cascade',
                          "data/haarcascades/haarcascade_frontalface_alt.xml")  # get trained face detected data
    nested_fn = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))  # Loads the classifier from a file.
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src,  # create_capture is a convenience function for capture creation,
                         # falling back to procedural video in case of error.
                         fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))

    while True:
        t = clock()  # start to count the time
        time = 0.0  # initialize the time counting
        ret, img = cam.read()  # capture a frame and store it
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the color of the image from bgr to gray
        gray = cv.equalizeHist(gray)  # Equalizes the histogram of a grayscale image.
        vis = img.copy()  # make a copy of the image
        if detected.all() == 0 or total_time > 3000:  # if we haven't detected a face or can't find face in
            # the current sub rectangle more than 3000ms,start to detect in the full image
            rects = detect(gray, cascade)
            if len(rects):  # if find a face, create the sub rectangle by add 50 pixel for each x and y
                # and initialize the total time
                detected = rects[0]
                detected = np.array([detected[0] - 50, detected[1] - 50, detected[2] + 50, detected[3] + 50])
                total_time = 0.0
        else:  # if we have a sub rectangle now and the total time is under 3000ms,
            # we detect the face in the sub rectangle
            sub_gray = gray[detected[1]:detected[3], detected[0]:detected[2]]  # restrict the area in the gray image
            sub_vis = vis[detected[1]:detected[3], detected[0]:detected[2]]  # create the data for drawing the rectangle
            sub_rects = detect(sub_gray.copy(), cascade)  # do the detection in the sub rectangle
            if len(sub_rects):  # if face detected, calculate the time we used and initialize the total time
                time = clock() - t
                total_time = 0.0
            draw_rects(sub_vis, sub_rects, (0, 255, 0))  # draw the rectangle for the detected face
        if not nested.empty():  # the same for the eye detection
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                vis_rects = detect(roi.copy(), nested)
                draw_rects(vis_roi, vis_rects, (255, 0, 0))
        cv.rectangle(vis, (detected[0], detected[1]), (detected[2], detected[3]), (0, 255, 255),
                     2)  # draw the sub rectangle
        if time == 0.0:  # the time equals to 0 which means no face has been detected
            draw_str(vis, (20, 20), 'No face detected')
            time = (clock() - t) * 1000
            total_time = total_time + time  # update the total time for detect a face
        else:  # face detected show the time we use to detect the face in ms
            draw_str(vis, (20, 20), 'time: %.1f ms' % (time * 1000))
        cv.imshow('facedetect', vis)
        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
