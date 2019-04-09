#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import getopt
import numpy as np
import cv2 as cv
import video
from video import presets
import os
import letter_recog_NN as lcn

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


class FaceDetect:
    def __init__(self):
        self.track_window = None
        self.cascade = None

    def initialization(self):  # setup for using the haar classifer
        args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
        args = dict(args)
        cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
        self.cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))

    def detect(self, img):  # the function which get the rectangle of the detected face
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                              flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects


class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets[
            'cube'])  # create_capture is a convenience function for capture creation,
        _ret, self.frame = self.cam.read()  # capture a frame and store it
        cv.namedWindow('camshift')

        self.selection = None
        self.show_backproj = False
        self.face_find = False
        self.track_window = None

    def show_hist(self):  # show the plot Of histogram
        bin_count = self.hist.shape[0]  # get the number of bins
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                         (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def writeTxt(self,letter,pic):
        file = open('D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt', 'a')
        samples = pic.reshape((1, 256))
        file.write(letter+',')
        for i in range(256):
            if i == 255:
                file.write(str(samples[0][i]))
            else:
                file.write(str(samples[0][i]) + ',')
        file.write('\n')
        file.close()
        print(letter,' writed')

    def run(self):  # main loop
        face_detect = FaceDetect()  # i put the creation of the object and the initialization
        # outside the loop to save time and memory
        face_detect.initialization()
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)  # convert from BGR to HSV
            mask = cv.inRange(hsv,
                              np.array((0., 60., 32.)),  # lower bound
                              np.array((180., 255., 255.)))  # upper bound
            rects = face_detect.detect(self.frame)
            if len(rects):  # face detected, set values of selection and track window
                self.selection = rects[0]
                xmin = min(self.selection[0], self.selection[2])
                xmax = max(self.selection[0], self.selection[2])
                ymin = min(self.selection[1], self.selection[3])
                ymax = max(self.selection[1], self.selection[3])
                self.selection = (xmin, ymin, xmax, ymax)
                x0, y0, x1, y1 = self.selection

            if self.selection and not self.face_find:  # we don't want to change the histogram after the fist detection
                # of a face, so the code below loop will run only once
                self.face_find = True
                self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist([hsv_roi],  # the image
                                   [0],  # channels used for calculate the histogram,[1] for gray scale.
                                   # 用于计算直方图的通道，灰度图是[1]，
                                   # colored image can use [0],[1],[2] to calculate blue, green and red
                                   # 彩色图可以用[0],[1],[2]来分别计算蓝，绿，红
                                   mask_roi,  # mask 掩图，想找到特定区域的histogram时使用
                                   [16],  # hitsize BIN的数量，表示直方图分成多少分（即有多少个柱子）
                                   [0, 180])  # ??
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)  # Inverts every bit of an array(why?)
                vis[mask == 0] = 0  # wherever the value in mask equals 0, set 0 for vis.(why?)

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv],  # image
                                          [0],  # channels
                                          self.hist,  # input histogram
                                          [0, 180],  # Destination back projection array that is a single-channel
                                          #  array of the same size and depth as images
                                          1)
                prob[y0-30:y1+30, x0-10:x1+10] = 0  # set the probability of detected face to 0
                prob &= mask  # 按位运算符
                term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)  # end criteria for camshift
                track_box, self.track_window = cv.CamShift(prob,  # Back projection of the object histogram
                                                           self.track_window,  # Initial search window
                                                           term_crit)
                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]
                try:
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)
                    cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
                except:
                    print(track_box)

            cv.imshow('camshift', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
            if ch == ord('c'):
                xs0, ys0, xs1, ys1 = self.track_window
                prob_hand = prob[ys0:ys1 + ys0, xs0:xs1 + xs0]  # get back project image
                small_pic = cv.resize(prob_hand, dsize=(16, 16))  # resize it to 16x16
                large_pic = cv.resize(prob_hand, dsize=(224, 224))  # resize it to 16x16
                cv.imwrite('D:/pythonProject/ComputerVision/TP2/files/handphoto/c.png',large_pic)
                self.writeTxt(letter='C',pic=small_pic)
            elif ch == ord('i'):
                xs0, ys0, xs1, ys1 = self.track_window
                prob_hand = prob[ys0:ys1 + ys0, xs0:xs1 + xs0]  # get back project image
                small_pic = cv.resize(prob_hand, dsize=(16, 16))  # resize it to 16x16
                large_pic = cv.resize(prob_hand, dsize=(224, 224))  # resize it to 16x16
                self.writeTxt(letter='I',pic=small_pic)
                cv.imwrite('D:/pythonProject/ComputerVision/TP2/files/handphoto/i.png',large_pic)
            elif ch == ord('o'):
                xs0, ys0, xs1, ys1 = self.track_window
                prob_hand = prob[ys0:ys1 + ys0, xs0:xs1 + xs0]  # get back project image
                small_pic = cv.resize(prob_hand, dsize=(16, 16))  # resize it to 16x16
                large_pic = cv.resize(prob_hand, dsize=(224, 224))  # resize it to 16x16
                self.writeTxt(letter='O',pic=small_pic)
                cv.imwrite('D:/pythonProject/ComputerVision/TP2/files/handphoto/o.png',small_pic)
            elif ch == ord('v'):
                xs0, ys0, xs1, ys1 = self.track_window
                prob_hand = prob[ys0:ys1 + ys0, xs0:xs1 + xs0]  # get back project image
                small_pic = cv.resize(prob_hand, dsize=(16, 16))  # resize it to 16x16
                large_pic = cv.resize(prob_hand, dsize=(224, 224))  # resize it to 16x16
                self.writeTxt(letter='V',pic=small_pic)
                cv.imwrite('D:/pythonProject/ComputerVision/TP2/files/handphoto/v.png',small_pic)
            elif ch == ord('p'):
                xs0, ys0, xs1, ys1 = self.track_window
                small_pic = cv.resize(prob[ys0:ys1 + ys0, xs0:xs1 + xs0], dsize=(16, 16))
                small_pic = small_pic.reshape((1, 256))
                small_pic = small_pic.astype(np.float32)
                print(lcn.prediction(small_pic))


        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
