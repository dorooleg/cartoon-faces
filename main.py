#!/usr/bin/env python3
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

xs = [142, 138, 131, 128, 130, 144, 157, 180, 203, 226, 255, 286, 313, 325, 332, 339, 345, 153, 177, 195, 217, 231, 264, 294, 315, 338, 350, 241, 238, 237, 227, 202, 210, 221, 229, 241, 160, 184, 211, 219, 194, 169, 266, 287, 314, 333, 318, 290, 163, 182, 203, 218, 237, 254, 271, 255, 237, 208, 187, 177, 169, 194, 217, 231, 249, 232, 217, 196]
ys = [201, 222, 248, 281, 304, 346, 356, 381, 391, 390, 375, 353, 331, 315, 293, 267, 250, 170, 164, 169, 182, 200, 209, 201, 199, 201, 215, 224, 247, 267, 288, 297, 303, 307, 309, 306, 205, 194, 207, 233, 236, 230, 249, 228, 226, 245, 265, 263, 306, 311, 316, 321, 322, 325, 329, 346, 359, 358, 346, 334, 321, 322, 343, 242, 339, 336, 337, 332]



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-r', '--picamera', type=int, default=-1,
                help='whether or not the Raspberry Pi camera should be used')
args = vars(ap.parse_args())




mermaid_img = cv2.imread('./data/mermaid.png', cv2.IMREAD_UNCHANGED)
print(mermaid_img)




# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# initialize the video stream and allow the cammera sensor to warmup
print('[INFO] camera sensor warming up...')
vs = VideoStream(usePiCamera=args['picamera'] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        # rows,cols,channels = frame.shape
        # roi = mermaid_img[0:rows, 0:cols ]
        # # Now create a mask of logo and create its inverse mask also
        # framegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(framegray, 10, 255, cv2.THRESH_BINARY)
        # mask_inv = cv2.bitwise_not(mask)
        # # Now black-out the area of logo in ROI
        # mermaid_img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # # Take only region of logo from logo image.
        # frame_fg = cv2.bitwise_and(frame,frame,mask = mask)
        # # Put logo in ROI and modify the main image
        # dst = cv2.add(mermaid_img_bg,frame_fg)
        # mermaid_img[0:rows, 0:cols ] = dst

        mermaid_img = cv2.resize(mermaid_img, (frame.shape[1], frame.shape[0]))
        print(mermaid_img.shape, frame.shape)

        # alpha = 0.5
        # beta = ( 1.0 - alpha );
        # cv2.addWeighted(frame, alpha, mermaid_img, beta, 0.0, frame);

        b_channel, g_channel, r_channel, alpha = cv2.split(mermaid_img)
        b_channel2, g_channel2, r_channel2 = cv2.split(frame)

        mm = cv2.merge((b_channel, g_channel, r_channel, alpha))
        nn = cv2.merge((b_channel2, g_channel2, r_channel2, 1.0 - alpha))
        frame = nn + mm



        # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
        # frame = mermaid_img * alpha + (1 - alpha) * frame

        # frame = cv2.merge((b_channel, g_channel, r_channel, 1.0 -alpha_channel))


        # img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))



        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for i, (x, y) in enumerate(shape):
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            # cv2.putText(frame, str(x) + ',' + str(y), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
            # cv2.circle(memaid_img, (xs[i], ys[i]), 1, (0, 0, 255), -1)
            # cv2.putText(memaid_img, str(xs[i]) + ',' + str(ys[i]), (xs[i],ys[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)





    # show the frame
    cv2.imshow('Frame', frame)
    # cv2.imshow('Frame', memaid_img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()