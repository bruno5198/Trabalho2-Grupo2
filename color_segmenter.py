#!/usr/bin/env python3

import cv2
import copy
import numpy as np

def on_trackbar(threshold):

    pass


def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas


def main():

    #Camera Setup
    capture = cv2.VideoCapture(0)

    #Window Setup
    window_name_video = 'Video'
    window_name_segmentation = 'Segmentation'
    window_name_mask = 'Mask'

    cv2.namedWindow(window_name_video,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_segmentation, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_mask, cv2.WINDOW_AUTOSIZE)


    # Create trackbars
    cv2.createTrackbar('B_min', window_name_segmentation, 0, 256, on_trackbar)
    cv2.createTrackbar('B_max', window_name_segmentation, 0, 256, on_trackbar)
    cv2.createTrackbar('G_min', window_name_segmentation, 0, 256, on_trackbar)
    cv2.createTrackbar('G_max', window_name_segmentation, 0, 256, on_trackbar)
    cv2.createTrackbar('R_min', window_name_segmentation, 0, 256, on_trackbar)
    cv2.createTrackbar('R_max', window_name_segmentation, 0, 256, on_trackbar)


    while True:
        _,image = capture.read() #O video é guardado na variavel image
        if image is None:
            print('Video is over, terminating')
            break



        key = cv2.waitKey(20) #

        #Segmentação da cor
        B, G, R = cv2.split(image)

        #Get trackbars positions

        b_min = cv2.getTrackbarPos('B_min', window_name_segmentation)
        b_max = cv2.getTrackbarPos('B_max', window_name_segmentation)
        g_min = cv2.getTrackbarPos('G_min', window_name_segmentation)
        g_max = cv2.getTrackbarPos('G_max', window_name_segmentation)
        r_min = cv2.getTrackbarPos('R_min', window_name_segmentation)
        r_max = cv2.getTrackbarPos('R_max', window_name_segmentation)

        ranges = {'b': {'min': b_min, 'max': b_max},
                  'g': {'min': g_min, 'max': g_max},
                  'r': {'min': r_min, 'max': r_max}}

        # Process image
        mins = np.array([ranges['b']['min'], ranges['g']['min'], ranges['r']['min']])
        max = np.array([ranges['b']['max'], ranges['g']['max'], ranges['r']['max']])
        image_processed = cv2.inRange(image, mins, max)

        height, width, _ = image.shape
        mask = np.ndarray((height, width), dtype=np.uint8)
        mask.fill(0)

        #Contour detection and isolation of biggest contour + fill
        if np.mean(image_processed) > 0:

            contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            largest_item = sorted_contours[0]
            cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))
            #cv2.drawContours(mask, largest_item, -1, (255, 0, 0), -1)

            #Centroid coordinates calculation + draw
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #cv2.circle(mask, (cX, cY), 5, (0, 0, 0), -1)
            cv2.circle(image, (cX, cY), 5, (255, 0, 255), -1)

        cv2.imshow(window_name_video, image)  # Mostra a janela com o video
        cv2.imshow(window_name_mask, mask)
        cv2.imshow(window_name_segmentation, image_processed)  # Mostra a janela com o video segmentado

        if key == ord('q'): #q for quit
            print('You pressed q, aborting')
            break




if __name__ == '__main__':
    main()