#!/usr/bin/python3

import argparse
import copy

import cv2
from colorama import Fore, Back, Style
import numpy as np
import json
import datetime


def main():
    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('-j', '--json', help='Full path to json file.')


    args = vars(parser.parse_args())

    capture = cv2.VideoCapture(0)                                                                                       # Setup video capture for camera.

    window_name_segmentation = 'Segmentation'                                                                           # Set window name.

    cv2.namedWindow(window_name_segmentation, cv2.WINDOW_AUTOSIZE)                                                      # Window Setup.

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))                                                                  # Get image dimensions (width).
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))                                                                # Get image dimensions (height).

    white_window = np.full((height, width, 3), 255, dtype=np.uint8)                                                     # Create a white image with the same size as the original image/video.

    pencil_color = (255, 0, 255)                                                                                        # Set pencil default color.
    pencil_dimension = 10                                                                                               # Set pencil default dimension.

    last_coordinates = ''                                                                                               # Initialization of an auxiliary variable.
    all_coordinates = []
    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------

    while True:
        _, image = capture.read()
        image=cv2.flip(image,1)                                                                                         # Get an image from the camera and store them at "image" variable.
        image_raw=copy.copy(image)                                                                                      #Do a copy of image for show the original
        if image is None:                                                                                               # Check if there are no camera image.
            print(Fore.YELLOW + Style.BRIGHT + 'Video is over, terminating.' + Style.RESET_ALL)                         # Test finished message.
            break                                                                                                       # Break/Stops the loop.

        data = json.load(open(args.get('json')))                                                                        # Get .json file data and store them at 'data' variable.

        mins = np.array([data['limits']['B']['min'], data['limits']['G']['min'], data['limits']['R']['min']])           # Gets minimum RGB/HSV color values from data variable.
        maxs = np.array([data['limits']['B']['max'], data['limits']['G']['max'], data['limits']['R']['max']])           # Gets maximum RGB/HSV color values from data variable.

        image_processed = cv2.inRange(image, mins, maxs)                                                                # Process original image/video according to RGB/HSV color values range.

        mask = np.ndarray((height, width), dtype=np.uint8)                                                              # Create a mask with the same size as image.
        mask.fill(0)                                                                                                    # Fill the mask with white.

        # Contour detection and isolation of biggest contour + fill
        if np.mean(image_processed) > 0:
            contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)           # Get "image_processed" external contour.
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)                                       # Get contour area.
            largest_item = sorted_contours[0]                                                                           # Get largest item/contour.
            cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))                                               # Fill contour with white color.

            M = cv2.moments(mask)                                                                                       # Centroid coordinates calculation.
            cX = int(M["m10"] / M["m00"])                                                                               # Centroid coordinates calculation.
            cY = int(M["m01"] / M["m00"])                                                                               # Centroid coordinates calculation.
            # cv2.circle(white_window, (cX, cY), pencil_dimension, pencil_color, -1)

            if last_coordinates == '':                                                                                      # Condition to execute the next command only once.
                last_coordinates = (cX, cY)                                                                                 # Save first centroid coordinates at "last_coordinates" variable.
            #cv2.line(white_window , (last_coordinates[0], last_coordinates[1]), (cX, cY), pencil_color, pencil_dimension)          # Draw a line.
            last_coordinates = (cX, cY)                                                                                     # Save last centroid coordinates.
            all_coordinates.append(last_coordinates)                                                                                      #create a list with all points
        # ------------------------------------------------------------
        # TERMINATION
        # ------------------------------------------------------------
        key = cv2.waitKey(20)

        if (key == ord('q')) or (key == ord('Q')) or (cv2.getWindowProperty(window_name_segmentation, 1) == -1):        # Check if user pressed the 'q' key or closed the window.
            print(Fore.YELLOW + Style.BRIGHT + 'Ar Paint Finished.' + Style.RESET_ALL)                                  # Program finished message.
            exit()                                                                                                      # Stops the program.
        elif (key == ord('r')) or (key == ord('Q')):                                                                    # Check if user pressed the 'r' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil color change to Red.' + Style.RESET_ALL)                         # Pencil color changed message.
            pencil_color = (0, 0, 255)                                                                                  # Change pencil color to red.
        elif (key == ord('g')) or (key == ord('G')):                                                                    # Check if user pressed the 'g' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil color change to Green.' + Style.RESET_ALL)                       # Pencil color changed message.
            pencil_color = (0, 255, 0)                                                                                  # Change pencil color to green.
        elif (key == ord('b')) or (key == ord('B')):                                                                    # Check if user pressed the 'b' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil color change to Blue.' + Style.RESET_ALL)                        # Pencil color changed message.
            pencil_color = (255, 0, 0)                                                                                  # Change pencil color to blue.
        elif key == ord('+'):                                                                                           # Check if user pressed the '+' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil size change (bigger).' + Style.RESET_ALL)                        # Pencil size changed message.
            pencil_dimension = pencil_dimension + 1                                                                     # Increase pencil dimension.
        elif (key == ord('-')) and (pencil_dimension >= 1):                                                             # Check if user pressed the '-' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil size change (smaller).' + Style.RESET_ALL)                       # Pencil size changed message.
            pencil_dimension = pencil_dimension - 1                                                                     # Decrease pencil dimension.
        elif (key == ord('c')) or (key == ord('C')):                                                                    # Check if user pressed the 'c' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Image cleared.' + Style.RESET_ALL)                                      # Image cleared message.
            white_window = np.full((height, width, 3), 255, dtype=np.uint8)                                             # Clear image.
        elif (key == ord('w')) or (key == ord('W')):                                                                    # Check if user pressed the 'w' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Image saved as PNG.' + Style.RESET_ALL)                                 # Image saved message.
            current_day_of_week = datetime.datetime.now().strftime("%a")                                                # Get current week day.
            month_number = str(datetime.datetime.today().month)                                                         # Get current month number.
            current_month = datetime.datetime.strptime(month_number, "%m").strftime("%b")                               # Tranform current month number to name/text.
            current_time = datetime.datetime.now().strftime("%d_%H:%M:%S_%Y")                                           # Get current day of month, time and year.
            date_format = str(current_day_of_week) + '_' + str(current_month) + '_' + str(current_time)                 # Concatenate all date parameters.
            cv2.imwrite('drawing_' + date_format + '.png', image)                                                       # Save image as png.


        for i in range(2, len(all_coordinates)):
            cv2.line(image,all_coordinates[i], all_coordinates[i-1], pencil_color, pencil_dimension)                    #draw in image
            cv2.line(white_window, all_coordinates[i], all_coordinates[i - 1], pencil_color, pencil_dimension)          #draw in white board

        cv2.imshow('Original Video Image',image_raw)                                                                    #show original image
        cv2.imshow('white_window', white_window)                                                                        # Display the white window.
        cv2.imshow(window_name_segmentation, image)                                                                     # Display the original image/video.


        #cv2.imshow('white_window', white_window)


if __name__ == '__main__':
    main()