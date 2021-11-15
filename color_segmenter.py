#!/usr/bin/env python3

# ------------------------------------------------------------
# Library's import
# ------------------------------------------------------------
import cv2
import copy
import argparse
import numpy as np
import json
import pprint
from colorama import Fore, Back, Style

# ------------------------------------------------------------
# Variables initialization
# ------------------------------------------------------------
# window_name_video = 'Video'
# window_name_mask = 'Mask'
window_name_segmentation = 'Segmentation'                                                                               # Set window name.
tkb_Names = ['min B', 'max B', 'min G', 'max G', 'min R', 'max R']                                                      # Variable with trackbars names (according to RGB).
tkb_max_value = 256                                                                                                     # Set trackbar maximum value.
tkb_min_init_value = 100                                                                                                # Set trackbar minimum initial value.


# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------
def on_min_B_value_trackbar(val):
    max_B_H_value = cv2.getTrackbarPos(tkb_Names[1], window_name_segmentation)                                          # Get trackbar position (Blue/Hue maximum value).
    min_B_H_value = min(max_B_H_value - 1, val)                                                                         # Set minimum Blue/Hue value allowed.
    cv2.setTrackbarPos(tkb_Names[0], window_name_segmentation, min_B_H_value)                                           # Set trackbar position (Blue/Hue minimum value).


def on_max_B_value_trackbar(val):
    min_B_H_value = cv2.getTrackbarPos(tkb_Names[0], window_name_segmentation)                                          # Get trackbar position (Blue/Hue minimum value).
    max_B_H_value = max(val, min_B_H_value + 1)                                                                         # Set maximum Blue/Hue value allowed.
    cv2.setTrackbarPos(tkb_Names[1], window_name_segmentation, max_B_H_value)                                           # Set trackbar position (Blue/Hue maximum value).


def on_min_G_value_trackbar(val):
    max_G_S_value = cv2.getTrackbarPos(tkb_Names[3], window_name_segmentation)                                          # Get trackbar position (Green/Saturation maximum value).
    min_G_S_value = min(max_G_S_value - 1, val)                                                                         # Set minimum Green/Saturation value allowed.
    cv2.setTrackbarPos(tkb_Names[2], window_name_segmentation, min_G_S_value)                                           # Set trackbar position (Green/Saturation minimum value).


def on_max_G_value_trackbar(val):
    min_G_S_value = cv2.getTrackbarPos(tkb_Names[2], window_name_segmentation)                                          # Get trackbar position (Green/Saturation minimum value).
    max_G_S_value = max(val, min_G_S_value + 1)                                                                         # Set maximum Green/Saturation value allowed.
    cv2.setTrackbarPos(tkb_Names[3], window_name_segmentation, max_G_S_value)                                           # Set trackbar position (Green/Saturation maximum value).


def on_min_R_value_trackbar(val):
    max_R_V_value = cv2.getTrackbarPos(tkb_Names[5], window_name_segmentation)                                          # Get trackbar position (Red/Value maximum value).
    min_R_V_value = min(max_R_V_value - 1, val)                                                                         # Set minimum Red/Value value allowed.
    cv2.setTrackbarPos(tkb_Names[4], window_name_segmentation, min_R_V_value)                                           # Set trackbar position (Red/Value minimum value).


def on_max_R_value_trackbar(val):
    min_R_V_value = cv2.getTrackbarPos(tkb_Names[4], window_name_segmentation)                                          # Get trackbar position (Red/Value minimum value).
    max_R_V_value = max(val, min_R_V_value + 1)                                                                         # Set maximum Red/Value value allowed.
    cv2.setTrackbarPos(tkb_Names[5], window_name_segmentation, max_R_V_value)                                           # Set trackbar position (Red/Value maximum value).


# def get_contour_areas(contours):
#     all_areas = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         all_areas.append(area)
#
#     return all_areas


def testDevice(capture, source):
    if capture is None or not capture.isOpened():                                                                                           # Check if camera index it's valid.
        print(Fore.YELLOW + Style.BRIGHT + 'Color segmentation Finished. Unable to open video source: ' + str(source) + Style.RESET_ALL)    # Program finished message.
        exit()                                                                                                                              # Stops the program.


def main():
    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    global tkb_Names

    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('-cn', '--camera_number', type=int, help='Camera number (Default = 0).', default=0)
    parser.add_argument('-RGB', '--RGB_segmentation', action='store_true', help='Type of segmentation (RGB or HSV) (Default = HSV).', default=False)
    args = vars(parser.parse_args())

    if args.get('camera_number') < 0:                                                                                   # Check if 'camera_number' input is valid.
        print(Fore.RED + Style.BRIGHT + 'error: Invalid input argument!' + Style.RESET_ALL)                             # Error message.
        exit()                                                                                                          # Stops the program.

    capture = cv2.VideoCapture(args.get('camera_number'))                                                               # Setup video capture for camera.

    testDevice(capture, args.get('camera_number'))                                                                      # Call 'testDevice' function to check if selected camera is available.

    # cv2.namedWindow(window_name_video, cv2.WINDOW_AUTOSIZE)                                                           # Window Setup.
    # cv2.namedWindow(window_name_mask, cv2.WINDOW_AUTOSIZE)                                                            # Window Setup.
    cv2.namedWindow(window_name_segmentation, cv2.WINDOW_AUTOSIZE)                                                      # Window Setup.

    image_channels = ['B', 'G', 'R']                                                                                    # Variable with image channels leters (RGB).

    if not args.get('RGB_segmentation'):                                                                                # Check if user select HSV or RGB segmentation type (Default = HSV).
        tkb_Names = ['min H', 'max H', 'min S', 'max S', 'min V', 'max V']                                              # Variable with trackbars names (according to HSV).
        image_channels = ['H', 'S', 'V']                                                                                # Variable with image channels leters (HSV).

    cv2.createTrackbar(tkb_Names[0], window_name_segmentation, tkb_min_init_value, tkb_max_value, on_min_B_value_trackbar)        # Create trackbars (Minimum Blue/Hue color).
    cv2.createTrackbar(tkb_Names[1], window_name_segmentation, tkb_max_value, tkb_max_value, on_max_B_value_trackbar)             # Create trackbars (Maximum Blue/Hue color).
    cv2.createTrackbar(tkb_Names[2], window_name_segmentation, tkb_min_init_value, tkb_max_value, on_min_G_value_trackbar)        # Create trackbars (Minimum Green/Saturation color).
    cv2.createTrackbar(tkb_Names[3], window_name_segmentation, tkb_max_value, tkb_max_value, on_max_G_value_trackbar)             # Create trackbars (Maximum Green/Saturation color).
    cv2.createTrackbar(tkb_Names[4], window_name_segmentation, tkb_min_init_value, tkb_max_value, on_min_R_value_trackbar)        # Create trackbars (Minimum Red/Value color).
    cv2.createTrackbar(tkb_Names[5], window_name_segmentation, tkb_max_value, tkb_max_value, on_max_R_value_trackbar)             # Create trackbars (Maximum Red/Value color).

    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------
    while True:
        _, image = capture.read()                                                                                       # Get an image from the camera and store them at "image" variable.
        if image is None:                                                                                               # Check if there are no camera image.
            print(Fore.YELLOW + Style.BRIGHT + 'Video is over, terminating.' + Style.RESET_ALL)                         # Test finished message.
            break                                                                                                       # Break/Stops the loop.

        # B, G, R = cv2.split(image)                                                                                    # Color segmentation.

        if not args.get('RGB_segmentation'):                                                                            # Check if user select HSV or RGB segmentation type (Default = HSV).
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                                                              # Convert image to HSV.

        B_H_min = cv2.getTrackbarPos(tkb_Names[0], window_name_segmentation)                                            # Get trackbars positions (Minimum Blue/Hue color).
        B_H_max = cv2.getTrackbarPos(tkb_Names[1], window_name_segmentation)                                            # Get trackbars positions (Maximum Blue/Hue color).
        G_S_min = cv2.getTrackbarPos(tkb_Names[2], window_name_segmentation)                                            # Get trackbars positions (Minimum Green/Saturation color).
        G_S_max = cv2.getTrackbarPos(tkb_Names[3], window_name_segmentation)                                            # Get trackbars positions (Maximum Green/Saturation color).
        R_V_min = cv2.getTrackbarPos(tkb_Names[4], window_name_segmentation)                                            # Get trackbars positions (Minimum Red/Value color).
        R_V_max = cv2.getTrackbarPos(tkb_Names[5], window_name_segmentation)                                            # Get trackbars positions (Maximum Red/Value color).

        ranges = {image_channels[0]: {'max': B_H_max, 'min': B_H_min},                                                  # Dictionary to store minimum and maximum RGB/HSV color values (Blue/Hue).
                  image_channels[1]: {'max': G_S_max, 'min': G_S_min},                                                  # Dictionary to store minimum and maximum RGB/HSV color values (Green/Saturation).
                  image_channels[2]: {'max': R_V_max, 'min': R_V_min}}                                                  # Dictionary to store minimum and maximum RGB/HSV color values (Red/Value).

        mins = np.array([ranges[image_channels[0]]['min'], ranges[image_channels[1]]['min'], ranges[image_channels[2]]['min']])     # Gets minimum RGB/HSV color values from dictionary.
        maxs = np.array([ranges[image_channels[0]]['max'], ranges[image_channels[1]]['max'], ranges[image_channels[2]]['max']])     # Gets maximum RGB/HSV color values from dictionary.
        image_processed = cv2.inRange(image, mins, maxs)                                                                            # Process original image/video according to RGB/HSV color values range.

        # height, width, _ = image.shape                                                                                  # Get image dimensions.
        # mask = np.ndarray((height, width), dtype=np.uint8)
        # mask.fill(0)
        #
        # # Contour detection and isolation of biggest contour + fill
        # if np.mean(image_processed) > 0:
        #
        #     contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        #     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #
        #     largest_item = sorted_contours[0]
        #     cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))
        #     #cv2.drawContours(mask, largest_item, -1, (255, 0, 0), -1)
        #
        #     #Centroid coordinates calculation + draw
        #     M = cv2.moments(mask)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     #cv2.circle(mask, (cX, cY), 5, (0, 0, 0), -1)
        #     cv2.circle(image, (cX, cY), 5, (255, 0, 255), -1)
        #
        # # cv2.imshow(window_name_video, image)  # Mostra a janela com o video
        # # cv2.imshow(window_name_mask, mask)
        cv2.imshow(window_name_segmentation, image_processed)                                                           # Display the processed image/video.

        # ------------------------------------------------------------
        # TERMINATION
        # ------------------------------------------------------------
        key = cv2.waitKey(20)

        if (key == ord('q')) or (key == ord('Q')) or (cv2.getWindowProperty(window_name_segmentation, 1) == -1):        # Check if user pressed the 'q' key or closed the window.
            print(Fore.YELLOW + Style.BRIGHT + 'Color segmentation Finished without store data in json file.' + Style.RESET_ALL)  # Program finished message.
            exit()                                                                                                      # Stops the program.
        elif (key == ord('w')) or (key == ord('W')):                                                                                           # Check if user pressed the 'w' key.
            dict_result = {'limits': ranges}                                                                            # Creation of the dictionary.

            file_name = 'limits.json'                                                                                   # Creation of .json file.
            print('\n============================= Results =============================\n')                            # Results message.
            json.dump(dict_result, open(file_name, 'w'))                                                                # Save results at .json file.
            pp = pprint.PrettyPrinter(indent=1)                                                                         # Set the dictionary initial indentation.
            pp.pprint(dict_result)                                                                                      # Print the entire dictionary.
            break                                                                                                       # Break/Stops the loop.


if __name__ == '__main__':
    main()
