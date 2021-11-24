#!/usr/bin/python3

import os
import random
import argparse
import copy
import math
import cv2
from colorama import Fore, Back, Style
import numpy as np
import json
import datetime
import colorsys


def moments_calc(mask):
    M = cv2.moments(mask)                                                                                               # Centroid coordinates calculation.
    cX = int(M["m10"] / M["m00"])                                                                                       # Centroid coordinates calculation.
    cY = int(M["m01"] / M["m00"])                                                                                       # Centroid coordinates calculation.
    return cX, cY


class adc_func_4_5(object):
    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def __init__(self, jsonFile, imageFile, pencilThickness):
        self.image_clear = ""
        self.num_contour = []                                                                                           # Variable initialization.
        self.num_white_pix_no_num = []                                                                                  # Variable initialization.
        self.contour_content = []                                                                                       # Variable initialization.
        self.jsonFile = jsonFile                                                                                        # Variable initialization.
        self.original_image = imageFile                                                                                 # Variable initialization.
        self.thickness = pencilThickness                                                                                # Variable initialization. Pencil default thickness.
        self.aux_image = ""                                                                                             # Variable initialization.
        self.mouseON = True                                                                                             # Variable initialization. To know if user can draw with mouse or not.
        self.aux_image2 = ""                                                                                            # Variable initialization.
        self.last_coordinates = ""                                                                                      # Variable initialization. Variable to save last coordinate.
        self.pencil_color = (255, 0, 0)                                                                                 # Variable initialization. Pencil default color.
        self.window_name = 'Advanced Functionality 4 and 5'                                                             # Variable initialization. Window default name.
        self.af4_image_width = 0                                                                                        # Variable initialization. Image width default value.
        self.af4_image_height = 0                                                                                       # Variable initialization. Image height default value.

        if self.original_image is None:                                                                                 # Condition to check problems with default image path.
            print('\n' + Fore.RED + Style.BRIGHT + 'Error!' + Style.RESET_ALL + ' Check default image path (' + self.original_image + '!' + '\n')  # Default image path error message.
            exit()                                                                                                      # Stops the program.

        if os.path.isfile(self.original_image) is False:                                                                # Condition to check problems with user defined image path.
            print('\n' + Fore.RED + Style.BRIGHT + 'Error!' + Style.RESET_ALL + ' Check inserted image path (' + self.original_image + ')!' + '\n')  # User defined image path error message.
            exit()                                                                                                      # Stops the program.

        print('\n========== PSR Ar Paint - Advanced Funcionality 4 and 5(Grupo 2) ==========\n')                                                        # Initial message.
        print('=> Program initial conditions.')
        print('    => Image path ("-if IMAGEFILE" or "--imageFile IMAGEFILE" arguments to change):                          ' + self.original_image)    # Initial message.
        print('    => Pencil thickness ("-pf PENCILTHICKNESS" or "--pencilThickness PENCILTHICKNESS" arguments to change):  ' + str(self.thickness))    # Initial message.
        print('    => Initial pencil color:                                                                                 Blue\n')                    # Initial message.
        print('    => By default your drawing with:                                                                         Mouse\n')                   # Initial message.
        print('=> You must paint image regions with the correct color. (Image: ' + self.original_image + ')')                                           # Preliminary notes.
        print('    => Region Number 1 -> ' + Fore.BLUE + Style.BRIGHT + 'Blue' + Style.RESET_ALL + ' Color.')                                           # Preliminary notes.
        print('    => Region Number 2 -> ' + Fore.GREEN + Style.BRIGHT + 'Green' + Style.RESET_ALL + ' Color.')                                         # Preliminary notes.
        print('    => Region Number 3 -> ' + Fore.RED + Style.BRIGHT + 'Red' + Style.RESET_ALL + ' Color.')                                             # Preliminary notes.
        print('\n=> Keys that you can press.')                                                                          # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"f" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"F" ' + Style.RESET_ALL + 'key -> Evaluate your paint accuracy!')                                                 # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"q" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"Q" ' + Style.RESET_ALL + 'key -> Abort!')                                                                        # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"c" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"C" ' + Style.RESET_ALL + 'key -> Clear image!')                                                                        # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"a" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"A" ' + Style.RESET_ALL + 'key -> Start drawing with camera color detection!')                                    # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"m" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"M" ' + Style.RESET_ALL + 'key -> Start drawing with mouse! (By default)')                                        # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"+" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"-" ' + Style.RESET_ALL + 'key -> Increase/decrease pencil thickness')                                                            # Preliminary notes
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"b" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"B" ' + Style.RESET_ALL + 'key -> Change pencil color to Blue! (By default)')                                     # Preliminary notes.
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"g" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"G" ' + Style.RESET_ALL + 'key -> Change pencil color to Green!')                                                 # Preliminary notes.
        print('    => ' + Fore.YELLOW + Style.BRIGHT + '"r" ' + Style.RESET_ALL + 'or ' + Fore.YELLOW + Style.BRIGHT + '"R" ' + Style.RESET_ALL + 'key -> Change pencil color to Red!\n')                                                 # Preliminary notes.

    # ------------------------------------------------------------
    # EXECUTION (Drawing with mouse)
    # ------------------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if self.mouseON:                                                                                                # Check if user can draw with mouse or not.
            if event == cv2.EVENT_LBUTTONDOWN:
                self.last_coordinates = (x, y)                                                                          # Actualize "last_coordinates" variable.
                cv2.line(self.aux_image, (x, y), (x, y), self.pencil_color, self.thickness)                             # Draw a line in image "aux_image".
                cv2.line(self.aux_image2, (x, y), (x, y), self.pencil_color, self.thickness)                            # Draw a line in image "aux_image2".
                cv2.imshow(self.window_name, self.aux_image2)                                                           # Display the image

            if event == cv2.EVENT_LBUTTONUP:
                self.last_coordinates = ''                                                                              # Actualize "last_coordinates" variable.
                cv2.line(self.aux_image, (x, y), (x, y), self.pencil_color, self.thickness)                             # Draw a line in image "aux_image".
                cv2.line(self.aux_image2, (x, y), (x, y), self.pencil_color, self.thickness)                            # Draw a line in image "aux_image2".
                cv2.imshow(self.window_name, self.aux_image2)                                                           # Display the image

            if event == cv2.EVENT_MOUSEMOVE:
                if self.last_coordinates != '':                                                                         # Check if there are any last coordinate.
                    cv2.line(self.aux_image, (self.last_coordinates[0], self.last_coordinates[1]), (x, y), self.pencil_color, self.thickness)       # Draw a line in image "aux_image".
                    cv2.line(self.aux_image2, (self.last_coordinates[0], self.last_coordinates[1]), (x, y), self.pencil_color, self.thickness)      # Draw a line in image "aux_image2".
                    cv2.imshow(self.window_name, self.aux_image2)                                                       # Display the image
                    self.last_coordinates = (x, y)                                                                      # Actualize "last_coordinates" variable.

    # ------------------------------------------------------------
    # EXECUTION (Drawing with camera)
    # ------------------------------------------------------------
    def draw_with_camera(self, drawable_image):
        capture = cv2.VideoCapture(0)                                                                                   # Setup video capture for camera.
        all_coordinates = []                                                                                            # Variable initialization.

        changes_detected = False                                                                                        # Initialize variable that indicates if user change something or not.

        while True:
            _, image = capture.read()                                                                                   # Read camera captured video.
            image = cv2.flip(image, 1)                                                                                  # Get an image from the camera and store them at "image" variable.
            image = cv2.resize(image, (self.af4_image_width, self.af4_image_height), interpolation=cv2.INTER_AREA)      # Resize image.
            if image is None:                                                                                           # Check if there are no camera image.
                print(Fore.YELLOW + Style.BRIGHT + 'Video is over, terminating.' + Style.RESET_ALL)                     # Test finished message.
                break                                                                                                   # Break/Stops the loop.

            data = json.load(open(self.jsonFile))                                                                       # Get .json file data and store them at 'data' variable.

            RGB_min = (0, 0, 0)                                                                                         # Variable initialization. Variable with RGB colors minimum value defined in "color_segmenter.py".
            RGB_max = (0, 0, 0)                                                                                         # Variable initialization. Variable with RGB colors maximum value defined in "color_segmenter.py".
            mins = []                                                                                                   # Variable initialization.
            maxs = []                                                                                                   # Variable initialization.

            if list(data['limits'].keys())[0] == 'H':                                                                   # Checks if color segmenter it was defined based in HSV.
                mins = np.array([data['limits']['H']['min'], data['limits']['S']['min'], data['limits']['V']['min']])   # Gets minimum HSV color values from data variable.
                maxs = np.array([data['limits']['H']['max'], data['limits']['S']['max'], data['limits']['V']['max']])   # Gets maximum HSV color values from data variable.

                RGB_min = colorsys.hsv_to_rgb(data['limits']['H']['min'], data['limits']['S']['min'], data['limits']['V']['min'])       # HSV to RGB conversion.
                RGB_max = colorsys.hsv_to_rgb(data['limits']['H']['max'], data['limits']['S']['max'], data['limits']['V']['max'])       # HSV to RGB conversion.

            elif list(data['limits'].keys())[0] == 'B':                                                                 # Checks if color segmenter it was defined based in RGB.
                mins = np.array([data['limits']['B']['min'], data['limits']['G']['min'], data['limits']['R']['min']])   # Gets minimum RGB color values from data variable.
                maxs = np.array([data['limits']['B']['max'], data['limits']['G']['max'], data['limits']['R']['max']])   # Gets maximum RGB color values from data variable.

                RGB_min = mins                                                                                          # Set RGB colors minimum values to "RGB_min" variable.
                RGB_max = maxs                                                                                          # Set RGB colors maxinum values to "RGB_min" variable.

            if RGB_min[0] >= 112:
                self.pencil_color = (255, 0, 0)                                                                         # Set pencil color (Blue).
            elif (RGB_max[0] <= 170) and (RGB_min[1] >= 100) and (RGB_max[2] <= 173):
                self.pencil_color = (0, 255, 0)                                                                         # Set pencil color (Green).
            elif (RGB_min[2] >= 128) and (RGB_max[1] <= 160) and (RGB_max[0] <= 147):
                self.pencil_color = (0, 0, 255)                                                                         # Set pencil color (Red).

            image_processed = cv2.inRange(image, mins, maxs)                                                            # Process original image/video according to RGB/HSV color values range.

            mask = np.ndarray((self.af4_image_height, self.af4_image_width), dtype=np.uint8)                            # Create a mask with the same size as image.
            mask.fill(0)                                                                                                # Fill the mask with white.

            if np.mean(image_processed) > 0:
                contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)       # Get "image_processed" external contour.
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)                                   # Get contour area.
                largest_item = sorted_contours[0]                                                                       # Get largest item/contour.
                cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))                                           # Fill contour with white color.

                cX, cY = moments_calc(mask)                                                                             # Call moments_calc() function to get centroid.

                last_coordinates = (cX, cY)                                                                             # Save last centroid coordinates.
                all_coordinates.append(last_coordinates)                                                                # create a list with all points

            for i in range(2, len(all_coordinates)):
                cv2.line(self.aux_image2, all_coordinates[i], all_coordinates[i - 1], self.pencil_color, self.thickness)    # Draw in white board.
                cv2.line(self.aux_image, all_coordinates[i], all_coordinates[i - 1], self.pencil_color, self.thickness)     # Draw in white board.

            # cv2.imshow('Original Video Image', image)                                                                   # Show original image.
            cv2.imshow(self.window_name, self.aux_image2)                                                               # Display the white window.

            key = cv2.waitKey(20)                                                                                       # Wait for a key press before proceeding.

            if (key == ord('q')) or (key == ord('Q')) or (cv2.getWindowProperty(self.window_name, 1) == -1):            # Check if user pressed the 'q' key or closed the window.
                print(Fore.YELLOW + Style.BRIGHT + '\nPaint aborted!' + Style.RESET_ALL)                                # Test aborted message.
                exit()                                                                                                  # Stops the program.
            elif (key == ord('r')) or (key == ord('R')) or (key == ord('g')) or (key == ord('G')) or (key == ord('b')) or (key == ord('B')):  # Check if user pressed the 'r' key.
                print('Only available when you paint with mouse!. You must ' + Fore.YELLOW + Style.BRIGHT + 'run "color_segmenter.py" ' + Style.RESET_ALL + 'script to change pencil color in camera detection mode!')  # Pencil color changed message.
            elif (key == ord('m')) or (key == ord('M')):                                                                # Check if user pressed the 'm' key or closed the window.
                self.mouseON = True                                                                                     # Set "mouseON" variable as True.
                break                                                                                                   # Break while cycle.
            elif key == ord('+'):                                                                                       # Check if user pressed the '+' key.
                print('Pencil size change (' + Fore.YELLOW + Style.BRIGHT + 'bigger' + Style.RESET_ALL + '). Thickness = ' + str(self.thickness))  # Pencil size changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.thickness = self.thickness + 1                                                                     # Increase pencil dimension.
            elif (key == ord('-')) and (self.thickness >= 2):                                                           # Check if user pressed the '-' key.
                print('Pencil size change (' + Fore.YELLOW + Style.BRIGHT + 'smaller' + Style.RESET_ALL + '). Thickness = ' + str(self.thickness))  # Pencil size changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.thickness = self.thickness - 1                                                                     # Decrease pencil dimension.
            elif (key == ord('f')) or (key == ord('F')):                                                                # Check if user pressed the 'f' key.
                if changes_detected is False:                                                                           # Check if user changed something during painting process.
                    print('No changes.')                                                                                # Paint changes message.
                self.paint_evaluation(drawable_image)                                                                   # Call "paint_evaluation" function to evaluate paint accuracy.
                exit()                                                                                                  # Stops the program.
            elif (key == ord('c')) or (key == ord('C')):                                                                # Check if user pressed the 'c' key.
                print(Fore.YELLOW + Style.BRIGHT + 'Warning!' + Style.RESET_ALL + ' Image clearing its just available when your drawing with mouse.')  # Image cleared message.

    # ------------------------------------------------------------
    # TERMINATION (Paint evaluation)
    # ------------------------------------------------------------
    def paint_evaluation(self, image):
        i = 0                                                                                                           # Initialization of auxiliar variable.

        blue_accuracy = []                                                                                              # Set default value to blue accuracy.
        green_accuracy = []                                                                                             # Set default value to green accuracy.
        red_accuracy = []                                                                                               # Set default value to red accuracy.

        for contour in self.contour_content:
            black_image = np.zeros(image.shape).astype(image.dtype)                                                     # Create a black image with the same size as original image.
            cv2.fillPoly(black_image, [np.array(contour)], [255, 255, 255])                                             # Fill contour with white color.
            fill_contour_aux_image = cv2.bitwise_and(self.aux_image, black_image)                                       # Get an auxiliary image just with filled contour.

            num_blue_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (255, 0, 0), (255, 0, 0)))              # Get the number of blue pixels of the filled contour image.
            if self.num_contour[i] == 1:                                                                                # Check if contour number it's equal to 1 (1 - blue).
                blue_accuracy.append(float((num_blue_pix * 100) / self.num_white_pix_no_num[i]))                        # Calculate blue accuracy.

            num_green_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (0, 255, 0), (0, 255, 0)))             # Get the number of green pixels of the filled contour image.
            if self.num_contour[i] == 2:                                                                                # Check if contour number it's equal to 2 (2 - green).
                green_accuracy.append(float((num_green_pix * 100) / self.num_white_pix_no_num[i]))                      # Calculate green accuracy.

            num_red_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (0, 0, 255), (0, 0, 255)))               # Get the number of red pixels of the filled contour image.
            if self.num_contour[i] == 3:                                                                                # Check if contour number it's equal to 3 (3 - red).
                red_accuracy.append(float((num_red_pix * 100) / self.num_white_pix_no_num[i]))                          # Calculate red accuracy.

            i += 1                                                                                                      # Increment auxiliary variable.

        blue_global_accuracy = 0                                                                                        # Variable initialization.
        green_global_accuracy = 0                                                                                       # Variable initialization.
        red_global_accuracy = 0                                                                                         # Variable initialization.
        for index in range(0, len(blue_accuracy)):                                                                      # "For cycle" to cycle through all accuracy values for blue regions.
            blue_global_accuracy = blue_global_accuracy + blue_accuracy[index]                                          # Calculate global blue accuracy (sum to get mean).

        for index in range(0, len(green_accuracy)):                                                                     # "For cycle" to cycle through all accuracy values for green regions.
            green_global_accuracy = green_global_accuracy + green_accuracy[index]                                       # Calculate global green accuracy (sum to get mean).

        for index in range(0, len(red_accuracy)):                                                                       # "For cycle" to cycle through all accuracy values for red regions.
            red_global_accuracy = red_global_accuracy + red_accuracy[index]                                             # Calculate global red accuracy (sum to get mean).

        if len(blue_accuracy) != 0:
            blue_global_accuracy = format(blue_global_accuracy / len(blue_accuracy), ".2f")                             # Calculate and format global blue accuracy (division to get mean).

        if len(green_accuracy) != 0:
            green_global_accuracy = format(green_global_accuracy / len(green_accuracy), ".2f")                          # Calculate and format global green accuracy (division to get mean).

        if len(red_accuracy) != 0:
            red_global_accuracy = format(red_global_accuracy / len(red_accuracy), ".2f")                                # Calculate and format global red accuracy (division to get mean).

        print(Fore.YELLOW + Style.BRIGHT + '\nPaint finished!' + Style.RESET_ALL)                                       # Test finished message.
        print('\r\n============================= Results =============================\n')                              # Results message.

        print('Blue Color Accuracy: ' + str(blue_global_accuracy) + '%')                                                # Display blue global accuracy value.
        print('Green Color Accuracy: ' + str(green_global_accuracy) + '%')                                              # Display green global accuracy value.
        print('Red Color Accuracy: ' + str(red_global_accuracy) + '%')                                                  # Display red global accuracy value.

        global_accuracy = (float(blue_global_accuracy) + float(green_global_accuracy) + float(red_global_accuracy)) / 3 # Calculate global accuracy.
        global_accuracy = format(global_accuracy, ".2f")                                                                # Set global accuracy value format (2 decimal places).
        print('Global Accuracy: ' + str(global_accuracy) + '%')                                                         # Display global accuracy value.

        self.mouseON = False                                                                                            # Set "mouseON" variable as False.
        print('\r\n============================= Save =============================\n')                                 # Save message.
        print('Do you wanna save the image? [Y,n]')                                                                     # Ask user if he wants to save the painted image.
        while True:
            key = cv2.waitKey(20)                                                                                       # Wait for a key press before proceeding.
            if (key == ord('n')) or (key == ord('N')) or (cv2.getWindowProperty(self.window_name, 1) == -1):            # Check if user pressed the 'n' key or closed the window.
                print(Fore.YELLOW + Style.BRIGHT + '\nPainted image not saved!\n' + Style.RESET_ALL)                    # Test finished message.
                exit()                                                                                                  # Stops the program.
            elif (key == ord('y')) or (key == ord('Y')):                                                                # Check if user pressed the 'y' key.
                current_day_of_week = datetime.datetime.now().strftime("%a")                                            # Get current week day.
                month_number = str(datetime.datetime.today().month)                                                     # Get current month number.
                current_month = datetime.datetime.strptime(month_number, "%m").strftime("%b")                           # Tranform current month number to name/text.
                current_time = datetime.datetime.now().strftime("%d_%H:%M:%S_%Y")                                       # Get current day of month, time and year.
                date_format = str(current_day_of_week) + '_' + str(current_month) + '_' + str(current_time)             # Concatenate all date parameters.
                image_name = 'drawing_AF4_' + date_format + '.png'
                cv2.imwrite(image_name, self.aux_image2)                                                                # Save image as png.
                print(Fore.YELLOW + Style.BRIGHT + '\nPainted image saved as ' + image_name + '!\n' + Style.RESET_ALL)  # Image saved message.
                exit()                                                                                                  # Stops the program.

    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------
    def run(self):
        image = cv2.imread(self.original_image, cv2.IMREAD_COLOR)                                                       # Load an image.

        self.aux_image = image.copy()                                                                                   # Copy original image to auxiliary "variables"/"images".
        self.aux_image2 = image.copy()                                                                                  # Copy original image to auxiliary "variables"/"images".

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)                                                            # Convert image to gray scale.

        _, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)                      # Convert image to binary.

        contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)                      # Find image contours/regions.

        smallest = 1                                                                                                    # Smallest random number allowed.
        largest = 3                                                                                                     # Biggest random number allowed.

        self.af4_image_height, self.af4_image_width, channels = image.shape                                             # Get image size.

        for contour in contours:
            (X, Y, W, H) = cv2.boundingRect(contour)                                                                    # Get coordinates and dimensions of contour bounding rectangle.

            black_image = np.zeros(image.shape).astype(image.dtype)                                                     # create a black image with the same size as original image.
            contour_image = cv2.drawContours(black_image, contour, -1, (255, 255, 255), -1)                             # Draw contour.
            number_of_white_pix_contour = np.sum(contour_image == 255)                                                  # Get the number of white pixels of contour.

            cv2.fillPoly(black_image, [np.array(contour)], [255, 255, 255])                                             # Fill contour with white color.
            contour_image_filled = cv2.bitwise_and(image, black_image)
            num_white_pix_contour_filled = np.sum(contour_image_filled == 255)                                          # Get the number of white pixels of filled contour.

            fill_contour_image_white_pix = cv2.countNonZero(cv2.inRange(contour_image_filled, (255, 255, 255), (255, 255, 255)))    # Get the number of white pixels of the filled contour image.
            self.num_white_pix_no_num.append(fill_contour_image_white_pix)                                                          # Get the number of white pixels of filled contour.

            if (cv2.contourArea(contour) > 100) and ((self.af4_image_width > (W + 50)) and (self.af4_image_height > (H + 50))) and (num_white_pix_contour_filled > (number_of_white_pix_contour * 2)):
                self.contour_content.append(contour)                                                                    # Save each contour an array.

                number = random.randint(smallest, largest)                                                              # Generate an random number between 1 and 4.
                cX, cY = moments_calc(contour)                                                                          # Call moments_calc() function to get centroid.
                self.num_contour.append(number)                                                                         # Save each contour text number.

                dist = cv2.pointPolygonTest(contour, (cX, cY), True)                                                    # Check if calculated centroid coordinates are inside contour.
                if dist < 0:                                                                                            # Check if "dist" value it's smaller than 0. It means that centroid it's outside contour.
                    while dist < 6:                                                                                     # Cycle that avoid number text to be over the contour line.
                        white_pix_coord = np.nonzero(contour_image_filled)                                              # Gets fill contour image white pixels location.
                        cX = random.choice(white_pix_coord[1])                                                          # Gets fill contour image white pixels location (X coordinate).
                        cY = random.choice(white_pix_coord[0])                                                          # Gets fill contour image white pixels location (Y coordinate).
                        dist = cv2.pointPolygonTest(contour, (cX, cY), True)                                            # Check if new "centroid" coordinates are inside contour.

                cv2.putText(self.aux_image2, str(number), (cX - 5, cY + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)     # Writes image region enumeration text and him characteristics.

        cv2.imshow(self.window_name, self.aux_image2)                                                                   # Display image enumeration.

        self.image_clear = copy.copy(self.aux_image2)

        self.last_coordinates = ''
        self.pencil_color = (255, 0, 0)                                                                                 # Set default pencil color.

        cv2.setMouseCallback(self.window_name, self.mouse_callback)                                                     # mouse_callback method.
        print('\r\n============================= Changes during paint =============================\n')                 # Changes during paint message.

        changes_detected = False                                                                                        # Initialize variable that indicates if user change something or not.

        while True:
            key = cv2.waitKey(20)                                                                                       # Wait for a key press before proceeding.
            if (key == ord('q')) or (key == ord('Q')) or (cv2.getWindowProperty(self.window_name, 1) == -1):            # Check if user pressed the 'q' key or closed the window.
                print(Fore.YELLOW + Style.BRIGHT + '\nPaint aborted!' + Style.RESET_ALL)                                # Test aborted message.
                exit()                                                                                                  # Stops the program.
            elif (key == ord('r')) or (key == ord('R')):                                                                # Check if user pressed the 'r' key.
                print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.RED + Style.BRIGHT + 'Red.' + Style.RESET_ALL)    # Pencil color changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.pencil_color = (0, 0, 255)                                                                         # Change pencil color to red.
            elif (key == ord('g')) or (key == ord('G')):                                                                # Check if user pressed the 'g' key.
                print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.GREEN + Style.BRIGHT + 'Green.' + Style.RESET_ALL)      # Pencil color changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.pencil_color = (0, 255, 0)                                                                         # Change pencil color to green.
            elif (key == ord('b')) or (key == ord('B')):                                                                # Check if user pressed the 'b' key.
                print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.BLUE + Style.BRIGHT + 'Blue.' + Style.RESET_ALL)        # Pencil color changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.pencil_color = (255, 0, 0)                                                                         # Change pencil color to blue.
            elif key == ord('+'):                                                                                       # Check if user pressed the '+' key.
                print('Pencil size change (' + Fore.YELLOW + Style.BRIGHT + 'bigger' + Style.RESET_ALL + '). Thickness = ' + str(self.thickness))  # Pencil size changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.thickness = self.thickness + 1                                                                     # Increase pencil dimension.
            elif (key == ord('-')) and (self.thickness >= 2):                                                           # Check if user pressed the '-' key.
                print('Pencil size change (' + Fore.YELLOW + Style.BRIGHT + 'smaller' + Style.RESET_ALL + '). Thickness = ' + str(self.thickness))  # Pencil size changed message.
                changes_detected = True                                                                                 # Set variable that indicates if user change something or not to True.
                self.thickness = self.thickness - 1                                                                     # Decrease pencil dimension.
            elif (key == ord('f')) or (key == ord('F')):                                                                # Check if user pressed the 'f' key.
                if changes_detected is False:                                                                           # Check if user chaged something during painting process.
                    print('No changes.')                                                                                # Paint changes message.
                self.paint_evaluation(image)                                                                            # Call "paint_evaluation" function to evaluate paint accuracy.
                exit()                                                                                                  # Stops the program.
            elif (key == ord('a')) or (key == ord('A')):                                                                # Check if user pressed the 'c' key.
                self.mouseON = False                                                                                    # Set "mouseON" variable as False.
                self.draw_with_camera(image)
            elif (key == ord('m')) or (key == ord('M')):                                                                # Check if user pressed the 'm' key.
                self.mouseON = True                                                                                     # Set "mouseON" variable as True.
                cv2.setMouseCallback(self.window_name, self.mouse_callback)                                             # mouse_callback method.
            elif (key == ord('c')) or (key == ord('C')):  # Check if user pressed the 'c' key.
                print('Do you really wanna clear the image? [Y,n]')                                                     # Ask user if he really wants to clear the painted image.
                while True:
                    key = cv2.waitKey(20)                                                                               # Wait for a key press before proceeding.
                    if (key == ord('n')) or (key == ord('N')):                                                          # Check if user pressed the 'n' key or closed the window.
                        pass
                    elif (key == ord('y')) or (key == ord('Y')):                                                        # Check if user pressed the 'y' key.
                        print(Fore.YELLOW + Style.BRIGHT + 'Image cleared.' + Style.RESET_ALL)                          # Image cleared message.
                        self.aux_image2 = copy.copy(self.image_clear)                                                   # Copy clear image to variable "aux_image2".
                        cv2.imshow(self.window_name, self.image_clear)                                                  # Display image enumeration.


def main():
    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('-j', '--json', help='Full path to json file.', default='limits.json')
    parser.add_argument('-af4', '--advancedFunctionality4', help='Run Advanced Functionality 4 and 5.', action='store_true', default=False)
    parser.add_argument('-if', '--imageFile', help='Full path to image file.', default='./pinguim2.png')                                        # Used just for advanced functionality 4.
    parser.add_argument('-pt', '--pencilThickness', help='Set pencil thickness.', type=int, default=10)                                         # Used just for advanced functionality 4.
    parser.add_argument('-usp', '--use_shake_protection',action='store_true', help='It doesnt draw when you dont want to.',default= False)      # Shake protection mode.
    args = vars(parser.parse_args())

    if args.get('advancedFunctionality4') is True:                                                                      # Check if user wants to run advanced functionality 4 and 5.
        class_report = adc_func_4_5(args.get('json'), args.get('imageFile'), args.get('pencilThickness'))               # Call class "adc_func_4_5()".
        class_report.run()                                                                                              # Call class "run()".
        exit()                                                                                                          # Stops the program.

    capture = cv2.VideoCapture(0)                                                                                       # Setup video capture for camera.

    window_name_segmentation = 'Segmentation'                                                                           # Set window name.

    cv2.namedWindow(window_name_segmentation, cv2.WINDOW_AUTOSIZE)                                                      # Window Setup.
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))                                                                  # Get image dimensions (width).
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))                                                                # Get image dimensions (height).

    white_window = np.full((height, width, 3), 255, dtype=np.uint8)                                                     # Create a white image with the same size as the original image/video.

    pencil_color = (0, 0, 0)                                                                                            # Set pencil default color.
    pencil_dimension = 10                                                                                               # Set pencil default dimension.

    last_coordinates = ''                                                                                               # Initialization of an auxiliary variable.
    all_coordinates = []
    data = json.load(open(args.get('json')))

    if (data['limits']['B']['min'] > data['limits']['G']['min']) and (data['limits']['B']['min'] > data['limits']['R']['min']):
        pencil_color = (255, 0, 0)                                                                                                      # Set pencil color (Blue).
    elif (data['limits']['G']['min'] > data['limits']['B']['min']) and (data['limits']['G']['min'] > data['limits']['R']['min']):
        pencil_color = (0, 255, 0)                                                                                                      # Set pencil color (Green).
    elif (data['limits']['R']['min'] > data['limits']['B']['min']) and (data['limits']['R']['min'] > data['limits']['G']['min']):
        pencil_color = (0, 0, 255)                                                                                                      # Set pencil color (Red).

    # ------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------
    while True:
        _, image = capture.read()
        image = cv2.flip(image, 1)                                                                                      # Get an image from the camera and store them at "image" variable.
        image_raw = copy.copy(image)                                                                                    # Do a copy of image for show the original
        if image is None:                                                                                               # Check if there are no camera image.
            print(Fore.YELLOW + Style.BRIGHT + 'Video is over, terminating.' + Style.RESET_ALL)                         # Test finished message.
            break                                                                                                       # Break/Stops the loop.

                                                                             # Get .json file data and store them at 'data' variable.

        if list(data['limits'].keys())[0] == 'H':                                                                       # Checks if color segmenter it was defined based in HSV.
            mins = np.array([data['limits']['H']['min'], data['limits']['S']['min'], data['limits']['V']['min']])       # Gets minimum HSV color values from data variable.
            maxs = np.array([data['limits']['H']['max'], data['limits']['S']['max'], data['limits']['V']['max']])       # Gets maximum HSV color values from data variable.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        elif list(data['limits'].keys())[0] == 'B':                                                                     # Checks if color segmenter it was defined based in RGB.
            mins = np.array([data['limits']['B']['min'], data['limits']['G']['min'], data['limits']['R']['min']])       # Gets minimum RGB color values from data variable.
            maxs = np.array([data['limits']['B']['max'], data['limits']['G']['max'], data['limits']['R']['max']])       # Gets maximum RGB color values from data variable.



        image_processed = cv2.inRange(image, mins, maxs)                                                                # Process original image/video according to RGB/HSV color values range.

        mask = np.ndarray((height, width), dtype=np.uint8)                                                              # Create a mask with the same size as image.
        mask.fill(0)                                                                                                    # Fill the mask with white.

        # Contour detection and isolation of biggest contour + fill
        if np.mean(image_processed) > 0:
            contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)           # Get "image_processed" external contour.
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)                                       # Get contour area.
            largest_item = sorted_contours[0]                                                                           # Get largest item/contour.
            cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))                                               # Fill contour with white color.
            cv2.fillPoly(image_raw, pts=[largest_item], color=(0, 255, 0))
            cv2.polylines(image_raw, pts=[largest_item], isClosed=True, color=(0, 255, 255), thickness=5)

            cX, cY = moments_calc(mask)                                                                                 # Call moments_calc() function to get centroid.
            cv2.line(image_raw, (cX ,cY), (cX + 4, cY), (0, 0, 0), 2)                                                   #Draw part of the cross on the centroid
            cv2.line(image_raw, (cX, cY), (cX - 4, cY), (0, 0, 0), 2)                                                   #Draw part of the cross on the centroid
            cv2.line(image_raw, (cX, cY), (cX, cY - 4), (0, 0, 0), 2)                                                   #Draw part of the cross on the centroid
            cv2.line(image_raw, (cX, cY), (cX, cY + 4), (0, 0, 0), 2)                                                   #Draw part of the cross on the centroid
            last_coordinates = ( (cX, cY), pencil_color,pencil_dimension)                                                                                 # Save last centroid coordinates.
            all_coordinates.append(last_coordinates)                                                                    # create a list with all points.

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
            #all_coordinates.clear()
        elif (key == ord('g')) or (key == ord('G')):                                                                    # Check if user pressed the 'g' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil color change to Green.' + Style.RESET_ALL)                       # Pencil color changed message.
            pencil_color = (0, 255, 0)                                                                                  # Change pencil color to green.
            #all_coordinates.clear()
        elif (key == ord('b')) or (key == ord('B')):                                                                    # Check if user pressed the 'b' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil color change to Blue.' + Style.RESET_ALL)                        # Pencil color changed message.
            pencil_color = (255, 0, 0)                                                                                  # Change pencil color to blue.
            #all_coordinates.clear()
        elif key == ord('+'):                                                                                           # Check if user pressed the '+' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil size: ' + str(pencil_dimension) + ' (bigger).'  + Style.RESET_ALL)                        # Pencil size changed message.
            pencil_dimension = pencil_dimension + 1                                                                     # Increase pencil dimension.
            #all_coordinates.clear()
        elif (key == ord('-')):                                                                                         # Check if user pressed the '-' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Pencil size: ' + str(pencil_dimension) + ' (smaller).' + Style.RESET_ALL)                       # Pencil size changed message.
            pencil_dimension = pencil_dimension - 1                                                                     # Decrease pencil dimension.
            if (pencil_dimension <=1):
                pencil_dimension = 1
            #all_coordinates.clear()
        elif (key == ord('c')) or (key == ord('C')):                                                                    # Check if user pressed the 'c' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Image cleared.' + Style.RESET_ALL)                                      # Image cleared message.
            all_coordinates.clear()
            white_window.fill(255)
        elif (key == ord('w')) or (key == ord('W')):                                                                    # Check if user pressed the 'w' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Image saved as PNG.' + Style.RESET_ALL)                                 # Image saved message.
            current_day_of_week = datetime.datetime.now().strftime("%a")                                                # Get current week day.
            month_number = str(datetime.datetime.today().month)                                                         # Get current month number.
            current_month = datetime.datetime.strptime(month_number, "%m").strftime("%b")                               # Tranform current month number to name/text.
            current_time = datetime.datetime.now().strftime("%d_%H:%M:%S_%Y")                                           # Get current day of month, time and year.
            date_format = str(current_day_of_week) + '_' + str(current_month) + '_' + str(current_time)                 # Concatenate all date parameters.
            cv2.imwrite('drawing_' + date_format + '.png', image)                                                       # Save image as png.
        elif (key == ord('o')) or (key == ord('O')):                                                                    # Check if user pressed the 'o' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Drawing circle.' + Style.RESET_ALL)
            circleX = last_coordinates[0]                                                                               # Saves centroid x position
            circleY = last_coordinates[1]                                                                               # Saves centroid y position
            while True:
                key1 = cv2.waitKey(10)                                                                                  # Waits for another key press
                _, image = capture.read()
                image = cv2.flip(image, 1)  # Get an image from the camera and store them at "image" variable.
                image_raw = copy.copy(image)  # Do a copy of image for show the original
                mask = np.ndarray((height, width), dtype=np.uint8)  # Create a mask with the same size as image.
                mask.fill(0)
                image_processed = cv2.inRange(image, mins, maxs)  # Process original image/video according to RGB/HSV color values range.
                # Contour detection and isolation of biggest contour + fill
                if np.mean(image_processed) > 0:
                    contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # Get "image_processed" external contour.
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Get contour area.
                    largest_item = sorted_contours[0]  # Get largest item/contour.
                    cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))  # Fill contour with white color.
                    cv2.fillPoly(image_raw, pts=[largest_item], color=(0, 255, 0))
                    cv2.polylines(image_raw, pts=[largest_item], isClosed=True, color=(0, 255, 255), thickness=5)
                    cX1, cY1 = moments_calc(mask)                                                                       # Call moments_calc() function to get centroid.
                    cv2.line(image_raw, (cX1, cY1), (cX1 + 4, cY1), (0, 0, 0), 2)                                       # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1 - 4, cY1), (0, 0, 0), 2)                                       # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 - 4), (0, 0, 0), 2)                                       # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 + 4), (0, 0, 0), 2)                                       # Draw part of the cross on the centroid
                    last_coordinates = (cX1, cY1)  # Save last centroid coordinates.
                    r = int(math.sqrt((circleX - last_coordinates[0]) ** 2 + (circleY - last_coordinates[1]) ** 2))     # Calculates the distance between the initial centroid position and the current centroid position
                    #cv2.circle(white_window, (circleX, circleY), r, pencil_color, -1)
                    cv2.circle(image, (circleX, circleY), r, pencil_color, -1)                                          # Draws circle with variable radius on camera
                    cv2.imshow('Original Video Image', image_raw)  # show original image
                    cv2.imshow('white_window', white_window)  # Display the white window.
                    cv2.imshow(window_name_segmentation, image)  # Display the original image/video.
                if (key1 == ord('o')) or (key1 == ord('O')):
                    cv2.circle(white_window, (circleX, circleY), r, pencil_color, -1)                                   # Draws the circle with the chosen size
                    break
        elif (key == ord('s')) or (key == ord('S')):                                                                    # Check if user pressed the 's' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Drawing rectangle.' + Style.RESET_ALL)
            circleX = last_coordinates[0]
            circleY = last_coordinates[1]
            while True:
                key1 = cv2.waitKey(10)                                                                                  # Waits for another key press
                _, image = capture.read()
                image = cv2.flip(image, 1)  # Get an image from the camera and store them at "image" variable.
                image_raw = copy.copy(image)  # Do a copy of image for show the original
                mask = np.ndarray((height, width), dtype=np.uint8)  # Create a mask with the same size as image.
                mask.fill(0)
                image_processed = cv2.inRange(image, mins,maxs)  # Process original image/video according to RGB/HSV color values range.
                # Contour detection and isolation of biggest contour + fill
                if np.mean(image_processed) > 0:
                    contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # Get "image_processed" external contour.
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Get contour area.
                    largest_item = sorted_contours[0]  # Get largest item/contour.
                    cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))  # Fill contour with white color.
                    cv2.fillPoly(image_raw, pts=[largest_item], color=(0, 255, 0))
                    cv2.polylines(image_raw, pts=[largest_item], isClosed=True, color=(0, 255, 255), thickness=5)
                    cX1, cY1 = moments_calc(mask)                                                                       # Call moments_calc() function to get centroid.
                    cv2.line(image_raw, (cX1, cY1), (cX1 + 4, cY1), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1 - 4, cY1), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 - 4), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 + 4), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    last_coordinates = (cX1, cY1)  # Save last centroid coordinates.
                    #cv2.rectangle(white_window,(circleX,circleY), (cX1,cY1),pencil_color,-1)
                    cv2.rectangle(image, (circleX, circleY), (cX1, cY1), pencil_color, -1)                              # Draws rectangle with variable size on camera
                    cv2.imshow('Original Video Image', image_raw)  # show original image
                    cv2.imshow('white_window', white_window)  # Display the white window.
                    cv2.imshow(window_name_segmentation, image)  # Display the original image/video.
                if (key1 == ord('s') or (key1 == ord('S'))):
                    cv2.rectangle(white_window, (circleX, circleY), (cX1, cY1), pencil_color, -1)                       # Draws the square with the chosen size
                    break
        elif (key == ord('p')) or (key == ord('P')):                                                                    # Check if user pressed the 'p' key.
            print(Fore.YELLOW + Style.BRIGHT + 'Drawing Polygon.' + Style.RESET_ALL)
            circleX = last_coordinates[0]
            circleY = last_coordinates[1]
            points = []
            while True:
                key1 = cv2.waitKey(10)                                                                                  # Waits for another key press
                _, image = capture.read()
                image = cv2.flip(image, 1)  # Get an image from the camera and store them at "image" variable.
                image_raw = copy.copy(image)  # Do a copy of image for show the original
                mask = np.ndarray((height, width), dtype=np.uint8)  # Create a mask with the same size as image.
                mask.fill(0)
                image_processed = cv2.inRange(image, mins,maxs)  # Process original image/video according to RGB/HSV color values range.
                # Contour detection and isolation of biggest contour + fill
                if np.mean(image_processed) > 0:
                    contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # Get "image_processed" external contour.
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Get contour area.
                    largest_item = sorted_contours[0]  # Get largest item/contour.
                    cv2.fillPoly(mask, pts=[largest_item], color=(255, 255, 255))  # Fill contour with white color.
                    cv2.fillPoly(image_raw, pts=[largest_item], color=(0, 255, 0))
                    cv2.polylines(image_raw, pts=[largest_item], isClosed=True, color=(0, 255, 255), thickness=5)
                    M = cv2.moments(mask)  # Centroid coordinates calculation.
                    cX1 = int(M["m10"] / M["m00"])  # Centroid coordinates calculation.
                    cY1 = int(M["m01"] / M["m00"])  # Centroid coordinates calculation.
                    cv2.line(image_raw, (cX1, cY1), (cX1 + 4, cY1), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1 - 4, cY1), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 - 4), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    cv2.line(image_raw, (cX1, cY1), (cX1, cY1 + 4), (0, 0, 0),2)                                        # Draw part of the cross on the centroid
                    last_coordinates = (cX1, cY1)  # Save last centroid coordinates.
                    if (key1 == ord('p')) or (key == ord('P')):
                        points.append((cX1,cY1))                                                                        # Saves a new point to the polygon with the current centroid position
                        print(Fore.YELLOW + Style.BRIGHT + 'Added point do Polygon.' + Style.RESET_ALL)
                    if (len(points) > 0):
                        for i in range(1, len(points)):
                            cv2.line(image, points[i], points[i - 1], pencil_color, 1)                                  # Draws line between the current saved point and the last point on camera
                            cv2.line(white_window, points[i], points[i - 1], pencil_color, 1)                           # Draws line between the current saved point and the last point on whiteboard
                    cv2.imshow('Original Video Image', image_raw)  # show original image
                    cv2.imshow('white_window', white_window)  # Display the white window.
                    cv2.imshow(window_name_segmentation, image)  # Display the original image/video.

                if (key1 == ord('x') or (key1 == ord('X'))):
                    if len(points) == 0:
                        print(Fore.YELLOW + Style.BRIGHT + 'No points were given.' + Style.RESET_ALL)
                        break
                    else:
                        cv2.fillPoly(white_window,np.array([points]),pencil_color,lineType=cv2.LINE_AA)                     # Draws the polygon with all the points and fill
                        break

        for i in range(2, len(all_coordinates)):
            if args.get('use_shake_protection'):
                x = np.array(all_coordinates[i][0])
                z = np.array(all_coordinates[i-1][0])
                w = abs(x - z)

                if str(w[0]) > str(1.5) and str(w[1]) > str(1.5):
                    pass
                else:
                    cv2.line(image, all_coordinates[i][0], all_coordinates[i-1][0], all_coordinates[i][1], all_coordinates[i][2])           # Draw in image.
                    cv2.line(white_window, all_coordinates[i][0], all_coordinates[i-1][0], all_coordinates[i][1], all_coordinates[i][2])
            else:
                cv2.line(image, all_coordinates[i][0], all_coordinates[i-1][0], all_coordinates[i][1], all_coordinates[i][2])               # Draw in image.
                cv2.line(white_window, all_coordinates[i][0], all_coordinates[i-1][0], all_coordinates[i][1], all_coordinates[i][2])      # Draw in white board.




        cv2.imshow(window_name_segmentation, image)
        cv2.imshow('Original Video Image',image_raw)                                                                    # Show original image.
        cv2.imshow('white_window', white_window)                                                                        # Display the white window.

        cv2.moveWindow(window_name_segmentation, 750, 0)  # Move it to (40,30)
        cv2.moveWindow('Original Video Image', 0,0)  # Move it to (40,30)
        cv2.moveWindow('white_window', 0, 730)  # Move it to (40,30)

if __name__ == '__main__':
    main()