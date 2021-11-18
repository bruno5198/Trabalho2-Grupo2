#!/usr/bin/python3
import random
import copy
import cv2
import argparse
import numpy as np
from colorama import Fore, Back, Style


def mouse_callback(event, x, y, flags, param):
    global aux_image, aux_image2, last_coordinates, pencil_color
    thickness = 200                                                                                                       # Set thickness value (5 pixels).
    if event == cv2.EVENT_LBUTTONDOWN:
        last_coordinates = (x, y)  # Circle coordinates.
        cv2.line(aux_image, (x, y), (x, y), pencil_color, thickness)
        cv2.line(aux_image2, (x, y), (x, y), pencil_color, thickness)
        cv2.imshow('window', aux_image2)  # Display the image

    if event == cv2.EVENT_LBUTTONUP:
        last_coordinates = ''
        cv2.line(aux_image, (x, y), (x, y), pencil_color, thickness)
        cv2.line(aux_image2, (x, y), (x, y), pencil_color, thickness)
        cv2.imshow('window', aux_image2)  # Display the image

    if event == cv2.EVENT_MOUSEMOVE:
        if last_coordinates != '':
            cv2.line(aux_image, (last_coordinates[0], last_coordinates[1]), (x, y), pencil_color, thickness)
            cv2.line(aux_image2, (last_coordinates[0], last_coordinates[1]), (x, y), pencil_color, thickness)
            cv2.imshow('window', aux_image2)  # Display the image
            last_coordinates = (x, y)


def paint_evaluation(image, contour_content, num_white_pix_no_num, num_contour):
    i = 0                                                                                                               # Initialization of auxiliar variable.

    blue_accuracy = []                                                                                                  # Set default value to blue accuracy.
    green_accuracy = []                                                                                                 # Set default value to green accuracy.
    red_accuracy = []                                                                                                   # Set default value to red accuracy.

    for contour in contour_content:
        black_image = np.zeros(image.shape).astype(image.dtype)                                                         # Create a black image with the same size as original image.
        cv2.fillPoly(black_image, [np.array(contour)], [255, 255, 255])                                                 # Fill contour with white color.
        fill_contour_aux_image = cv2.bitwise_and(aux_image, black_image)                                                # Get an auxiliar image just with filled contour.
        # cv2.imshow('fill_contour_aux_image', fill_contour_aux_image)                                                    # Display the filled contour image.

        # print('white: ' + str(num_white_pix_no_num[i]))

        num_blue_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (255, 0, 0), (255, 0, 0)))                  # Get the number of blue pixels of the filled contour image.
        # print('The number of blue pixels is: ' + str(num_blue_pix))
        if num_contour[i] == 1:                                                                                         # Check if contour number it's equal to 1 (1 - blue).
            blue_accuracy.append(float((num_blue_pix * 100) / num_white_pix_no_num[i]))                                 # Calculate blue accuracy.

        num_green_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (0, 255, 0), (0, 255, 0)))                 # Get the number of green pixels of the filled contour image.
        # print('The number of green pixels is: ' + str(num_green_pix))
        if num_contour[i] == 2:                                                                                         # Check if contour number it's equal to 2 (2 - green).
            green_accuracy.append(float((num_green_pix * 100) / num_white_pix_no_num[i]))                               # Calculate green accuracy.

        num_red_pix = cv2.countNonZero(cv2.inRange(fill_contour_aux_image, (0, 0, 255), (0, 0, 255)))                   # Get the number of red pixels of the filled contour image.
        # print('The number of red pixels is: ' + str(num_red_pix))
        if num_contour[i] == 3:                                                                                         # Check if contour number it's equal to 3 (3 - red).
            red_accuracy.append(float((num_red_pix * 100) / num_white_pix_no_num[i]))                                   # Calculate red accuracy.

        i += 1                                                                                                          # Increment auxiliar variable.
        # cv2.waitKey(0)

    blue_global_accuracy = 0                                                                                            # Variable initialization.
    green_global_accuracy = 0                                                                                           # Variable initialization.
    red_global_accuracy = 0                                                                                             # Variable initialization.
    for index in range(0, len(blue_accuracy)):                                                                          # "For cycle" to cycle through all accuracy values for blue regions.
        blue_global_accuracy = blue_global_accuracy + blue_accuracy[index]                                              # Calculate global blue accuracy (sum to get mean).

    for index in range(0, len(green_accuracy)):                                                                         # "For cycle" to cycle through all accuracy values for green regions.
        green_global_accuracy = green_global_accuracy + green_accuracy[index]                                           # Calculate global green accuracy (sum to get mean).

    for index in range(0, len(red_accuracy)):                                                                           # "For cycle" to cycle through all accuracy values for red regions.
        red_global_accuracy = red_global_accuracy + red_accuracy[index]                                                 # Calculate global red accuracy (sum to get mean).

    if len(blue_accuracy) != 0:
        blue_global_accuracy = format(blue_global_accuracy / len(blue_accuracy), ".2f")                                 # Calculate and format global blue accuracy (division to get mean).

    if len(green_accuracy) != 0:
        green_global_accuracy = format(green_global_accuracy / len(green_accuracy), ".2f")                              # Calculate and format global green accuracy (division to get mean).

    if len(red_accuracy) != 0:
        red_global_accuracy = format(red_global_accuracy / len(red_accuracy), ".2f")                                    # Calculate and format global red accuracy (division to get mean).

    print(Fore.YELLOW + Style.BRIGHT + '\nPaint finished!' + Style.RESET_ALL)                                           # Test finished message.
    print('\r\n============================= Results =============================\n')                                  # Results message.

    print('Blue Color Accuracy: ' + str(blue_global_accuracy) + '%')                                                    # Display blue global accuracy value.
    print('Green Color Accuracy: ' + str(green_global_accuracy) + '%')                                                  # Display green global accuracy value.
    print('Red Color Accuracy: ' + str(red_global_accuracy) + '%')                                                      # Display red global accuracy value.

    global_accuracy = (float(blue_global_accuracy) + float(green_global_accuracy) + float(red_global_accuracy)) / 3     # Calculate global accuracy.
    global_accuracy = format(global_accuracy, ".2f")                                                                    # Set global accuracy value format (2 decimal places).
    print('Global Accuracy: ' + str(global_accuracy) + '%')                                                             # Display global accuracy value.


def main():
    global aux_image, aux_image2, last_coordinates, pencil_color

    image_filename = 'pinguim2.png'
    # image_filename = 'drawing1.png'
    # image_filename = 'drawing3.png'
    # image_filename = 'drawing3_V2.png'
    # image_filename = 'squares.png'

    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)                                                                # Load an image.

    aux_image = image.copy()                                                                                            # Copy original image to auxiliary "variables"/"images".
    aux_image2 = image.copy()                                                                                           # Copy original image to auxiliary "variables"/"images".

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)                                                                # Convert image to gray scale.

    _, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)                          # Convert image to binary.

    contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)                          # Find image contours/regions.

    smallest = 1                                                                                                        # Smallest random number allowed.
    largest = 3                                                                                                         # Biggest random number allowed.

    height, width, channels = image.shape                                                                               # Get image size.

    num_white_pix_no_num = []                                                                                           # Variable initialization.
    num_contour = []                                                                                                    # Variable initialization.
    contour_content = []                                                                                                # Variable initialization.

    for contour in contours:
        (X, Y, W, H) = cv2.boundingRect(contour)                                                                        # Get coordinates and dimensions of contour bounding rectangle.

        # crop_img = image_binary[Y:Y + H, X:X + W]
        # cv2.imshow("cropped", crop_img)

        black_image = np.zeros(image.shape).astype(image.dtype)                                                         # create a black image with the same size as original image.
        contour_image = cv2.drawContours(black_image, contour, -1, (255, 255, 255), -1)                                 # Draw contour.
        number_of_white_pix_contour = np.sum(contour_image == 255)                                                      # Get the number of white pixels of contour.
        # cv2.imshow('contour_image', contour_image)                                                                      # Display the contour image.

        cv2.fillPoly(black_image, [np.array(contour)], [255, 255, 255])                                                 # Fill contour with white color.
        contour_image_filled = cv2.bitwise_and(image, black_image)
        num_white_pix_contour_filled = np.sum(contour_image_filled == 255)                                              # Get the number of white pixels of filled contour.
        # cv2.imshow('contour_image_filled', contour_image_filled)                                                        # Display the fill contour image.

        fill_contour_image_white_pix = cv2.cvtColor(contour_image_filled, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('contour_image_filled_GRAY', contour_image_filled)                                                   # Display the fill contour image.
        num_white_pix_no_num.append(np.sum(fill_contour_image_white_pix == 255))                                        # Get the number of white pixels of filled contour.

        if (cv2.contourArea(contour) > 100) and ((width > (W + 50)) and (height > (H + 50))) and (num_white_pix_contour_filled > (number_of_white_pix_contour * 2)):
            contour_content.append(contour)                                                                             # Save each contour an array.

            number = random.randint(smallest, largest)                                                                  # Generate an random number between 1 and 4.
            M = cv2.moments(contour)                                                                                    # Get contour moments.
            cX = int(M["m10"] / M["m00"])                                                                               # Calculate centroid (X coordinate).
            cY = int(M["m01"] / M["m00"])                                                                               # Calculate centroid (Y coordinate).
            # cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)                                                    # Draw the contour in original image.
            # cv2.circle(image, (cX, cY), 7, (0, 255, 0), -1)

            num_contour.append(number)                                                                                  # Save each contour text number.

            dist = cv2.pointPolygonTest(contour, (cX, cY), True)                                                        # Check if calculated centroid coordinates are inside contour.
            if dist < 0:                                                                                                # Check if "dist" value it's smaller than 0. It means that centroid it's outside contour.
                while dist < 6:                                                                                         # Cycle that avoid number text to be over the contour line.
                    white_pix_coord = np.nonzero(contour_image_filled)                                                  # Gets fill contour image white pixels location.
                    cX = random.choice(white_pix_coord[1])                                                              # Gets fill contour image white pixels location (X coordinate).
                    cY = random.choice(white_pix_coord[0])                                                              # Gets fill contour image white pixels location (Y coordinate).
                    dist = cv2.pointPolygonTest(contour, (cX, cY), True)                                                # Check if new "centroid" coordinates are inside contour.

            # cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(aux_image2, str(number), (cX - 5, cY + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)         # Writes image region enumeration text and him characteristics.
            # cv2.imshow('Enumerated Image', aux_image2)                                                                       # Display image enumeration.
            # cv2.waitKey(0)

    cv2.imshow('window', aux_image2)                                                                                    # Display image enumeration.



    print('\n========== PSR Ar Paint - Advanced Funcionality 4 and 5(Grupo 2) ==========\n')        # Initial message.
    print('=> You must paint image regions with the correct color.')                                                    # Preliminary notes.
    print('    => Region Number 1 -> ' + Fore.BLUE + Style.BRIGHT + 'Blue'  + Style.RESET_ALL + ' Color.')              # Preliminary notes.
    print('    => Region Number 2 -> ' + Fore.GREEN + Style.BRIGHT + 'Green'  + Style.RESET_ALL + ' Color.')            # Preliminary notes.
    print('    => Region Number 3 -> ' + Fore.RED + Style.BRIGHT + 'Red'  + Style.RESET_ALL + ' Color.')                # Preliminary notes.
    print('\n=> Press "f" or "F" key to evaluate your paint accuracy!')                                                 # Preliminary notes
    print('=> Press "q" or "Q" key to exit without evaluate your paint accuracy!\n')                                      # Preliminary notes.



    last_coordinates = ''
    pencil_color = (255, 0, 0)                                                                                          # Set default pencil color.

    cv2.setMouseCallback('window', mouse_callback)                                                                      # mouse_callback method.

    while True:
        key = cv2.waitKey(20)                                                                                           # Wait for a key press before proceeding.

        if (key == ord('q')) or (key == ord('Q')) or (cv2.getWindowProperty('window', 1) == -1):                        # Check if user pressed the 'q' key or closed the window.
            print(Fore.YELLOW + Style.BRIGHT + '\nPaint finished!' + Style.RESET_ALL)                                   # Test finished message.
            exit()                                                                                                      # Stops the program.
        elif (key == ord('r')) or (key == ord('R')):                                                                    # Check if user pressed the 'r' key.
            print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.RED + Style.BRIGHT + 'Red.' + Style.RESET_ALL)                         # Pencil color changed message.
            pencil_color = (0, 0, 255)                                                                                  # Change pencil color to red.
        elif (key == ord('g')) or (key == ord('G')):                                                                    # Check if user pressed the 'g' key.
            print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.GREEN + Style.BRIGHT + 'Green.' + Style.RESET_ALL)                         # Pencil color changed message.
            pencil_color = (0, 255, 0)                                                                                  # Change pencil color to green.
        elif (key == ord('b')) or (key == ord('B')):                                                                    # Check if user pressed the 'b' key.
            print('Pencil ' + Fore.YELLOW + Style.BRIGHT + 'color change' + Style.RESET_ALL + ' to ' + Fore.BLUE + Style.BRIGHT + 'Blue.' + Style.RESET_ALL)                         # Pencil color changed message.
            pencil_color = (255, 0, 0)                                                                                  # Change pencil color to blue.
        elif (key == ord('f')) or (key == ord('F')):                                                                    # Check if user pressed the 'f' key.
            # print(Fore.YELLOW + Style.BRIGHT + 'Paint analysis.' + Style.RESET_ALL)                                     # Paint evaluation message.
            paint_evaluation(image, contour_content, num_white_pix_no_num, num_contour)                                 # Call "paint_evaluation" function to evaluate paint accuracy.
            exit()                                                                                                      # Stops the program.

    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
