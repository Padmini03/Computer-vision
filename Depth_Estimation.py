#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np

def focal_length(image_path, chessboard_width, chessboard_height, chessboard_distance):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)
    

    if ret:
        # Calculate the focal length using the perspective projection equation
        focal_length_width = chessboard_distance * (corners[-1, 0, 0] - corners[0, 0, 0]) / chessboard_width
        focal_length_height = chessboard_distance * (corners[-chessboard_width, 0, 1] - corners[0, 0, 1]) / chessboard_height

        return focal_length_width, focal_length_height

    else:
        print("Chessboard corners not found.")
        return None

def calculate_and_display_focal_length(image_path, chessboard_width, chessboard_height, chessboard_distance):
    # Calculate the focal length
    focal_lengths = focal_length(image_path, chessboard_width, chessboard_height, chessboard_distance)

    if focal_lengths:
        focal_length_width, focal_length_height = focal_lengths
        print("Focal length (width):", focal_length_width, "pixels")
        print("Focal length (height):", focal_length_height, "pixels")

# path of the image
image_path = "CV/Task2/task_b/frame_1295.jpg"

# dimensions of the chessboard (in squares)
chessboard_width = 7
chessboard_height = 7

# Known distance from the camera to the chessboard (in meters)
chessboard_distance = 4.7

# Calculate and display the focal length
calculate_and_display_focal_length(image_path, chessboard_width, chessboard_height, chessboard_distance)


def calculate_distance(image_paths, chessboard_width, chessboard_height, focal_length_width, focal_length_height):
    distances = []
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)

        # Find the dimensions of the image
        image_height, image_width = image.shape[:2]

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), None)

        if ret:
            # Calculate the distances using the focal lengths and perspective projection equations
            distance_width = (chessboard_width * focal_length_width) / (corners[-1, 0, 0] - corners[0, 0, 0])
            distance_height = (chessboard_height * focal_length_height) / (corners[-chessboard_width, 0, 1] - corners[0, 0, 1])

            distances.append((distance_width, distance_height))
        else:
            print("Chessboard corners not found for image:", image_path)

    return distances

# Specify the paths to the images
image_paths = ["CV/Task2/Task_b/frame_0890.jpg","CV/Task2/task_b/frame_1295.jpg", "CV/Task2/Task_b/frame_1904.jpg"]

# Specify the dimensions of the chessboard (in squares)
chessboard_width = 7
chessboard_height = 7

# Specify the focal lengths (previously calculated)
focal_length_width = 64.37526332310269
focal_length_height = 67.29055175781251

# Calculate the distances
distances = calculate_distance(image_paths, chessboard_width, chessboard_height, focal_length_width, focal_length_height)

# Print the distances
for i, (distance_width, distance_height) in enumerate(distances):
    print("Distance", i+1, "- Width:", distance_width, "meters, Height:", distance_height, "meters")


# In[ ]:




