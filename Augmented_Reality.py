#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2 as cv
import os
import numpy as np

print(os.getcwd())

# Load the overlay image
overlay = cv.imread('overlay_image.png')

# Counter Variable
count = 1

# Loop through the pictures with the ARuco markers
for filename in os.listdir("ArUco Markers"):
    # Load the image with the ARuco marker
    image_path = os.path.join('ArUco Markers', filename)
    aruco = cv.imread(image_path)

    # Define the dictionary and parameters for ARuco detection
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the ARuco markers in the image
    corners, ids, rejectedImgPoints = detector.detectMarkers(aruco)
    
    # Draw the detected markers on the image
    aruco_with_corners = cv.aruco.drawDetectedMarkers(aruco, corners, ids)

    # Display the image with the detected corners
    int4_filename = "detected_corners_Image " + str(count) + ".jpg"
    cv.imwrite(int4_filename,aruco_with_corners)

    # If no markers are detected, continue to the next image
    if ids is None:
        print("No markers detected in the image: {}".format(filename))
        continue

    # Dimensions of the overlay image
    #height, width, _ = overlay.shape
    
    #print("shape output is: ",overlay.shape)
    
    top_left = [0,0]
    top_right = [width,0]
    bot_right = [width,height]
    bot_left = [0,height]
    center = [width/2, height/2]
    # Scale factor to adjust the size of the overlay image
    w = width/5
    h = height/5
    # Compute first homography
        # Source points - Scaled corners of the overlay image
    c1 = (center[0] - w/2, center[1] - h/2) 
    c2 = (center[0] + w/2, center[1] - h/2)
    c3 = (center[0] + w/2, center[1] + h/2)
    c4  =(center[0] - w/2, center[1] + h/2)
    src_pts1 = np.float32([c1,c2,c3,c4])
    
    #print("updated corners points are:", c1,c2,c3,c4)

        # Destination points - Corners of the ARuco marker
    dst_pts1 = corners[0][0].astype(np.float32)
    # Compute the homography matrix to obtain the transformation
    M1, _ = cv.findHomography(src_pts1, dst_pts1)
    M1 = M1.astype(np.float32)
    
    #print("Homography matrix is: ", M1)

    # Compute the final destination points
        # Source points - Corners of the overlay image
    src_pts2 = np.float32([top_left, top_right, bot_right, bot_left])
        # Destination points - Modified corners of the overlay image
    dst_pts2 = cv.perspectiveTransform(src_pts2.reshape(-1, 1, 2), M1)

    # Warp the overlay image
    overlay_warped = cv.warpPerspective(overlay, M1, (aruco.shape[1], aruco.shape[0]))
    int1_filename = "overlay_warped_Image " + str(count) + ".jpg"
    cv.imwrite(int1_filename,overlay_warped)

    # Create a mask for the overlay image
    overlay_gray = cv.cvtColor(overlay_warped, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(overlay_gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Apply the mask to the original image
    img_masked = cv.bitwise_and(aruco, aruco, mask=mask_inv)
    mask_filename = "masked_original_Image " + str(count) + ".jpg"
    cv.imwrite(mask_filename,img_masked)

    # Combine the masked original image and the overlay image
    img_final = cv.add(img_masked, overlay_warped)

    # Save the final image in a jpg file
    output_filename = "Image " + str(count) + ".jpg"
    cv.imwrite(output_filename,img_final)
    count += 1





