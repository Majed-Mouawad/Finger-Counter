import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


def detectHandsLandmarks(image, hands, draw=True, display = True):
    
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:
        
        # Iterate over the found hands.
        for num, hand in enumerate(results.multi_hand_landmarks):
            
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(output_image, hand,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))
            
    
    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and results of hands landmarks detection.
        return output_image, results   

def countFingers(image, results, draw=True, display=True):

    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        
        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
        cv2.rectangle(output_image, (20, 0), (280, 200), (0, 0, 0), cv2.FILLED)
        cv2.putText(output_image, str(sum(count.values())), (45, 140), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 255, 255), 25)
        # Write the total count of the fingers of both hands on the output image.
        #cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_PLAIN,
                    #8.9, (255,0,0), 10, 10)

    # Check if the output image is specified to be displayed.
    if display:
        
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count

def getHandType(image, results, draw=True, display = True):

    # Create a copy of the input image to write hand type label on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the classification info of both hands.
    hands_status = {'Right': False, 'Left': False, 'Right_index' : None, 'Left_index': None}
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        
        # Retrieve the label of the found hand.
        hand_type = hand_info.classification[0].label
        
        # Update the status of the found hand.
        hands_status[hand_type] = True
        
        # Update the index of the found hand.
        hands_status[hand_type + '_index'] = hand_index 
        
        # Check if the hand type label is specified to be written.
        if draw:
        
            # Write the hand type on the output image. 
            cv2.putText(output_image, hand_type + ' Hand Detected', (10, (hand_index+1) * 30),cv2.FONT_HERSHEY_PLAIN,
                        2, (0,255,0), 2)
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
        
        # Return the output image and the hands status dictionary that contains classification info.
        return output_image, hands_status

def drawBoundingBoxes(image, results, hand_status, padd_amount = 10, draw=True, display=True):

    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    output_image = image.copy()
    
    # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
    output_landmarks = {}

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Iterate over the found hands.
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        # Get all the x-coordinate values from the found landmarks of the hand.
        x_coordinates = np.array(landmarks)[:,0]
        
        # Get all the y-coordinate values from the found landmarks of the hand.
        y_coordinates = np.array(landmarks)[:,1]
        
        # Get the bounding box coordinates for the hand with the specified padding.
        x1  = int(np.min(x_coordinates) - padd_amount)
        y1  = int(np.min(y_coordinates) - padd_amount)
        x2  = int(np.max(x_coordinates) + padd_amount)
        y2  = int(np.max(y_coordinates) + padd_amount)

        # Initialize a variable to store the label of the hand.
        label = "Unknown"
        
        # Check if the hand we are iterating upon is the right one.
        if hand_status['Right_index'] == hand_index:
            
            # Update the label and store the landmarks of the hand in the dictionary. 
            label = 'Right'
            output_landmarks['Right'] = landmarks
        
        # Check if the hand we are iterating upon is the left one.
        elif hand_status['Left_index'] == hand_index:
            
            # Update the label and store the landmarks of the hand in the dictionary. 
            label = 'Left'
            output_landmarks['Left'] = landmarks
        
        # Check if the bounding box and the classified label is specified to be written.
        if draw:
            
            # Draw the bounding box around the hand on the output image.
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_8)
            
            # Write the classified label of the hand below the bounding box drawn. 
            cv2.putText(output_image, label, (x1, y2+25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)
    
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:

        # Return the output image and the landmarks dictionary.
        return output_image, output_landmarks

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(1)
camera_video.set(3,1280)
camera_video.set(4,960)

# Create named window for resizing purposes.
cv2.namedWindow('Finger Counter with Hand Classification', cv2.WINDOW_NORMAL)
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                             min_detection_confidence=0.7, min_tracking_confidence=0.4)
# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
 
    
    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_video, display=False)
    
    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
            
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, draw= True, display=False)
        _, hands_status = getHandType(frame.copy(), results, draw=False, display=False)
        
        # Get the landmarks dictionary that stores each hand landmarks as different elements. 
        frame, landmark_dict = drawBoundingBoxes(frame, results, hands_status, display=False)
           
    # Display the frame.
    cv2.imshow('Finger Counter with Hand Categorization', frame)
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()