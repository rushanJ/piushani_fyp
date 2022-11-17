import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import os
from pynput.keyboard import Key, Controller
keyboard = Controller()

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

padding = 20

def age_gender_detector(frame):
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        return gender

    return "Unknown"

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''

    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Calculate the required angles.
    # ----------------------------------------------------------------------------------------------------------------

    # Get the angle between the left shoulder, elbow and wrist points.
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points.
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # ----------------------------------------------------------------------------------------------------------------

    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # ----------------------------------------------------------------------------------------------------------------

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 30 and left_shoulder_angle < 60 and right_shoulder_angle > 30 and right_shoulder_angle < 60:
            label = 'A Pose'
            cv2.imwrite(label + '.png', output_image)
        if left_shoulder_angle > 300 and left_shoulder_angle < 360 and right_shoulder_angle > 300 and right_shoulder_angle < 360 :
            label = 'L Pose'
            cv2.imwrite(label+'.png', output_image)

   # -----------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

        # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Check if the resultant image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        # Return the output image and the classified label.
        return output_image, label
# Initializing mediapipe pose class.

lPoseDetected=False
aPoseDetected=False
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


# Read an image from the specified path.
sample_img = cv2.imread('media/sample.jpg')

# Specify a size of the figure.
# plt.figure(figsize = [10, 10])

# Display the sample image, also convert BGR to RGB for display.
# plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

# Perform pose detection after converting the image into RGB format.
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

# Check if any landmarks are found.
if results.pose_landmarks:

    # Iterate two times as we only want to display first two landmarks.
    for i in range(2):
        # Display the found normalized landmarks.
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

    # Retrieve the height and width of the sample image.
image_height, image_width, _ = sample_img.shape

# Check if any landmarks are found.
if results.pose_landmarks:

    # Iterate two times as we only want to display first two landmark.
    for i in range(2):
        # Display the found landmarks after converting them into their original scale.
        print(f'{mp_pose.PoseLandmark(i).name}:')
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
        print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

# Create a copy of the sample image to draw landmarks on.
img_copy = sample_img.copy()

# Check if any landmarks are found.
if results.pose_landmarks:
    # Draw Pose landmarks on the sample image.
    mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                              connections=mp_pose.POSE_CONNECTIONS)

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
# cams_test = 500
# for i in range(0, cams_test):
    # if i== 1: continue
    # cap = cv2.VideoCapture(i)
    # test, frame = cap.read()
    # if test:
    #     print("i : "+str(i)+" /// result: "+str(test))
camera_video = cv2.VideoCapture()
camera_video.set(3, 1280)
camera_video.set(4, 960)
# camera_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# camera_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera_video.set(cv2.CAP_PROP_FPS, 30)

height = camera_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = camera_video.get(cv2.CAP_PROP_FRAME_WIDTH)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly.
    if not ok:
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape

    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)

    # Check if the landmarks are detected.
    if landmarks:
        # Perform the Pose Classification.
        frame, lable = classifyPose(landmarks, frame, display=False)

    lPoseDetected = True
    if lable=='L Pose':
        lPoseDetected = True
        print(lable)
    if lable=='A Pose':
        aPoseDetected = True
        print(lable)
    if aPoseDetected==True and lPoseDetected==True:
        f = open("data.txt", "w")
        f.write(str(1))
        f.close()

        lPoseDetected = False
        aPoseDetected = False

        # break
    # Display the frame.
    cv2.imshow('Pose Classification', frame)

    # Wait until a key is pressed.aa
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed.
    if (k == 27):
        # Break the loop.
        break


# Release the VideoCapture object and close the windows.

input = cv2.imread("A Pose.png")
gender = age_gender_detector(input)
print(gender);

def cropImg(img):
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    y0 = 0
    yMax = 150
    x0 = 0
    xMax = 700
    # converting to its binary form
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    found = 0  # To reduce big O notation
    #
    # Get  boudnries
    for index, i in enumerate(bw[1]):
        if found == 1: break
        for j in i:
            if j == 0:
                y0 = index
                found = 1
                break

    found = 0
    for index, i in enumerate(reversed(bw[1])):
        if found == 1: break
        for j in i:
            if j == 0:
                yMax = len(bw[1]) - index
                found = 1
                break

    rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate the image by 90 degrees.
    rotated_bw = cv2.threshold(rotated_image, 127, 255, cv2.THRESH_BINARY)

    found = 0
    for index, i in enumerate(rotated_bw[1]):
        if found == 1: break
        for j in i:
            if j == 0:
                x0 = index
                found = 1
                break

    found = 0
    for index, i in enumerate(reversed(rotated_bw[1])):
        if found == 1: break
        for j in i:
            if j == 0:
                xMax = len(rotated_bw[1]) - index
                found = 1
                break

    return x0,xMax,y0,yMax

def face(img):
    #face
    x0, xMax, y0, yMax = cropImg(img)
    crop_img = img[y0:yMax, x0:xMax]
    start_presentage=0
    end_presentage=16
    xRight=0
    xLeft=xMax
    crop_img_face = img[y0:int(yMax/100*end_presentage), x0:xMax]
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # converting to its binary form
    bw = cv2.threshold(crop_img_face, 127, 255, cv2.THRESH_BINARY)
    #
    # Get  boudnries
    for index1, i in enumerate(bw[1]):
        for index,j in enumerate(i):
            if j == 0:
                # print(index)
                if xRight<index:
                    xRight = index
                # break

    for index1, i in enumerate(bw[1]):
        for index,j in enumerate(i):
            if j == 0:
                # print(index)
                if xLeft>index:
                    xLeft = index
                break

    start_point = (xRight, int(yMax/100*start_presentage))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (xLeft, int(yMax/100*end_presentage))

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(crop_img, start_point, end_point, color, thickness)
    return image ,xRight - xLeft

def chest(img):
    x0, xMax, y0, yMax = cropImg(img)
    crop_img = img[y0:yMax, x0:xMax]
    start_presentage = 20
    end_presentage = 25
    xRight = 0
    xLeft = xMax
    crop_img_chest = img[y0:int(yMax / 100 * end_presentage), x0:xMax]
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # converting to its binary form
    bw = cv2.threshold(crop_img_chest, 127, 255, cv2.THRESH_BINARY)
    #
    # Get  boudnries
    for index1, i in enumerate(bw[1]):
        for index, j in enumerate(i):
            if j == 0:
                # print(index)
                if xRight < index:
                    xRight = index
                # break

    for index1, i in enumerate(bw[1]):
        for index, j in enumerate(i):
            if j == 0:
                # print(index)
                if xLeft > index:
                    xLeft = index
                break

    start_point = (xRight, int(yMax / 100 * start_presentage))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (xLeft, int(yMax / 100 * end_presentage))

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(crop_img, start_point, end_point, color, thickness)
    return image ,xRight - xLeft

def waist(img):
    x0, xMax, y0, yMax = cropImg(img)
    crop_img = img[y0:yMax, x0:xMax]
    start_presentage = 30
    end_presentage = 35
    xRight = 0
    xLeft = xMax
    crop_img_chest = img[y0:int(yMax / 100 * end_presentage), x0:xMax]
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # converting to its binary form
    bw = cv2.threshold(crop_img_chest, 127, 255, cv2.THRESH_BINARY)
    #
    # Get  boudnries
    for index1, i in enumerate(bw[1]):
        for index, j in enumerate(i):
            if j == 0:
                # print(index)
                if xRight < index:
                    xRight = index
                # break

    for index1, i in enumerate(bw[1]):
        for index, j in enumerate(i):
            if j == 0:
                # print(index)
                if xLeft > index:
                    xLeft = index
                break

    start_point = (xRight, int(yMax / 100 * start_presentage))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (xLeft, int(yMax / 100 * end_presentage))

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(crop_img, start_point, end_point, color, thickness)
    return image ,xRight - xLeft


def hip(img):
    rightHandFlag=0
    x0, xMax, y0, yMax = cropImg(img)
    crop_img = img[y0:yMax, x0:xMax]
    start_presentage=40
    end_presentage=50
    xRight=0
    xLeft=xMax
    crop_img_face = img[y0:int(yMax/100*end_presentage), x0:xMax]
    # converting to its binary form
    bw = cv2.threshold(crop_img_face, 127, 255, cv2.THRESH_BINARY)
    #
    # Get  boudnries
    for index1, i in enumerate(bw[1]):
        for index,j in enumerate(i):
            if j == 0:
                if rightHandFlag==1:
                    continue
                elif rightHandFlag==2:
                    if xRight<index:
                        xRight = index
                elif rightHandFlag == 0:
                    rightHandFlag == 1
            else:
                if rightHandFlag == 1:
                    rightHandFlag =2
                # break

    for index1, i in enumerate(bw[1]):
        for index,j in enumerate(i):
            if j == 0:
                # print(index)
                if xLeft>index:
                    xLeft = index
                break

    print (xLeft)
    print(xRight)

    start_point = (xRight, int(yMax/100*start_presentage))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (xLeft, int(yMax/100*end_presentage))

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(crop_img, start_point, end_point, color, thickness)
    return image ,xRight - xLeft

# read the image file
img = cv2.imread('A Pose.png', 2)
img2 = cv2.imread('L Pose.png', 2)



# imgplot = plt.imshow(image)
# plt.show()


# x20,x2Max,y20,y2Max=cropImg(img2)
# crop_img2 = img2[y20:y2Max, x20:x2Max]

# face_image,face_width=face(img)
hip_image,hip_width=hip(img)
chest_image,chest_width=chest(img2)
waist_image,waist_width=waist(img2)

print("waist_width",waist_width)
print("chest_width",chest_width)
# print("face_width",face_width)
cv2.imshow("image", hip_image)
camera_video.release()
cv2.destroyAllWindows()