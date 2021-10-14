import cv2
# for random color generation in face rectangle
from random import randrange

# Pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default - Copy.xml')

# Chosen img to detect faces in
# img = cv2.imread('Robert_Downey_Jr.jpeg')
# to capture video from webcam (0 = default cam or video file (".mp4"))
webcam = cv2.VideoCapture(0)

# Iterate forever over frames (for video)
while True:
    # Read the current Frame
    successful_frame_read, frame = webcam.read()

# Must convert to grayscale (img or frame)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        #colors can be set with bgr, here I used randrange and specified the upper limit for the colors
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# Face coordinates (just for checking against data)
    #print(face_coordinates)

# Open Window to display the image + Wait till key to close
    cv2.imshow("Clever Programmer Face Detector", frame)
    # When using() wait key will wait untill a key is hit for every frame if using video, 1 indicated that it will wait 1 millisecond before the next frame
    key = cv2.waitKey(1)

    # Stop if Q key is pressed (113 and 81 are ascii numbers for q and Q)
    if key==81 or key==113:
        break
# Release the Video Capture Object
webcam.release()
# To prove code completed with no errors
print("code completed")
