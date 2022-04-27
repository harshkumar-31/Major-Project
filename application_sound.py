from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
import webbrowser
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("Classifier.hdf5")
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

prev_label = "neutral"
count = 0
# keep looping
while True:
    canvas = np.zeros((500, 500, 3), dtype="uint8")*255

    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # clone the frame so we can draw on it
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:

        # determine the largest face area
        rect = sorted(rects, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        # extract the face ROI from the image, then pre-process
        # it for the network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        # make a prediction on the ROI, then lookup the class
        # label
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]


 
        #for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
        #text = "{}: {:.2f}%".format(emotion, prob * 100)
        prob = max(preds)
        prob = prob*100
        final_text = str(label) + " - " +str(prob) + " %"

        cv2.putText(frameClone, final_text, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('Face', frameClone)

    if len(rects) > 0:
        print (label,prev_label)
        if (str(prev_label) == str(label)):
            count += 1
            print (count)
            #prev_label = label
        else :
            count = 0
        prev_label = label

        if count > 50:
            if (label == "happy"):
                cv2.putText(canvas, 'Playing Happy Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                print ("playing Happy songs")
                webbrowser.open('https://www.youtube.com/watch?v=OYuRPH17pLs')    
                time.sleep(2)
                
            elif (label == "angry"):
                cv2.putText(canvas, 'Playing Angry Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                print ("Playing angry songs")
                webbrowser.open('https://www.youtube.com/watch?v=wnj_y2ZDjBI&t=58s')    
                time.sleep(2)
                
            elif (label == "sad"):
                cv2.putText(canvas, 'Playing Sad Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                webbrowser.open('https://www.youtube.com/watch?v=JKCcTZOZYx8')    
                time.sleep(2)
                
            elif (label == "scared"):
                cv2.putText(canvas, 'Playing scared Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                print ("playing scared songs")
                webbrowser.open('https://www.youtube.com/watch?v=lTKeVdbiLjA')    
                time.sleep(2)
                
            elif (label == "surprised"):
                cv2.putText(canvas, 'Playing surprised Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                print ("playing surprised songs")
                webbrowser.open('https://youtu.be/i4c4uP04iBk')    
                time.sleep(2)
                

            elif (label == "disgust"):
                cv2.putText(canvas, 'Playing disgust Songs', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face", canvas)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                count = 0
                print ("playing disgust songs")
                webbrowser.open('https://www.youtube.com/watch?v=ZAfAud_M_mg')    
                time.sleep(2)
                


    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
