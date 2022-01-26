import streamlit as st
import numpy as np
import imutils
import winsound
import cv2
from itertools import combinations
import math

MIN_CONF = 0.3
NMS_THRESH = 0.3

def is_close(p1, p2):
    dst = math.sqrt(p1**2 + p2**2)
    return dst 

st.title("Social Distancing Detector")

st.subheader('Test on a video or use webcam')
option = st.selectbox('Choose your option', ('Demo1', 'Demo2', 'Try Live Detection Using Webcam'))


MIN_DISTANCE = 50

file_name="yolo.weights"
labelsPath = "Files/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = file_name
configPath = "Files/yolov3.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if st.button('Start'):
    st.info("Loading Video")
    if option == "Demo1":
        vs = cv2.VideoCapture("Videos/testvideo2.mp4")
    elif option == "Demo2":
        vs = cv2.VideoCapture("Videos/pedestrians.mp4")
    else:
        vs = cv2.VideoCapture(0)

    image_placeholder = st.empty()
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
    
        (H, W) = frame.shape[:2]
        results = []

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == LABELS.index("person") and confidence > MIN_CONF:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

        if len(idxs) > 0:

            for i in idxs.flatten():

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                r = (confidences[i], (x, y, x+w, y+h), centroids[i])
                results.append(r)

        if len(results) >= 2:
            centroid_dict = dict() 
            objectId = 0				
            for detection in results:				
                x,y=detection[2][0],detection[2][1]
                xmin, ymin, xmax, ymax = detection[1][0], detection[1][1], detection[1][2],detection[1][3]
                confidence = detection[0] 
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax,confidence)
                objectId += 1

            red_zone_list = []
            red_line_list = []
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]  
                distance = is_close(dx, dy) 		
                if distance < MIN_DISTANCE:				
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)     
                        red_line_list.append(p1[0:2])
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)		
                        red_line_list.append(p2[0:2])
            
            for idx, box in centroid_dict.items():
                cv2.putText(frame,str(format(box[6]*100,".1f")),(box[2]+1,box[3]-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
                if idx in red_zone_list:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                    winsound.Beep(440, 1000)
                else:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
            
            text = "Violations: {}".format(len(red_zone_list))
            fps = vs.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame,"FPS : " + str(fps),(2,25),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
            cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            
        
        image_placeholder.image(frame, channels="BGR")

st.success("Made By Tanmay Anand")
