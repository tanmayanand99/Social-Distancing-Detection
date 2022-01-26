import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from detection import detect_people
from scipy.spatial import distance as dist
import imutils
import datetime

MIN_DISTANCE = 50


file_name="yolo.weights"
labelsPath = "data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)

weightsPath = file_name
configPath = "data/yolov3.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print(net)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


window = tk.Tk() 
window.wm_title("Physical Distancing Detection")
window.config(background="#FFFFFF")

imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)


lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)

filename=0

def UploadAction(event=None):
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    global cap
    cap =cv2.VideoCapture(filename)

cap =cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    violate = set()

    if len(results) >= 2:

        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):

                if D[i, j] < MIN_DISTANCE:

                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):

        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    datet = str(datetime.datetime.now())
    frame = cv2.putText(frame, datet, (0, 35), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
    text = "Physical Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 



sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame = tk.Button(window,text='Open',command=UploadAction)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)

show_frame()
window.mainloop() 