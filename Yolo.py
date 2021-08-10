#Packages
import cv2 as cv
import numpy as np
import urllib.request
import threading
from pyfirmata import SERVO
import pyfirmata
import time
import paho.mqtt.publish as publish

obj=0 #No. of Objects
flag=0
j=0
k=0
bio=0 # No. of Biodegradable Objects
nbio=0 # No. of Non-Biodegradable Objects


# Robotic-Arm Setup
board = pyfirmata.Arduino('COM3')
servo = board.get_pin('d:11:o')
board.digital[11].mode = SERVO

#Capture Video
cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold=0.3

## Object Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print("classNames: ", classNames)
print(len(classNames))

recyclableFile = "recycle.names"
recyclable = []
with open(recyclableFile, 'rt') as f:
    recyclable = f.read().rstrip('\n').split('\n')
print("recyclable: ", recyclable)

# Arduino Connection
#arduinoData=serial.Serial('com3',9600)

# YOLO Model Configurations
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

# Creating Deep Neural Network(DNN)
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    global flag
    global i
    global j
    hT, wT, cT = img.shape
    bbox = [] # bounding box corner points
    classIds = []  # class id with the highest confidence
    confs = []     # confidence value of the highest class
    nbdg = [] #Non-Biodegradable
    bdg = [] #Biodegradable
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores) #Find the Maximum Score
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold) #Non-Maximum Suppression

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(classNames[classIds[i]])
        #thingspeak_post(l)
        if(classNames[classIds[i]] in recyclable):
            flag = 1
            bdg.append(classNames[classIds[i]])
        else:
            flag = 0
            nbdg.append(classNames[classIds[i]])
        thingspeak_post(l, flag)
        Robotic_Arm(flag)
        #arduinoData.write(flag)
    l = len(classIds)-3
    global obj
    obj = l
    print("Objects Scanned: ", l)
    print(bdg)
    print(nbdg)
    remove_duplicates1(bdg)
    remove_duplicates2(nbdg)
    #mqtt_publish2(i)
    #mqtt_publish3(j)

def remove_duplicates1(bdg):
    global bio
    bl=[]
    for i in bdg:
        if i not in bl:
            bl.append(i)
    bio=len(bl)
    mqtt_publish2(bio)

def remove_duplicates2(nbdg):
    global nbio
    nbl = []
    for i in nbdg:
        if i not in nbl:
            nbl.append(i)
    nbio=len(nbl)
    mqtt_publish3(nbio)

def mqtt_publish2(obj):
    publish.single("Shailesh/Bio-DG/IOT", obj, hostname="test.mosquitto.org")
    print("BDG Obj Count Done")

def mqtt_publish3(obj):
    publish.single("Shailesh/Non-BDG", obj, hostname="test.mosquitto.org")
    print("NBDG Obj Count Done")

def mqtt_publish1(obj):
    publish.single("Shailesh/Nobjects", obj, hostname="test.mosquitto.org")
    print("Obj Count Done")

def check_flag(flag):
    global k
    global j
    if flag==1:
        k=k+1
        mqtt_publish2(k)
    else:
        j=j+1
        mqtt_publish3(j)


def Robotic_Arm(f):
    if(f == 1):
        servo.write(0)
        time.sleep(1)
    else:
        servo.write(180)
        time.sleep(1)

def thingspeak_post(val1,val2):
    threading.Timer(1,thingspeak_post,[val1, val2]).start()
    URl='https://api.thingspeak.com/update?api_key='
    KEY='NJM6WXH3J936SEZU'
    HEADER='&field1={}&field2={}'.format(val1, val2)
    NEW_URL = URl + KEY + HEADER
    data = urllib.request.urlopen(NEW_URL)

# Reading Image and converting to blob
while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    cv.imshow('Image', img)
    cv.waitKey(1)
    mqtt_publish1(obj)
