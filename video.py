# Run: python3 video.py --video videos/video1.mp4
import cv2
import time
import numpy as np
import argparse
from sort import *
import json
from bb_polygon import *

tracker = Sort()
memory = {}
line = []
counter = 0

# Load yolov4-custom weight and config of dataset Vietnam's traffic
weightsPath = 'yolov4-custom_last.weights'
configPath = 'yolov4-custom.cfg'
net = cv2.dnn.readNet(configPath, weightsPath)

# Load class and color of object
classes = ["motorbike","car","truck","bus"]
colors = {"motorbike":(255,255,0),"car":(255,0,255),"truck":(0,255,255),"bus":(0,0,255)}

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def check_bbox_intersect_polygon(polygon, bbox):
    """

    Args:
      polygon: List of points (x,y)
      bbox: A tuple (xmin, ymin, xmax, ymax)

    Returns:
      True if the bbox intersect the polygon
    """
    x1, y1, x2, y2 = bbox
    bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return is_bounding_box_intersect(bb, polygon)

def load_zone_anno(json_filename):
    """
    Load the json with ROI and MOI annotation.

    """
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
        paths = {}
        for it in dd['shapes'][1:]:
            kk = str(int(it['label'][-2:]))
            paths[kk] = [(int(x), int(y)) for x, y
                         in it['points']]
    return polygon, paths

polygon, paths = load_zone_anno('sample_02.json')
arr = np.array(polygon)
pts = np.array(arr, np.int32)
pts = pts.reshape((-1, 1, 2))
print(paths)


def predict(img):
    counter = 0
    height, width, channels = img.shape

    # Detecting objects
    start_time=time.time()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224,224), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    end=time.time()
    # Processing speed (FPS)
    print("FPS : {:.2}".format(1/(end-start_time)))

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    class_ids = []
    confidences = []
    boxes = []

    # loop over each of the layer outputs
    for out in outs:
        # loop over each of the detections
        for detection in out:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if check_bbox_intersect_polygon(polygon, (x,y,x+w,y+h)):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color=colors[label]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, label, (x, y-2), font, 0.5, color, 1)
            text = "{:.2f}%".format(confidences[i]*100)
            cv2.putText(img, text, (x, y - 15), font, 0.5, color, 1)
            cv2.polylines(img, [pts],True, (0,255,255), 1)

    return img


if __name__ == "__main__":

    cap = cv2.VideoCapture('video2.mp4')

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = predict(frame)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
