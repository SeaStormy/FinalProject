# Run: python3 video.py --video videos/video1.mp4
import cv2
import time
import numpy as np
import argparse
from sort import *
import json
from bb_polygon import *
import math




# Load yolov4-custom weight and config of dataset Vietnam's traffic
# weightsPath = 'yolov4-custom_last.weights'
# configPath = 'yolov4-custom.cfg'

weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'
net = cv2.dnn.readNet(configPath, weightsPath)

# Load class and color of object
#YOLOv4
#classes = ["motorbike","car","truck","bus"]
#colors = {"motorbike":(255,255,0),"car":(255,0,255),"truck":(0,255,255),"bus":(0,0,255)}

#YOLOv3
classes = None

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



moto_tracker = Sort()
car_tracker = Sort()
truck_tracker = Sort()
bus_tracker = Sort()


memory = {}
counter = 0
vector_list = []
track_dict0 = {}
track_dict1 = {}
track_dict2 = {}
track_dict3 = {}


paths_list = []
MOTO_CLASS_ID = 1




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


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


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


#load and convert polygon and paths to numpy array
polygon, paths = load_zone_anno('sample_01.json')
pts = np.array(polygon)
pts = pts.reshape((-1, 1, 2))


for key, value in paths.items():
    temp = value
    paths_list.append(temp)
    paths_list = list(paths_list)



frame_id = 1


def cosin_similarity(a2d, b2d):
    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def counting_moi(paths, vector_list):
    moi_detection_list = []
    for moto_vector in vector_list:
        max_cosin = -2
        movement_id = ''
        last_frame = 0
        for movement_label, movement_vector in paths.items():
            cosin = cosin_similarity(movement_vector, moto_vector)
            if cosin > max_cosin:
                max_cosin = cosin
                movement_id = movement_label
                last_frame = moto_vector[2]
        moi_detection_list.append((last_frame, movement_id))
    return moi_detection_list


def drawBox(img, bbox):
    x, y , w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 255), 3, 1)
    cv2.circle(img, (int(x+w/2), int(y+h/2)), 4, (0, 255, 0), -1)


def append_track_dict(trackers, track_dict):
    for xmin, ymin, xmax, ymax, track_id in trackers:
        track_id = int(track_id)

        if track_id not in track_dict.keys():
            track_dict[track_id] = [(xmin, ymin, xmax, ymax, frame_id)]
        else:
            track_dict[track_id].append((xmin, ymin, xmax, ymax, frame_id))



def predict(img):
    global frame_id
    height, width, channels = img.shape

    # Detecting objects
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) #complete detected
    end = time.time()
    # Processing speed (FPS)
    print("FPS : {:.4}".format(1 / (end - start_time)))

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
            scores = detection[5:]  # show the probability of 4 class for 1 object [a b c d]
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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #class ID: 0: motorbike, 1: car, 2: truck, 3: bus
    #using non maximum suppression algorithm to remove overlap bounding box
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print("class_ids",class_ids)
    dets = []
    for bb, s, c in zip(boxes, confidences, class_ids):
        xmin, ymin, w, h = bb
        if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmin+w, ymin+h)):
            dets.append((frame_id, c, xmin, ymin, xmin+w, ymin+h, s))
    print("dets: ", dets)



    dets = np.array(dets)
    moto_dets = dets[dets[:, 1] == 0]
    car_dets = dets[dets[:, 1] == 1]
    truck_dets = dets[dets[:, 1] == 2]
    bus_dets = dets[dets[:, 1] == 3]

    #extract bbox parameters for 4 classes
    moto_dets = moto_dets[:,2:6]
    car_dets = car_dets[:, 2:6]
    truck_dets = truck_dets[:, 2:6]
    bus_dets = bus_dets[:, 2:6]


    #covert to np array
    moto_dets = np.array(moto_dets)
    car_dets = np.array(car_dets)
    truck_dets = np.array(truck_dets)
    bus_dets = np.array(bus_dets)


    #tracking object
    trackers0 = moto_tracker.update(moto_dets)
    trackers1 = car_tracker.update(car_dets)
    trackers2 = truck_tracker.update(truck_dets)
    trackers3 = bus_tracker.update(bus_dets)

    #write to track_dict
    append_track_dict(trackers0, track_dict0)
    append_track_dict(trackers1, track_dict1)
    append_track_dict(trackers2, track_dict2)
    append_track_dict(trackers3, track_dict3)



    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            #color using for YOLOv4
            #color=colors[label]

            #color using for YOLOv3
            color = colors[class_ids[i]]

            bbox = (x,y,w,h)


            drawBox(img, bbox)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, label, (x, y-2), font, 0.5, color, 1)
            text = "{:.2f}%".format(confidences[i]*100)
            cv2.putText(img, text, (x, y - 15), font, 0.5, color, 1)

    frame_id +=1
    return img




if __name__ == '__main__':

    cap = cv2.VideoCapture('video4.mp4')
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    while(cap.isOpened()):

        ret, frame = cap.read()
        cv2.polylines(frame,[pts],True, (0,255,0),1)
        if ret == True:
            predict(frame)


            print("track_dict0: ", track_dict0)
            print("track_dict1: ", track_dict1)
            print("track_dict2: ", track_dict2)
            print("track_dict3: ", track_dict3)




            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                moto_vector_list = []
                for tracker_id, tracker_list in track_dict0.items():
                    if len(tracker_list) > 1:
                        first = tracker_list[0]
                        last = tracker_list[-1]
                        first_point = ((first[2] - first[0]) / 2, (first[3] - first[1]) / 2)
                        last_point = ((last[2] - last[0]) / 2, (last[3] - last[1]) / 2)
                        moto_vector_list.append((first_point, last_point, last[4]))
                print(moto_vector_list)
                moto_moi_detections = counting_moi(paths, moto_vector_list)
                print("moto_moi_detections", moto_moi_detections)
                break
        else:

            break

    cap.release()
