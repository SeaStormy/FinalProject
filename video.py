'''FINAL PROJECT
VEHICLE COUNTING AND ORIENTATION DETECTION
SVTH: PHAM MINH HAI - PHAM VAN HUY
'''


#Import library
import cv2
import time
import numpy as np
import argparse
from sort import *
import json
from bb_polygon import *
import math




# Load YOLO model
weightsPath = 'yolov4-custom_last.weights'
configPath = 'yolov4-custom.cfg'

# weightsPath = 'yolov3.weights'
# configPath = 'yolov3.cfg'
net = cv2.dnn.readNet(configPath, weightsPath)

# Load class and color of object
#YOLOv4
classes = ["motorbike","car","truck","bus"]
colors = {"motorbike":(255,255,0),"car":(255,0,255),"truck":(0,255,255),"bus":(0,0,255)}

#YOLOv3
# classes = None
#
# with open("classes.txt", 'r') as f:
#     classes = [line.strip() for line in f.readlines()]
#
# colors = np.random.uniform(0, 255, size=(len(classes), 3))



np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



#Instantiate trackers and variables
moto_tracker = Sort()
car_tracker = Sort()
truck_tracker = Sort()
bus_tracker = Sort()


memory = {}
vector_list = []
track_dict0 = {}
track_dict1 = {}
track_dict2 = {}
track_dict3 = {}


paths_list = []
MOTO_CLASS_ID = 1
CAR_CLASS_ID = 2
TRUCK_CLASS_ID = 3
BUS_CLASS_ID = 4
frame_id = 1




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
polygon, paths = load_zone_anno('sample_02.json')
pts = np.array(polygon)
pts = pts.reshape((-1, 1, 2))

#convert path into list
for key, value in paths.items():
    temp = value
    paths_list.append(temp)
    paths_list = list(paths_list)
print(paths_list)



def cosin_similarity(a2d, b2d):

    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def counting_moi(paths, vector_list, CLASS_ID):
    """

    Args:
      paths: List of direction and movement id
      vector_list: vehicle movement vector

    Returns:
      movement id and frame with respective vehicle class
    """
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
        moi_detection_list.append((last_frame, movement_id, CLASS_ID))
    return moi_detection_list


def drawBox(img, bbox):
    """
    DRAW A RECTANGLE BOX AROUND THE OBJECT
    DETECTED
    Args:
      img: frame read from the file
      bbox: A tuple (xmin, ymin, xmax, ymax)

    """
    x, y , w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 255), 3, 1)
    cv2.circle(img, (int(x+w/2), int(y+h/2)), 4, (0, 255, 0), -1)


def append_track_dict(trackers, track_dict):
    """

    Args:
      trackers: vehicle trackers
      track_dict: a dictionary of bbox and frame id

    Returns:

    """
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
    counter = 0

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
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    dets = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            #color using for YOLOv4
            color=colors[label]

            #color using for YOLOv3
            # color = colors[class_ids[i]]

            bbox = (x,y,w,h)


            drawBox(img, bbox)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, label, (x, y-2), font, 0.5, color, 1)
            text = "{:.2f}%".format(confidences[i]*100)
            cv2.putText(img, text, (x, y - 15), font, 0.5, color, 1)


            if check_bbox_intersect_polygon(polygon, bbox):
                counter += 1
                dets.append((frame_id, class_ids[i], x, y, x + w, y + h, confidences[i]))


    # Tracking vehicle MOI in ROI
    dets = np.array(dets)

    #Only tracking if detect vehicle inside ROI
    if (len(dets)>0):
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

    cv2.putText(img, "Number of vehicle in ROI: "+ str(counter), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,100), 1)
    frame_id +=1

    return img


def tracking_moi(track_dict, vehicle_vector_list):
    for tracker_id, tracker_list in track_dict.items():
        if len(tracker_list) > 1:
            first = tracker_list[0]
            last = tracker_list[-1]
            first_point = ((first[2] - first[0]) / 2, (first[3] - first[1]) / 2)
            last_point = ((last[2] - last[0]) / 2, (last[3] - last[1]) / 2)
            vehicle_vector_list.append((first_point, last_point, last[4]))
    return  vehicle_vector_list


def summary(vehicle_moi_detection):
    huong1 = 0
    huong2 = 0
    huong3 = 0
    huong4 = 0
    huong5 = 0
    huong6 = 0

    for element in vehicle_moi_detection:
        if element[1] == '1':
            huong1 +=1
        if element[1] == '2':
            huong2 +=1
        if element[1] == '3':
            huong3 +=1
        if element[1] == '4':
            huong4 +=1
        if element[1] == '5':
            huong5 +=1
        if element[1] == '6':
            huong6 +=1
    return huong1, huong2, huong3, huong4, huong5, huong6

if __name__ == '__main__':

    cap = cv2.VideoCapture('video2.mp4')
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    while(cap.isOpened()):

        ret, frame = cap.read()
        cv2.polylines(frame,[pts],True, (0,255,0),1)
        if ret == True:
            predict(frame)

            #Draw paths
            for element in paths_list:
                cv2.arrowedLine(frame, element[0], element[1],(25,255,255), 1, tipLength= 0.05)
                cv2.putText(frame, str(paths_list.index(element)+1), (int((element[0][0]+element[1][0])/2),int((element[0][1]+element[1][1])/2)), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,255), 2)


            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):

                # moto_vector_list = []
                # car_vector_list = []
                # truck_vector_list = []
                # bus_vector_list = []
                #
                # tracking_moi(track_dict0, moto_vector_list)
                # tracking_moi(track_dict1, car_vector_list)
                # tracking_moi(track_dict2, truck_vector_list)
                # tracking_moi(track_dict3, bus_vector_list)
                #
                # moto_moi_detections = counting_moi(paths, moto_vector_list, MOTO_CLASS_ID)
                # car_moi_detections = counting_moi(paths, car_vector_list,CAR_CLASS_ID)
                # truck_moi_detections = counting_moi(paths, truck_vector_list, TRUCK_CLASS_ID)
                # bus_moi_detections = counting_moi(paths, bus_vector_list, BUS_CLASS_ID)
                #
                # a, b, c, d, e, f = summary(moto_moi_detections)
                # print("So xe may di theo huong 1: ", a)
                # print("So xe may di theo huong 2: ", b)
                # print("So xe may di theo huong 3: ", c)
                # print("So xe may di theo huong 4: ", d)
                # print("So xe may di theo huong 5: ", e)
                # print("So xe may di theo huong 6: ", f)
                #
                # result_filename = 'result.txt'
                # video_id = 'sample_02'
                # with open(result_filename, 'a') as result_file:
                #     for frame_id, movement_id, vehicle_class_id in moto_moi_detections or car_moi_detections:
                #         result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))
                break
        else:
            moto_vector_list = []
            car_vector_list = []
            truck_vector_list = []
            bus_vector_list = []

            tracking_moi(track_dict0, moto_vector_list)
            tracking_moi(track_dict1, car_vector_list)
            tracking_moi(track_dict2, truck_vector_list)
            tracking_moi(track_dict3, bus_vector_list)

            moto_moi_detections = counting_moi(paths, moto_vector_list, MOTO_CLASS_ID)
            car_moi_detections = counting_moi(paths, car_vector_list, CAR_CLASS_ID)
            truck_moi_detections = counting_moi(paths, truck_vector_list, TRUCK_CLASS_ID)
            bus_moi_detections = counting_moi(paths, bus_vector_list, BUS_CLASS_ID)

            a, b, c,d,e,f = summary(moto_moi_detections)
            print("So xe may di theo huong 1: ", a)
            print("So xe may di theo huong 2: ", b)
            print("So xe may di theo huong 3: ", c)
            print("So xe may di theo huong 4: ", d)
            print("So xe may di theo huong 5: ", e)
            print("So xe may di theo huong 6: ", f)


            result_filename = 'result.txt'
            video_id = 'sample_02'
            with open(result_filename, 'a') as result_file:
                for frame_id, movement_id, vehicle_class_id in moto_moi_detections:
                    result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))
                for frame_id, movement_id, vehicle_class_id in car_moi_detections:
                    result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))
                for frame_id, movement_id, vehicle_class_id in truck_moi_detections:
                    result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))
                for frame_id, movement_id, vehicle_class_id in bus_moi_detections:
                    result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))
            break

    cap.release()
