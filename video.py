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
weightsPath = 'yolov4-custom_last.weights'
configPath = 'yolov4-custom.cfg'
net = cv2.dnn.readNet(configPath, weightsPath)

# Load class and color of object
classes = ["motorbike","car","truck","bus"]
colors = {"motorbike":(255,255,0),"car":(255,0,255),"truck":(0,255,255),"bus":(0,0,255)}

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


tracker = Sort()
memory = {}
line = [(7, 441), (1000, 439)]
counter = 0
vector_list = []
track_dict = {}
paths_list = []




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
polygon, paths = load_zone_anno('sample_02.json')




for key, value in paths.items():
    temp = value
    paths_list.append(temp)
    paths_list = list(paths_list)



frame_id = 0


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

def is_old(center_Xd, center_Yd, boxes):
    for box_tracker in boxes:
        (xt, yt, wt, ht) = [int(c) for c in box_tracker]
        center_Xt, center_Yt = int((xt + (xt + wt)) / 2.0), int((yt + (yt + ht)) / 2.0)
        distance = math.sqrt((center_Xt - center_Xd) ** 2 + (center_Yt - center_Yd) ** 2)

        if distance < 50:
            return True
    return False


def predict(img):
    global counter
    height, width, channels = img.shape

    # Detecting objects
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
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
                #if check_bbox_intersect_polygon(polygon, (x, y, x + w, y + h)):
                boxes.append([x, y, w, h])
                print(boxes)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         color=colors[label]
    #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(img, label, (x, y-2), font, 0.5, color, 1)
    #         text = "{:.2f}%".format(confidences[i]*100)
    #         #text1 = "{}".format(class_ids[i])
    #         cv2.putText(img, text, (x, y - 15), font, 0.5, color, 1)
    #
    #         if check_bbox_intersect_polygon(polygon, (x,y,x+w,y+h)):
    #             dets.append((frame_id, class_ids[i], x, y, x+w, y+h, confidences[i]) )
    #         if onSegment(polygon[3], (x,y), polygon[0]):
    #             counter +=1
    #
    # cv2.putText(img, str(counter), (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

    return boxes

def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_x = int((x + w / 2.0))
    center_y = int((y +h / 2.0))
    return x, y, w, h, center_x, center_y


max_distance = 50
input_h = 360
input_w = 460
laser_line = input_h - 50
# Khoi tao tham so
frame_count = 0
car_number = 0
obj_cnt = 0
curr_trackers = []



while True:

    cap = cv2.VideoCapture('video2.mp4')

    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    while(cap.isOpened()):
        laser_line_color = (0, 0, 255)
        boxes = []
        ret, frame = cap.read()
        if ret == True:

            # Resize nho lai
            frame = cv2.resize(frame, (input_w, input_h))
            # Duyet qua cac doi tuong trong tracker
            old_trackers = curr_trackers
            curr_trackers = []


            for car in old_trackers:
                # Update tracker
                tracker = car['tracker']
                (_, box) = tracker.update(frame)
                boxes.append(box)

                new_obj = dict()
                new_obj['tracker_id'] = car['tracker_id']
                new_obj['tracker'] = tracker

                # Tinh toan tam doi tuong
                x, y, w, h, center_x, center_y = get_box_info(box)

                # Ve hinh chu nhat quanh doi tuong
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Ve hinh tron tai tam doi tuong
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

                # So sanh tam doi tuong voi duong laser line
                if center_y > laser_line:
                    # Neu vuot qua thi khong track nua ma dem xe
                    laser_line_color = (0, 255, 255)
                    car_number += 1
                else:
                    # Con khong thi track tiep
                    curr_trackers.append(new_obj)


            # Thuc hien object detection moi 1 frame
            if frame_count % 1 == 0:
                # Detect doi tuong
                boxes_d = predict(frame)

                for box in boxes_d:
                    old_obj = False

                    xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(box)

                    if center_Yd <= laser_line - max_distance:

                        # Duyet qua cac box, neu sai lech giua doi tuong detect voi doi tuong da track ko qua max_distance thi coi nhu 1 doi tuong
                        if not is_old(center_Xd, center_Yd, boxes):
                            cv2.rectangle(frame, (xd, yd), ((xd + wd), (yd + hd)), (0, 255, 255), 1)
                            # Tao doi tuong tracker moi

                            tracker = cv2.TrackerMOSSE_create()

                            obj_cnt += 1
                            new_obj = dict()
                            tracker.init(frame, tuple(box))

                            new_obj['tracker_id'] = obj_cnt
                            new_obj['tracker'] = tracker

                            curr_trackers.append(new_obj)

            # Tang frame
            frame_count += 1

            # Hien thi so xe

            cv2.putText(frame, "Number of vehicle: " + str(car_number), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0),
                        1)
            cv2.putText(frame, "Press Esc to quit", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

            # Draw laser line
            cv2.line(frame, (0, laser_line), (input_w, laser_line), laser_line_color, 2)
            cv2.putText(frame, "Line", (10, laser_line - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, laser_line_color, 1)



            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:

            break
    cap.release()
