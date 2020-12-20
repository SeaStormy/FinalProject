import cv2
import os
vidcap = cv2.VideoCapture('D:/MachineLearning/Dataset/video5_Trim.mp4')
success,image = vidcap.read()
count = 0
count1 = 0
path = 'D:/MachineLearning/Dataset/VehicleDataset3'
while success:
  if count%600 == 0:
    count1 += 1
    cv2.imwrite(os.path.join(path, "frame%d.jpg" % count1), image) # save frame as JPEG file
  success,image = vidcap.read()
  count += 1