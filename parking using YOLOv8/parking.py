import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone



my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model=YOLO('yolov8s.pt')

with open("computervision",'rb') as f:
                data = pickle.load(f)
                polylines , areaName = data['polylines'] , data['areaName']
            
cap=cv2.VideoCapture('easy1.mp4')




count=0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
   
    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(1200,700))
    frame_copy = frame.copy()
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    list1=[]
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2

        if 'car' in c:
            list1.append([cx,cy])
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            
    counter1 = []        
    for i, polyline in enumerate (polylines):
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame,f'{areaName[i]}' , tuple(polyline[0]),1,1)
        
        for l in list1:
            cx1 = l[0]
            cy1 = l[1]
            result = cv2.pointPolygonTest(polyline,(cx1,cy1),False)
            if result >=0:
                cv2.circle(frame,(cx1,cy1),5,(0,255,0),-1)
                cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                counter1.append(cx1)
     
    car_count = len(counter1)
    cvzone.putTextRect(frame,f'busy spots is {car_count} ' ,(50,40),2,2)
    free_space = 5 - car_count
    cvzone.putTextRect(frame,f'free spots is {free_space} ' ,(50,90),2,2)
    
    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()

