import cv2
import numpy as np
import pickle
import cvzone

cap = cv2.VideoCapture('easy1.mp4')

drawing = False
areaName = []

try:
    with open("computervision",'rb') as f:
                data = pickle.load(f)
                polylines , areaName = data['polylines'] , data['areaName']
except:
    polylines = []

points = []
current_name = " "

def draw(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = input("areaName")
        if current_name:
            areaName.append(current_name)
            polylines.append(np.array(points, np.int32))

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1200, 700))

    for i, polyline in enumerate (polylines):
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame,f'{areaName[i]}' , tuple(polyline[0]),1,1)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        with open("computervision",'wb') as f:
            data = {'polylines':polylines,'areaName':areaName}
            pickle.dump(data,f)
            
    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
