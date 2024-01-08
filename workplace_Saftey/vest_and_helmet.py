#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import torch
import numpy as np
import os

os.chdir(r'G:\courses\machine and Deep Learning\Computer vision\احمد ابراهيم\work\204-workplace safety - computer vision')
# Set working directory to the folder containing the model and video file


# Load the model with error handling
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best final.pt', force_reload=True)
except Exception as e:
    print("Error loading the model:", e)
    exit()

# Load the video file
video_file = r"video2.ts"
cap = cv2.VideoCapture(video_file)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output.mp4'
out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (1020, 600))

# Define font for labeling
font = cv2.FONT_HERSHEY_SIMPLEX
frame_idx = 0


# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    # Detect safety helmets, vests, and workers
    results = model(frame)
    #print("results",results.pred[0])#will contain the cordinates then confidance then predected class
    
    pred = results.pred[0].cpu().numpy()#in case of tensor :first should convert it to cpu then numpy 
    #print("pred",pred)
    helmets = pred[pred[:, 5] == 0]
    #print("helmets",helmets)
    vests = pred[pred[:, 5] == 1]
    workers = pred[pred[:, 5] == 2]

    # Draw bounding boxes and labels for helmets and vests
    for helmet in helmets:
        x1, y1, x2, y2 = map(int, helmet[:4])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Helmet', (x1, y1-10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    for vest in vests:
        x1, y1, x2, y2 = map(int, vest[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Vest', (x1, y1-10), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # Check for workers without safety gear and save their images
    for worker_idx, worker in enumerate(workers):
        x1, y1, x2, y2 = map(int, worker[:4])
        has_helmet = False
        has_vest = False

        # Check if worker has safety helmet or vest
        for helmet in helmets:
            if x1 <= helmet[2] and x2 >= helmet[0] and y1 <= helmet[3] and y2 >= helmet[1]:
                has_helmet = True
                break

        for vest in vests:
            if x1 <= vest[2] and x2 >= vest[0] and y1 <= vest[3] and y2 >= vest[1]:
                has_vest = True
                break
                
            # If worker has no safety gear, save their image
        if not has_helmet or not has_vest:
            worker_img = frame[y1:y2, x1:x2]
            worker_type = "No Helmet and Vest" if not has_helmet and not has_vest else "No Helmet" if not has_helmet else "No Vest"
            cv2.imwrite(f'output/{frame_idx}_{worker_type}_worker_{worker_idx}.png', worker_img)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, worker_type, (x1, y1 - 25), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
    # Draw bounding boxes and labels for workers
    for worker in workers:
        x1, y1, x2, y2 = map(int, worker[:4])
        if x1 <= x2 and y1 <= y2:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (245, 222, 179), cv2.FILLED)
            cv2.putText(frame, 'Worker', (x1, y1-10), font, 0.5, (255, 0, 255), 2, cv2.LINE_AA)        

# Display the output frame with detected objects and labels
    out.write(frame)
    cv2.imshow('frame', frame)

# Quit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed successfully. Output file saved as", output_file)


# In[ ]:




