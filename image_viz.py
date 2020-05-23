import cv2
import numpy as np
from dist_calc import *
from itertools import compress

def visualize(img, instances):
    predictions=instances.pred_boxes
    score=instances.scores
    class_detected= instances.pred_classes
    c=0    
    output_dict={}
    output_dict['detection_boxes']=[]
    for i in range(len(predictions)):
        
        if score[i]>=0.50 and class_detected[i]==0:
            x1,y1,x2,y2=predictions[i].tensor.cpu().numpy().astype(np.int).tolist()[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            output_dict['detection_boxes'].append([x1,y1,x2,y2])
            c+=1
    #print(output_dict['detection_boxes'])
    height = img.shape[0]
    width = img.shape[1] 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale=(width * height) / (1000 * 1000)
    cv2.rectangle(img, (0, 0), (int(width/10),int(height/25)), (255,0,0), -1)
    strr="persons: "+str(c)
    cv2.putText(img, strr, (3,int(height/25*0.75)), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
    
    midpoints = [mid_point(img,output_dict['detection_boxes'],i) for i in range(len(output_dict['detection_boxes']))]
    num = len(midpoints)
    dist= compute_distance(midpoints,num)
    p1,p2,d=find_closest(dist,num,100)
    img = change_2_red(img,output_dict['detection_boxes'],p1,p2)
    return img
