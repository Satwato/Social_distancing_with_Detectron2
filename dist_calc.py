import math
import itertools
import cv2 
import numpy as np


def mid_point(img,person,idx):
#get the coordinates
    x1,y1,x2,y2 = person[idx]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

    #compute bottom center of bbox
    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid   = (x_mid,y_mid)

    cv2.circle(img, mid, 5, (0, 0, 255), -1)
    cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)

    return mid

from scipy.spatial import distance
def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist

def find_closest(dist,num,thresh):
  p1=[]
  p2=[]
  d=[]
  for i in range(num):
    for j in range(i,num):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
  return p1,p2,d

def change_2_red(img,person,p1,p2):
  risky = np.unique(p1+p2)
  for i in risky:
    x1,y1,x2,y2 = person[i]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)  
  return img