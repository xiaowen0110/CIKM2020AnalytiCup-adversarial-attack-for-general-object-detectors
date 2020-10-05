import numpy as np
import cv2
from queue import Queue,LifoQueue,PriorityQueue

a=np.array([1,2,3])
b=np.array([a,a,a])
print(b)
print(np.sum(b[0:2,0:2],axis=0))
print(np.sum(b[0:2,0:2],axis=1))
