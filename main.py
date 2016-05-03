'''
Created on 02/05/2016

@author: diego
'''

import cv2
import numpy as np
import numpy.random as rnd
from numpy import linalg as LA
import itertools
from os import listdir
from os.path import isfile, join



# border_C = None
# center_C = None
# 
# border_Closed = None
# center_Closed = None
# 
# border_Opened = None
# center_Opened = None
# 
# border_Pointer = None
# center_Pointer = None
# 
# border_V = None
# center_V = None



def getPalmCenter(frame,border):
    C0 = np.nonzero(border)[:][0]
    C1 = np.nonzero(border)[:][1]
    
    frame_noborder = frame - border
                
    I0 = np.nonzero(frame_noborder)[:][0]
    I1 = np.nonzero(frame_noborder)[:][1]

    d = dict()
    d_min = dict()
    d_max = dict()
    d_max_pos = dict()
    
    for j in range(0,len(I0)):
        i = I0[j], I1[j]
        #print i
        
        d[j] = 0
        d_min[j] = 10000
        d_max[j] = 0
        d_max_pos[j] = i
        
        for m in range(0,len(C0),5):
            c = C0[m], C1[m]
            
            dist = sqrt((i[0] - c[0])*(i[0] - c[0]) + (i[1] - c[1])*(i[1] - c[1]))
            
             
            if(dist < d_min[j]):
                d_min[j] = dist
                d[j] = i
            
            #if(d_min[j] > d_max[j]):
                #d_max[j] = d_min[j]
                #d_max_pos[j] = i
    
    
    center = 0
    center_pos = 0
    
    for o in range(0,len(d_min)):
        if(d_min[o] > center):
            center = d_min[o]
            center_pos = d[o]

    return center_pos



def getBorder(image):
    border = image.copy()
    
    #contours, hierarchy = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(border,contours,-1,255,0)
    
    border = cv2.Canny(border, 0, 0)
    
    return border
    
    

def checkCreateDTW(frame):
    pass
    
    
    

def getImages(mypath):
    files = getFiles(mypath)
    
    images = list()
    
    for file in files:
        images.append(cv2.imread(join(mypath,file),-1))
        
    return images
    
    

def getFiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    return onlyfiles



if __name__ == "__main__":
    
    directory = 'Opened'
    initialFrame = 100
    
    print "Started"
    
    from pylab import *
    
    images_model_C = getImages("Models/C")
    images_model_Closed = getImages("Models/Closed")
    images_model_Opened = getImages("Models/Opened")
    images_model_Pointer = getImages("Models/Pointer")
    images_model_V = getImages("Models/V")
    
    #for frame in images_model_V:
    #    cv2.imshow('frame',frame)
    #    cv2.waitKey(10)
    
    frames = list()
    
    nFiles = len([f for f in listdir('TrackingFrames/' + directory) if isfile(join('TrackingFrames/' + directory, f))])
    
    for k in range(initialFrame,nFiles+1):
        if k > initialFrame: 
            if k < 10:
                filename = 'TrackingFrames/' + directory + '/0000' + str(k) + '.png'
            elif k < 100:
                filename = 'TrackingFrames/' + directory + '/000' + str(k) + '.png'
            elif k < 1000:
                filename = 'TrackingFrames/' + directory + '/00' + str(k) + '.png'
            elif k < 10000:
                filename = 'TrackingFrames/' + directory + '/0' + str(k) + '.png'
            else:
                filename = 'TrackingFrames/' + directory + '/' + str(k) + '.png'
            
            frame = cv2.imread(filename,-1)
            frames.append(frame)
    
    #for frame in frames:
    #    cv2.imshow('frame',frame)
    #    cv2.waitKey(10)
    
    total_frames = len(frames)
    
    print 'Directory:', directory
    print 'Frames:', total_frames
    
    border_C = getBorder(images_model_C[39])
    center_C = getPalmCenter(images_model_C[39],border_C)
    C0_C = np.nonzero(border_C)[:][0]
    C1_C = np.nonzero(border_C)[:][1]
        
    border_Closed = getBorder(images_model_Closed[39])
    center_Closed = getPalmCenter(images_model_Closed[39],border_Closed)
    C0_Closed = np.nonzero(border_Closed)[:][0]
    C1_Closed = np.nonzero(border_Closed)[:][1]
    
    border_Opened = getBorder(images_model_Opened[39])
    center_Opened = getPalmCenter(images_model_Opened[39],border_Opened)
    C0_Opened = np.nonzero(border_Opened)[:][0]
    C1_Opened = np.nonzero(border_Opened)[:][1]
    
    border_Pointer = getBorder(images_model_Pointer[39])
    center_Pointer = getPalmCenter(images_model_Pointer[39],border_Pointer)
    C0_Pointer = np.nonzero(border_Pointer)[:][0]
    C1_Pointer = np.nonzero(border_Pointer)[:][1]
    
    border_V = getBorder(images_model_V[39])
    center_V = getPalmCenter(images_model_V[39],border_V)
    C0_V = np.nonzero(border_V)[:][0]
    C1_V = np.nonzero(border_V)[:][1]
    
    features = dict()
    
    for num_frame in range(39,len(frames)):
        print "Frame:", num_frame, "/", total_frames
        
        frame = frames[num_frame]
        
        #checkCreateDTW(current_frame)
        border = getBorder(frame)
        center = getPalmCenter(frame,border)
        
        features_vector = list()
        
        C0 = np.nonzero(border)[:][0]
        C1 = np.nonzero(border)[:][1]
        
        dist_C = 0
        dist_Closed = 0
        dist_Opened = 0
        dist_Pointer = 0
        dist_V = 0
        
        for pt in range(0,len(C0),10):
            if(pt < len(C0_C) and pt < len(C0_Closed) and pt < len(C0_Opened) and pt < len(C0_Pointer) and pt < len(C0_V)):
                c = C0[pt] - center[0], C1[pt] - center[1]
                
                c_C = C0_C[pt] - center_C[0], C1_C[pt] - center_C[1]                
                dist_C = dist_C + (c[0] - c_C[0])*(c[0] - c_C[0]) + (c[1] - c_C[1])*(c[1] - c_C[1])
                
                c_Closed = C0_Closed[pt] - center_Closed[0], C1_Closed[pt] - center_Closed[1]                
                dist_Closed = dist_Closed + (c[0] - c_Closed[0])*(c[0] - c_Closed[0]) + (c[1] - c_Closed[1])*(c[1] - c_Closed[1])
                
                c_Opened = C0_Opened[pt] - center_Opened[0], C1_Opened[pt] - center_Opened[1]                
                dist_Opened = dist_Opened + (c[0] - c_Opened[0])*(c[0] - c_Opened[0]) + (c[1] - c_Opened[1])*(c[1] - c_Opened[1])
                
                c_Pointer = C0_Pointer[pt] - center_Pointer[0], C1_Pointer[pt] - center_Pointer[1]                
                dist_Pointer = dist_Pointer + (c[0] - c_Pointer[0])*(c[0] - c_Pointer[0]) + (c[1] - c_Pointer[1])*(c[1] - c_Pointer[1])
                
                c_V = C0_V[pt] - center_V[0], C1_V[pt] - center_V[1]                
                dist_V = dist_V + (c[0] - c_V[0])*(c[0] - c_V[0]) + (c[1] - c_V[1])*(c[1] - c_V[1])
            
        
        dist_C = sqrt(dist_C)
        dist_Closed = sqrt(dist_Closed)
        dist_Opened = sqrt(dist_Opened)
        dist_Pointer = sqrt(dist_Pointer)
        dist_V = sqrt(dist_V)
        
        print "dist_C:", dist_C
        print "dist_Closed:", dist_Closed
        print "dist_Opened:", dist_Opened
        print "dist_Pointer:", dist_Pointer
        print "dist_V:", dist_V
        print "--"
        
        
        if(center):
            border = cv2.cvtColor(border,cv2.COLOR_GRAY2RGB)
            cv2.circle(border,(center[1],center[0]), 4, (0,255,0), -1)
            #print (center_pos[1],center_pos[0])
                        
        cv2.imshow('border',border)
        #cv2.imshow('frame_noborder',frame_noborder)
        #cv2.imshow('frame1',frame1)
        #cv2.imshow('border_frame2',border_frame2)
        cv2.waitKey(1)
    
    
    
    
    