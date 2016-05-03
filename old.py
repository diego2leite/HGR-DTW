#!python

'''
Created on 12/04/2016

@author: diego
'''

import cv2
import numpy as np
import numpy.random as rnd
from numpy import linalg as LA
import itertools


if __name__ == "__main__":
    
    from pylab import *

    directory = 'Opened'
    
    nFiles = 2089
    initialFrame = 40
    
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
            
            
            #filename = 'hand.png'
            
            frame = cv2.imread(filename,-1)
            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)
                    
            #border = cv2.Canny(frame, 0, 0)
            
            border = frame.copy()
            contours, hierarchy = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(border,contours,-1,255,0)
                
            
            # PALM CENTER
            C0 = np.nonzero(border)[:][0]
            C1 = np.nonzero(border)[:][1]
            
            if(True):     
                frame_noborder = frame - border
                
                I0 = np.nonzero(frame_noborder)[:][0]
                I1 = np.nonzero(frame_noborder)[:][1]
            
                d = dict()
                d_min = dict()
                d_max = dict()
                d_max_pos = dict()
                
                for j in range(0,len(I0)):
                    i = I0[j], I1[j]
                 
                    d[j] = 0
                    d_min[j] = 0
                    d_max[j] = 0
                    d_max_pos[j] = i
                    
                    for m in range(0,len(C0),5):
                        c = C0[m], C1[m]
                        
                        dist = sqrt((i[0] - c[0])*(i[0] - c[0]) + (i[1] - c[1])*(i[1] - c[1]))
                        
                         
                        if(dist > d_min[j]):
                            d_min[j] = dist
                            d[j] = i 
                        
                        #if(d_min[j] > d_max[j]):
                            #d_max[j] = d_min[j]
                            #d_max_pos[j] = i
                
                
                center = 1000
                center_pos = 0
                
                for o in range(0,len(d_min)):
                    if(d_min[o] < center):
                        center = d_min[o]
                        center_pos = d[o]
            
            last = False
            
            border_points = list()
            if(len(C0)):
                border_points.append((C0[len(C0)-1], C1[len(C0)-1]))
                
                for pt in border_points:
                    upper = pt[0]-1, pt[1]
                    bottom = pt[0]+1, pt[1]
                    right = pt[0], pt[1] + 1
                    right_upper = pt[0]-1, pt[1] + 1
                    right_bottom = pt[0]+1, pt[1] + 1
                    left = pt[0], pt[1] + 1
                    left_upper = pt[0]-1, pt[1] - 1
                    left_bottom = pt[0]+1, pt[1] - 1
                    
                    added = 0
                     
                    if(border[right] == 255):
                        if(right not in border_points):
                            border_points.append(right)
                            added = added + 1
                     
                    if(border[upper] == 255):
                        if(upper not in border_points):
                            border_points.append(upper)
                            added = added + 1
                             
                    if(border[bottom] == 255):
                        if(bottom not in border_points):
                            border_points.append(bottom)
                            added = added + 1
                             
                    if(border[right_upper] == 255):
                        if(right_upper not in border_points):
                            border_points.append(right_upper)
                            added = added + 1
                             
                    if(border[right_bottom] == 255):
                        if(right_bottom not in border_points):
                            border_points.append(right_bottom)
                            added = added + 1
                    
                    if(added == 0):  
                        if(border[left] == 255):
                            if(left not in border_points):
                                border_points.append(left)
                                added = added + 1
                                  
                        if(border[left_upper] == 255):
                            if(left_upper not in border_points):
                                border_points.append(left_upper)
                                added = added + 1
                                  
                        if(border[left_bottom] == 255):
                            if(left_bottom not in border_points):
                                border_points.append(left_bottom)
                                added = added + 1
                                
                                
                    if(added == 0):
                        window = 2
                         
                        for v in range(pt[0]-window,pt[0]+window):
                            for w in range(pt[1]-window,pt[1]+window):
                                current = v,w
                                #print border[current]
                                if(border[current] == 255):
                                    if(current not in border_points):
                                        #print current[0] - border_points[len(border_points)-1][0], current[1] - border_points[len(border_points)-1][1]
                                        
                                        if(abs(current[0] - border_points[len(border_points)-1][0]) < 8):
                                            if(abs(current[1] - border_points[len(border_points)-1][1]) < 8):
                                                border_points.append(current)
                                                added = added + 1
                     
                    if(added == 0):
                        window = 3
                         
                        for v in range(pt[0]-window,pt[0]+window):
                            for w in range(pt[1]-window,pt[1]+window):
                                current = v,w
                                #print border[current]
                                if(border[current] == 255):
                                    if(current not in border_points):
                                        
                                        if(abs(current[0] - border_points[len(border_points)-1][0]) < 8):
                                            if(abs(current[1] - border_points[len(border_points)-1][1]) < 8):
                                                border_points.append(current)
                                                added = added + 1

                    if(added == 0):
                        window = 4
                        
                        for v in range(pt[0]-window,pt[0]+window):
                            for w in range(pt[1]-window,pt[1]+window):
                                current = v,w
                                #print border[current]
                                if(border[current] == 255):
                                    if(current not in border_points):
                                        
                                        if(abs(current[0] - border_points[len(border_points)-1][0]) < 8):
                                            if(abs(current[1] - border_points[len(border_points)-1][1]) < 8):
                                                border_points.append(current)
                                                added = added + 1


#                     if(added == 0):  
#                         print '???????????????/'
#                         for new_pt in [right_bottom,right_upper]:
#                             print '---------', new_pt
#                             upper = new_pt[0]-1, new_pt[1]
#                             bottom = new_pt[0]+1, new_pt[1]
#                             right = new_pt[0], new_pt[1] + 1
#                             right_upper = new_pt[0]-1, new_pt[1] + 1
#                             right_bottom = new_pt[0]+1, new_pt[1] + 1
#                             left = new_pt[0], new_pt[1] + 1
#                             left_upper = new_pt[0]-1, new_pt[1] - 1
#                             left_bottom = new_pt[0]+1, new_pt[1] - 1
#                                                          
#                             if(border[right] > 0):
#                                 if(right not in border_points):
#                                     border_points.append(right)
#                                     added = added + 1
#                              
#                             if(border[upper] > 0):
#                                 if(upper not in border_points):
#                                     border_points.append(upper)
#                                     added = added + 1
#                                      
#                             if(border[bottom] > 0):
#                                 if(bottom not in border_points):
#                                     border_points.append(bottom)
#                                     added = added + 1
#                                      
#                             if(border[right_upper] > 0):
#                                 if(right_upper not in border_points):
#                                     border_points.append(right_upper)
#                                     added = added + 1
#                                      
#                             if(border[right_bottom] > 0):
#                                 if(right_bottom not in border_points):
#                                     border_points.append(right_bottom)
#                                     added = added + 1
#                             
#                             if(added == 0):  
#                                 if(border[left] > 0):
#                                     if(left not in border_points):
#                                         border_points.append(left)
#                                         added = added + 1
#                                           
#                                 if(border[left_upper] > 0):
#                                     if(left_upper not in border_points):
#                                         border_points.append(left_upper)
#                                         added = added + 1
#                                           
#                                 if(border[left_bottom] > 0):
#                                     if(left_bottom not in border_points):
#                                         border_points.append(left_bottom)
#                                         added = added + 1
#                         
#                         
#                         
#                         for new_pt in [left_bottom,left_upper]:
#                             upper = new_pt[0]-1, new_pt[1]
#                             bottom = new_pt[0]+1, new_pt[1]
#                             right = new_pt[0], new_pt[1] + 1
#                             right_upper = new_pt[0]-1, new_pt[1] + 1
#                             right_bottom = new_pt[0]+1, new_pt[1] + 1
#                             left = new_pt[0], new_pt[1] + 1
#                             left_upper = new_pt[0]-1, new_pt[1] - 1
#                             left_bottom = new_pt[0]+1, new_pt[1] - 1
#                                                          
#                             if(border[right] > 0):
#                                 if(right not in border_points):
#                                     border_points.append(right)
#                                     added = added + 1
#                              
#                             if(border[upper] > 0):
#                                 if(upper not in border_points):
#                                     border_points.append(upper)
#                                     added = added + 1
#                                      
#                             if(border[bottom] > 0):
#                                 if(bottom not in border_points):
#                                     border_points.append(bottom)
#                                     added = added + 1
#                                      
#                             if(border[right_upper] > 0):
#                                 if(right_upper not in border_points):
#                                     border_points.append(right_upper)
#                                     added = added + 1
#                                      
#                             if(border[right_bottom] > 0):
#                                 if(right_bottom not in border_points):
#                                     border_points.append(right_bottom)
#                                     added = added + 1
#                             
#                             if(added == 0):  
#                                 if(border[left] > 0):
#                                     if(left not in border_points):
#                                         border_points.append(left)
#                                         added = added + 1
#                                           
#                                 if(border[left_upper] > 0):
#                                     if(left_upper not in border_points):
#                                         border_points.append(left_upper)
#                                         added = added + 1
#                                           
#                                 if(border[left_bottom] > 0):
#                                     if(left_bottom not in border_points):
#                                         border_points.append(left_bottom)
#                                         added = added + 1
                        
#                     if(added == 0):
#                         window = 2
#                         
#                         for v in range(pt[0]-window,pt[0]+window):
#                             for w in range(pt[1]-window,pt[1]+window):
#                                 current = v,w
#                                 print border[current]
#                                 if(border[current] == 255):
#                                     if(current not in border_points):
#                                         border_points.append(current)
#                                         added = added + 1
#                     
#                     if(added == 0):
#                         window = 3
#                         
#                         for v in range(pt[0]-window,pt[0]+window):
#                             for w in range(pt[1]-window,pt[1]+window):
#                                 current = v,w
#                                 print border[current]
#                                 if(border[current] == 255):
#                                     if(current not in border_points):
#                                         border_points.append(current)
#                                         added = added + 1
#                             
            C = border_points
            P1 = 0
            Ca = P1 - k
            Cb = P1 + k
            k = 10
            
            num_fingers = 0
            fingers = list()
            last = False
            
            for c in range(0,len(C)-1):    
                if(c < k):
                    Ca = len(C) - k + c
                else:
                    Ca = c - k
                    
                if(c + k > len(C)):
                    Cb = c + k - len(C)
                else:
                    Cb = c + k
                    
                    
                if(Cb > len(C)-1):
                    Cb = len(C)-1
                    
                print Ca, c, Cb
                
                
                #Va = np.array([C[Ca]-C[c]])
                #Vb = np.array([C[Cb]-C[c]])
                
                Va = np.array(C[Ca]) - np.array(C[c])
                Vb = np.array(C[Cb]) - np.array(C[c])
                
                print "Ca", C[Ca]
                print "Cc", C[c]
                print "Cb", C[Cb]
                print "Va", Va
                print "Vb", Vb
                
                
                #a = np.array([1,2,3])
                #b = np.array([0,1,0])
                inner_v = np.inner(Va, Vb)
                norm_v = LA.norm(Va) * LA.norm(Vb)
                
                print 'degrees', np.degrees(np.arccos(inner_v/norm_v))
                #print 'degrees', np.degrees(np.arccos(abs(inner_v/norm_v)))
                
                graus = np.degrees(np.arccos(inner_v/norm_v))

                if(graus >= 25 and graus <= 55):
                    
                    if(last):
                        if abs(last[0] - C[c][0]) > k or abs(last[1] - C[c][1]) > k:
                            fingers.append(C[c])
                            num_fingers = num_fingers + 1
                            last = C[c] 
                    else:
                        fingers.append(C[c])
                        num_fingers = num_fingers + 1
                        last = C[c]
                    
                    
                    
                    #print graus
                    
                    #border = cv2.cvtColor(border,cv2.COLOR_GRAY2RGB)
                    #cv2.circle(border,(C[c][1],C[c][0]), 3, (0,0,255), -1)
    
                    #cv2.imshow('frame', frame)
                    #cv2.waitKey(50)
                
                #Va_mod = sqrt(Va[0]*Va[0] + Va[1]*Va[1])
                #Vb_mod = sqrt(Vb[0]*Vb[0] + Vb[1]*Vb[1])
                
                
                #print Va_mod, Vb_mod
                
                print '---'
            
            print 'num_fingers', num_fingers
            
            
            # ?????????????????????????? Escolher quais matrizes DTW criar - Distancia Euclidiana dos pontos relativos da borda. Pontos da borda em relacao ao centro da mao
            for pos in range(0,len(border_points)):
                pass
                
                
                
            
                
            if(True):
                if(center_pos):
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                    cv2.circle(frame,(center_pos[1],center_pos[0]), 4, (0,255,0), -1)

            if(False):
                # Visualizar borda
                B0 = np.nonzero(border)[:][0]
                B1 = np.nonzero(border)[:][1]
                
                for ii in range(0,len(B0)):
                    #print B0[ii],B1[ii]
                    cv2.circle(frame,(B1[ii],B0[ii]), 1, (0,0,255), -1)
    
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
            
            #border_seed = B0[0], B1[0]
             
            if(True):
                #print '--'
                border = cv2.cvtColor(border,cv2.COLOR_GRAY2RGB)
                
                for pt in border_points:
                    #print pt
                    #cv2.circle(border,(pt[1],pt[0]), 1, (0,0,255), -1)
                    
                    border[pt[0],pt[1],0] = 0
                    border[pt[0],pt[1],1] = 255
                    border[pt[0],pt[1],2] = 0 
                
                
                for finger in fingers:
                    cv2.circle(border,(finger[1],finger[0]), 3, (0,0,255), -1)
                    
                    
                cv2.imshow('border', border)
                cv2.waitKey(1)
                            
                
                
            #cv2.drawContours(frame,contours,-1,(0,255,0),0)

            cv2.imshow('frame', frame)
            cv2.imshow('border', border)
            cv2.waitKey(1)
            
        
    