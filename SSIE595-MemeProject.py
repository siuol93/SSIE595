"""
@Code run using: Python Version 3.7
###TEST CODE WITH BEST PARAMETERS
@Description: SSIE 595 Memes Project
@author: Louis Lo
@B-Number: B00244341
"""
#Libraries and Imports
from mpl_toolkits import mplot3d
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv

#initialize function for memes data 
def initialize():
    global x,y,z,resultx,resulty,resultz,t,timesteps,tempvar
    x = 0.1 #0.1
    y = 0.2 #0.2
    z = 0.3 #0.3
    
    resultx = [x]
    resulty = [y]
    resultz = [z]
    
    t = 0.
    timesteps = [t]
    
    #used for multiple redentions of different variable numbers
    tempvar=0.
    
#observe function to append data to lists    
def observe():
    global x,y,z, resultx,resulty,resultz, t, timesteps
    
    resultx.append(x)
    resulty.append(y)
    resultz.append(z)
    timesteps.append(t)
    
    return resultx,resulty,resultz
    
#update function to depict the differential equations
def update(a,b,c,d,e,f):
    global x, y, z, t,timesteps
    
    x = x + x*(c + d*z + e*y - f*y)*Dt
    y = y + y*(b*z - e*x)*Dt
    z = z + z*(a - b*y - d*x)*Dt
    
    t = t + Dt

#csv information from actual
data = pd.read_csv(r"D:\Users\louis\Binghamton U\Final Project\multiTimeline - cat_bbl_psrptr CLEAN.csv")

#Original Data columns
catdata = pd.DataFrame(data, columns= ['cat memes'])
bbldata = pd.DataFrame(data, columns= ['bad luck brian'])
phdata = pd.DataFrame(data, columns= ['philosoraptor'])

#cast data to numbers in numpy array
xdata = catdata.to_numpy()
ydata = bbldata.to_numpy()
zdata = phdata.to_numpy()

#Save displays of graphs
filenamepre="memerun"
imgnameend=".png"

#Variables with best euclidean distances per -FINAL run.
vara = [1.2,1.3,1.3,1.3,1.3,1.2,0.99]
varb = [0.8,0.4,0.4,0.9,0.4,0.7,0.9]
varc = [0.001,0.000001,0.00001,0.001,0.0001,0.001,0.001]
vard = [0.17,0.08,0.08,0.2,0.08,0.14,0.2]
vare = [0.1,0.05,0.05,0.1,0.05,0.1,0.15]


#Variables for 5.998 and 7.410 only
var_aa = [1.2, 1.2]
var_bb = [0.8, 0.7]
var_cc = [0.001, 0.001]
var_dd = [0.17, 0.14]
var_ee = [0.1, 0.1]
var_ff = [0.00007, 0.02]

Dt = 0.01
#run 7 times, for each good parameter. 
#for i in range(0,7):

#Run only for var_first and var_second Euclidean dist: 5.998 and 7.410
def memerun(aa,bb,cc,dd,ee,ff):
    
    total_eucdissqrt = 0.0
    
    
    #parameters to try
    a = aa#vara[i]#1.2 #0.0-2.0
    b = bb#varb[i]#0.5 #0.0-1.0
    c = cc#varc[i]#0.0001 #0.0-1.0
    d = dd#vard[i]#0.11 #0.0-1.0
    e = ee#vare[i]#0.09 #0.0-1.0
    f = ff#varf[i]
    
    
    #call initialize to start
    initialize()
    #Euclidean Distance, a,b,c,d,e
    #5.998, 1.2, 0.8, 0.001, 0.17, 0.1
    #6.612, 1.3, 0.4, 0.000001, 0.08, 0.05
    #6.645, 1.3, 0.4, 0.00001, 0.08, 0.05
    #6.848: 1.3, 0.9, 0.001, 0.2, 0.1
    #7.045: 1.3, 0.4, 0.0001, 0.08, 0.05
    #7.410, 1.2, 0.7, 0.001, 0.14, 0.1
    #7.419, 0.99, 0.9, 0.001, 0.2, 0.15
    
    while t < 190.:
        update(a,b,c,d,e,f)
        resultall = observe()
        
    #Plot 2D graph of meme data with parameters specified    
    plt.plot(timesteps,resultx,'k:',label='x')
    plt.plot(timesteps,resulty,'b--',label='y')
    plt.plot(timesteps,resultz,'r.',label='z')
    plt.suptitle('Estimated slopes for X,Y,Z')
    plt.legend()
    
    plt.close()
    
    #convert list to float64 for scatter3D plot
    rx = np.array(resultx, dtype =np.float64)
    ry = np.array(resulty, dtype =np.float64)
    rz = np.array(resultz, dtype =np.float64)
    
    #Find the maxes of each array and then divide to find from original data to be able to scale
    scalerx = np.amax(xdata)/np.amax(rx)
    scalery = np.amax(ydata)/np.amax(ry)
    scalerz = np.amax(zdata)/np.amax(rz)
    
    #Convert data range to same as actuals
    transrx = np.multiply(rx,scalerx) #for ex. max from resultx is 11.93, need to multiply by 5.28 to match actual (max of orig is 63)
    transry = np.multiply(ry,scalery) #for ex. max from resulty is 9.08, need to multiply by 11.01 to match actual (max of orig is 100)
    transrz = np.multiply(rz,scalerz) #for ex. max from resultz is 4.04, need to multiply by 6.93 to match actual (max of orig is 28)
    
    
    transrowsx = [525,565,700,843,891,995,1080,8180,8738,8880]
    transrowsy = [0,444,758,1110,1167,1211,1226,1265,1386,1426]
    transrowsz = [270,325,349,372,385,8700,9048,9305,9390,9405]
    ##Original data points, max is 190.
    normrows = [97,101,108,117,124,131,136,147,183,189]
    
    
    #for each datapoint(10), calculate the euclidean distance
    for w in range(0,10):
        
    #euclidean distance
    #points 98, 102, 109, 116, 125, 132, 137, 148, 184, 190
        eucdis = abs((transrx[transrowsx[w]]-xdata[normrows[w]]))**2 + abs((transry[transrowsy[w]]-ydata[normrows[w]]))**2 + abs((transrz[transrowsz[w]]-zdata[normrows[w]]))**2
        eucdissqrt = sqrt(eucdis)
       
        #sum the euclidean distances and output an average
        total_eucdissqrt = total_eucdissqrt + eucdissqrt
        
    #calculate the average euclidean distance and save in excel    
    total_eucdissqrt = total_eucdissqrt/10.0
    
    print(total_eucdissqrt, a,b,c,d,e,f)
    
    return transrx, transry, transrz, total_eucdissqrt

    #filenamemid=str(total_eucdissqrt)

    #imgname = filenamepre + filenamemid + imgnameend
    
data1 = memerun(var_aa[0],var_bb[0],var_cc[0],var_dd[0],var_ee[0],var_ff[0])
data2 = memerun(var_aa[1],var_bb[1],var_cc[1],var_dd[1],var_ee[1],var_ff[1])

#setup plot labels 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('cat meme (x)')
ax.set_ylabel('bbl (y)')
ax.set_zlabel('phil (z)')
fig.suptitle('3D Plot Estimate Vs Actuals')

#Plot 3D scatter points
ax.scatter3D(data1[0],data1[1],data1[2], c='r', label='ED = '+ str(data1[3]))
ax.scatter3D(data2[0],data2[1],data2[2], c='g', label='ED = '+ str(data2[3]))
ax.scatter3D(xdata, ydata, zdata, c='b', label='Actual')
ax.legend()
plt.show()
    
    
    #fig.savefig(imgname,dpi = fig.dpi)
    ##Close figures so it does not clog up
    #plt.close()