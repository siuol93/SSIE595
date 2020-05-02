"""
@Code run using: Python Version 3.7
###textbk Euler Forward Path pg 124 - 126
@Description: SSIE 595 Memes Project
@author: Louis Lo
@B-Number: B00244341
@Related files: multiTimeline - cat_bbl_psrptr CLEAN.csv
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
    global x,y,z,resultx,resulty,resultz,t,timesteps
    x = 0.1 #0.1
    y = 0.2 #0.2
    z = 0.3 #0.3
    
    resultx = [x]
    resulty = [y]
    resultz = [z]
    
    t = 0.
    timesteps = [t]
    
#observe function to append data to lists    
def observe():
    global x,y,z, resultx,resulty,resultz, t, timesteps
    
    resultx.append(x)
    resulty.append(y)
    resultz.append(z)
    timesteps.append(t)
    
    return resultx,resulty,resultz
    
#update function to depict the differential equations
def update(a,b,c,d,e):
    global x, y, z, t,timesteps
    
    x = x + x*(c + d*z + e*y)*Dt
    y = y + y*(b*z - e*x)*Dt
    z = z + z*(a - b*y - d*x)*Dt
    
    t = t + Dt


#Capture data by creating .csv filenames and image names.
filenamepre="memerun"
filenameend=".csv"
imgnameend=".png"
#used for multiple redentions of different variable numbers, initialized to 0.
tempvar = 0.0
#tempvar = 0.000001
filenamemid = str(0.0)
#filename for writing to .csv
filename = filenamepre + filenamemid + filenameend
#csv information from actual
data = pd.read_csv(r"D:\Users\louis\Binghamton U\Final Project\multiTimeline - cat_bbl_psrptr CLEAN.csv")

#array of testing variable c parameters
variablec_num = [0.000001,0.00001,0.0001,0.001,0.01,0.1]

#Original Data columns
catdata = pd.DataFrame(data, columns= ['cat memes'])
bbldata = pd.DataFrame(data, columns= ['bad luck brian'])
phdata = pd.DataFrame(data, columns= ['philosoraptor'])

#cast data to numbers in numpy array
xdata = catdata.to_numpy()
ydata = bbldata.to_numpy()
zdata = phdata.to_numpy()

#field names
fields = ['Variable a', 'Variable b' , 'Variable c', 'Variable d', 'Variable e', 'Timepoints', 'Estimated x','Actual x', 'Estimated y','Actual y', 'Estimated z','Actual z', 'Scale Value X', 'Scale Value Y', 'Scale Value Z', 'Euclidean Distance b/w Estimate and Normal and Squared', 'Euclidean Distance SqRt']

#write data to csv file
with open(filename,'w') as csvfile:
    #create a csv write object
    csvwriter = csv.writer(csvfile)
    
    #writing the fields
    csvwriter.writerow(fields)

#initialize euclidean distance dictionary.
top_eucdiss = {}

#initialize array to store variable values 
variable_pars = []

#intialize top 5 counter
count = 0

#for loops to traverse through a,b,c,d,e

for vara in np.arange(0.1,2.0,0.1):    
    for varb in np.arange(0.2,1.0,0.1):
        for varc in variablec_num:
            for vard in np.arange(0.05,0.5,0.05):
                for vare in np.arange(0.1,1.0,0.1):
                #testing purpose only
                #for i in np.arange(0.2, 1.2,0.2):
    
                    #call initialize to start
                    initialize()
                    scalerx,scalery,scalerz = 0.0,0.0,0.0
                    transrx,transry,transrz = 0.0,0.0,0.0
                    total_eucdissqrt = 0.0
                    tempvar+=0.2
                    
                    filenamemid = str(tempvar)
                    
                    imgname = filenamepre + filenamemid + imgnameend
                    
                
                    #create datapoints that will allow us to measure euclidean distance
                    #Since Dt is 0.01, our estimated data will have up to 19002 points.
                    transrows = [9800,10200,10900,11600,11800,11900,12500,13200,13700,14200,17900,19000]
                    #Original data points, max is 190.
                    normrows = [97,101,108,115,117,118,124,131,136,141,178,189]
                    rowpoints = ['Point 98', 'Point 102', 'Point 109', 'Point 116', 'Point 118', 'Point 119', 'Point 125', 'Point 132', 'Point 137', 'Point 142', 'Point 179', 'Point 190']
                    
                    #parameters to try
                    a = vara#1.2 #0.2-2.0 1.2
                    b = varb#0.6 #0.2-1.0 0.5
                    c = varc#0.00001 #0.000001-1.0 0.00001(multiples of 10)
                    d = vard#0.11 #0.05-0.5  0.11 increments of 0.05
                    e = vare#0.15 #0.00-1.0 0.09 increments of .10
                    Dt = 0.01
                    
                    while t < 190.:
                        update(a,b,c,d,e)
                        resultall = observe()
                
                    
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
                    
                    #open excel file and write data to it
                    with open(filename,'a') as csvfile:
                        #create a csv write object
                        csvwriter = csv.writer(csvfile)
                    
                        #for each datapoint(12), calculate the euclidean distance
                        for w in range(0,12):
                            
                        #euclidean distance
                        #points 98, 102, 109, 116, 118, 119, 125, 132, 137, 142, 179, 190
                            eucdis = (transrx[transrows[w]]-xdata[normrows[w]])**2 + (transry[transrows[w]]-ydata[normrows[w]])**2 + (transrz[transrows[w]]-zdata[normrows[w]])**2
                            eucdissqrt = sqrt(eucdis)
                            #Write data rows
                            csvwriter.writerow([a,b,c,d,e,rowpoints[w],transrx[transrows[w]],float(xdata[normrows[w]]),transry[transrows[w]],float(ydata[normrows[w]]),transrz[transrows[w]],float(zdata[normrows[w]]),scalerx,scalery,scalerz,float(eucdis), float(eucdissqrt)])
                            
                            #sum the euclidean distances and output an average
                            total_eucdissqrt = total_eucdissqrt + eucdissqrt
                            
                        #calculate the average euclidean distance and save in excel    
                        total_eucdissqrt = total_eucdissqrt/12.0
                        csvwriter.writerow(['','','','','','','','','','','','','','Average euclidean distance is',total_eucdissqrt])
                        
                    #Create array to store alternating variables with top 5 lowest euclidean distances saved.
                    variable_pars =[a,b,c,d,e]
                    
                    #Save top 5 lowest euclidean distances
                    if(count < 5):
                        top_eucdiss[float(total_eucdissqrt)] = variable_pars
                        count+=1
                    else:
                        if(total_eucdissqrt < max(top_eucdiss)):
                            top_eucdiss.pop(max(top_eucdiss))
                            top_eucdiss[float(total_eucdissqrt)] = variable_pars
                                
                    #reset average euclidean distance    
                    total_eucdissqrt = 0.0
    
##Plot 2D graph of meme data with parameters specified    
#plt.plot(timesteps,resultx,'k--',label='x')
#plt.plot(timesteps,resulty,'b--',label='y')
#plt.plot(timesteps,resultz,'r--',label='z')
#plt.legend()
#
#plt.plot(timesteps,transrx,'g-',label='rx')
#plt.plot(timesteps,transry,'c-',label='ry')
#plt.plot(timesteps,transrz,'m-',label='rz')
#plt.legend()
#
    #setup plot labels 
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.set_xlabel('cat meme (x)')
#    ax.set_ylabel('bbl (y)')
#    ax.set_zlabel('phil (z)')
    
    
    #Plot 3D scatter points
    ##ax.scatter3D(resultx,resulty,resultz, cmap='Greens');
    #ax.scatter3D(transrx,transry,transrz, cmap='Greens');
    #plt.show()
    
    #filename for saving graph
#    fig.savefig(imgname,dpi = fig.dpi)
#    #Close figures so it does not clog up
#    plt.close()
