import postgkyl as pg

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pylab as pyplt
import os.path
import math
import pandas
import matplotlib.lines as mlines

cx_frequency = 57399.73307055999
timeframe_start = 0
timeframe_end = 100
timestep = (10e-6)/100
Te0 = 6.408705948e-18
Ti0 = 6.408705948e-18
a0 = 0.011831417882554278
elec_mass = 9.1093837015e-31 #kg
elec_charge = 1.60217662e-19 #coulombs
n0 = 2.5e19  #1/m^3
mi = 3.368659976918e-27
c_so = math.sqrt(Te0 / mi)
c_s = math.sqrt(80*elec_charge/mi)#43617.079640066
L_c = 30
R = 2.3
c_s3 = math.sqrt((Te0+3*Ti0)/mi)
omega_ci = 70307921.342553
rho_s = 0.00062037219714627
rho_so = c_so / omega_ci

vb_comparisonlist = pandas.read_csv("postprocess/vb_comparisonlist_CMbase.csv", header=None).to_numpy()#.as_matrix()
compactness = pandas.read_csv("postprocess/compactness_averaged.csv", header=None).to_numpy()#.as_matrix()
colors=['turquoise','springgreen','palegreen']
steps = np.array(np.arange(0,3,0.01))

delP = np.average(vb_comparisonlist[:,3])

fig = plt.figure()
ax = fig.add_subplot(111)

handles_list = []
alpha_list=[1,0.5,0.1]
typename='thermal ion'
v_b_upper = 2*((steps*a0/R)**0.5)*c_so*(delP) ## assumption: delta_p/p = 0.9
for i in range(0, len(alpha_list)):
    v_b_lower = 1 + (  ((2*math.sqrt(R)) /(L_c*(rho_so**2)))*((steps*a0)**2.5)*alpha_list[i] )
    handles_list.append(plt.plot(steps, v_b_upper/v_b_lower,label=('Sigma = ' + str(alpha_list[i])), linestyle='dotted', color = colors[i])[0])

for k in range(0, len(vb_comparisonlist)):
    if(vb_comparisonlist[k][2].find("nFac") == -1):
        if(vb_comparisonlist[k][2].find("1fourthLz") != -1):
            line2, = plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'red', marker ='^')
        if(vb_comparisonlist[k][2].find("1Lz") != -1):        
            plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'blue', marker ='D')



# Create a legend for the first line.
first_legend = plt.legend(handles=handles_list, loc=1)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)


mark1 = mlines.Line2D([], [], color='red', marker='^', linestyle='None', markersize=10, label='aPar = 0.25*Lz')
mark2 = mlines.Line2D([], [], color='blue', marker='D', linestyle='None', markersize=10, label='aPar = Lz')
plt.legend(handles=[mark1,mark2], loc=2)

            
plt.title(" Vb vs. blob size - " + typename)
plt.xlabel("aPerp / a0")
plt.ylabel("Vb (m/s)")
plt.xticks(np.arange(0,3,0.5))
plt.yticks(np.arange(0, 5000, 1000))
plt.savefig("postprocess/sizeVvb_" + typename + ".png")
plt.show()





fig = plt.figure()
ax = fig.add_subplot(111)
alpha = 0.5
handles_list = []
colors=['teal','springgreen','crimson','teal','springgreen','crimson']
freq=[57870, 289350, 578700]
typename='neutrals eq 30'
v_b_upper = 2*((steps*a0/R)**0.5)*c_so*(delP)
for i in range(0, len(freq)):
    print('freq is: ' + str(freq[i]))
    v_b_lower = 1 + (  ((2*math.sqrt(R)) /(L_c*(rho_so**2)))*((steps*a0)**2.5)*0.5 ) + (((steps*R)**0.5)*freq[i])/(2*c_so)
    plt.plot(steps, v_b_upper/v_b_lower,label=('CX Frequency = ' + str(freq[i])), linestyle='dotted', color = colors[i])
    #handles_list.append(plt.plot(steps, v_b_upper/v_b_lower,label=('CX Frequency = ' + str(freq[i])), linestyle='dotted', color = colors[i])[0])

for k in range(0, len(vb_comparisonlist)):    
    print('k is: ' + str(k) + 'item is: ' + vb_comparisonlist[k][2])
    if(vb_comparisonlist[k][2].find("nFac") != -1):
        print('hello')
        if(vb_comparisonlist[k][2].find("nFac01") != -1):
            print('hello 2')
            plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'teal', markersize=10, marker ='^')
        if(vb_comparisonlist[k][2].find("nFac05") != -1):        
            plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'springgreen', markersize=10, marker ='D')
        if(vb_comparisonlist[k][2].find("nFac10") != -1):        
            plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'crimson', markersize=10, marker ='o')
    else:
        if((vb_comparisonlist[k][2].find("050aPerp_1Lz") != -1) or (vb_comparisonlist[k][2].find("100aPerp_1Lz") != -1)):
            print("ENTERED LOOP ________________________") 
            plt.plot(vb_comparisonlist[k][0], vb_comparisonlist[k][1], color = 'steelblue', marker ='D')



# Create a legend for the first line.
#first_legend = plt.legend(handles=handles_list, loc=1)

# Add the legend manually to the current Axes.
#ax = plt.gca().add_artist(first_legend)

mark1 = mlines.Line2D([], [], color='teal', marker='^', linestyle='None', markersize=10, label='n_n=0.1*n0')
mark2 = mlines.Line2D([], [], color='springgreen', marker='D', linestyle='None', markersize=10, label='n_n=0.5*n0')
mark3 = mlines.Line2D([], [], color='crimson', marker='o', linestyle='None', markersize=10, label='n_n=n0')
mark4 = mlines.Line2D([], [], color='steelblue', marker='D', linestyle='None', markersize=10, label='aPar = 1Lz, no neutrals')
plt.legend(handles=[mark1,mark2,mark3,mark4], loc=2)

            
plt.title("Blob size vs. velocity, " + typename)
plt.xlabel("aPerp / a0")
plt.ylabel("Velocity (m/s)")
plt.xticks(np.arange(0,3,0.5))
plt.yticks(np.arange(0, 5000, 1000))
plt.savefig("postprocess/sizeVvb_" + typename + ".png")
plt.show()







handles_list = []
alpha_list=[1,0.5,0.1]
typename='compactness'

for k in range(0, len(compactness)):
    print('filename is: ' + compactness[k][3]) 
    if(compactness[k][3].find("1fourthLz") != -1):
        if(compactness[k][3].find("nFac") == -1):
                plt.plot(compactness[k][0], compactness[k][1], color = 'navy', marker ='.', markersize=10)
                plt.plot(compactness[k][0], compactness[k][2], color = 'dodgerblue', marker ='.', markersize=10)
    if(compactness[k][3].find("1Lz") != -1):
            print('inside last 4')
            if(compactness[k][3].find("nFac") == -1):
                plt.plot(compactness[k][0], compactness[k][1], color = 'firebrick', marker ='.', markersize=10)
                plt.plot(compactness[k][0], compactness[k][2], color = 'red', marker ='.', markersize=10)
            else:
                print('LOCATED 2')
                plt.plot(compactness[k][0], compactness[k][1], color = 'forestgreen', marker ='D', markersize=5)
                plt.plot(compactness[k][0], compactness[k][2], color = 'lime', marker ='D', markersize=5) 
    


# Create a legend for the first line.
first_legend = plt.legend(handles=handles_list, loc=1)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)


mark1 = mlines.Line2D([], [], color = 'navy', marker ='.', markersize=10, linestyle='None', label='aPar = 0.25*Lz, Thrysoe equation')
mark2 = mlines.Line2D([], [], color = 'dodgerblue', marker ='.', markersize=10, linestyle='None', label='aPar = 0.25*Lz circularity')

mark3 = mlines.Line2D([], [], color = 'firebrick', marker ='.', markersize=10, linestyle='None', label='aPar = 1*Lz, Thrysoe equation')
mark4 = mlines.Line2D([], [], color = 'red', marker ='.', markersize=10, linestyle='None', label='aPar = 1*Lz circularity')

mark7 = mlines.Line2D([], [], color = 'forestgreen', marker ='.', markersize=10, linestyle='None', label='aPar = 1*Lz with neutrals, Thrysoe equation')
mark8 = mlines.Line2D([], [], color = 'lime', marker ='.', markersize=10, linestyle='None', label='aPar = 1*Lz with neutrals, circularity')

plt.legend(handles=[mark1,mark2,mark3,mark4,mark7,mark8], loc=4, fontsize='x-small')

            
plt.title("Compactness, two methods")
plt.xlabel("aPerp / a0")
plt.ylabel("Compactness")
plt.xticks(np.arange(0,3,0.5))
plt.savefig(("postprocess/compactness_" + typename + ".png"), bbox_inches='tight', fontsize=3)
plt.show()

