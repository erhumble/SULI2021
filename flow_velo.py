import matplotlib.lines as mlines
import postgkyl as pg
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pylab as pyplt
import os.path
import math

#    List the names of the scans you would like to analyze. RULES:
#         - Names must follow the form XXXaPerp_XLz_[diagnostic]_[timeframe].bp.
#         - Names must be uniform across files.
#         - Names must be same as folder they are contained in.
#    For example, the path to get to the first electron MO file for a scan
#    with aPerp = 1.5*a0 and aPar = 5*Lz would look like:
#     ~/path_to_folder/150aPerp_5Lz_electron_M0_0.bp .

filename_list =['150aPerp_1fourthLz_cs3','200aPerp_1fourthLz_cs3','250aPerp_1fourthLz_cs3','150aPerp_1Lz_cs3','200aPerp_1Lz_cs3','250aPerp_1Lz_cs3']

#    input parameters - some must be pulled from the .lua input files for
#    the simulations you are doing.

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
c_s3 = math.sqrt((Te0+3*Ti0)/mi) 
c_so = math.sqrt(Te0 / mi)
c_s = math.sqrt(80*elec_charge/mi)#43617.079640066
L_c = 30
R = 2.3
omega_ci = 70307921.342553
rho_s = 0.00062037219714627
rho_so = c_so / omega_ci
rowCutSize = 5

all_densityflux = []
all_normalized = []




for filename in filename_list:

    if(filename.find("nFac") != -1):
                    open_path =  'final_scans/neutrals/'  + filename + '/'
    else:
                    open_path =  'final_scans/baseline/'  + filename + '/'
    save_path = 'postprocess/' + filename + '_postprocess/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    normalized_flow_list = []
    flux_density_list = []
    
    ########### opening first frame to get basic coordinates. ###############
    data_ionUpar = pg.Data(open_path + '%s_ion_Upar_%d.bp' %(filename, timeframe_start))
    dg_ionUpar = pg.GInterpModal(data_ionUpar)
    grid_ionUpar, values_ionUpar = dg_ionUpar.interpolate()
    z_midplane = int(len((grid_ionUpar[2][1:] + grid_ionUpar[2][:-1])/2)/2) # slicing at midplane
    z_minimum = int(grid_ionUpar[2][0])
    ionUparGrid_midplane = values_ionUpar[:, :, z_midplane, 0]
    ionUparGrid_min = values_ionUpar[:, :, z_minimum, 0]
    ionUparGrid_min_trimmed = ionUparGrid_min[rowCutSize:(len(ionUparGrid_min)-rowCutSize-1), rowCutSize:(len(ionUparGrid_min[0])-rowCutSize-1)]
    ionUparGrid_midplane_trimmed = ionUparGrid_midplane[rowCutSize:(len(ionUparGrid_midplane)-rowCutSize-1), rowCutSize:(len(ionUparGrid_midplane[0])-rowCutSize-1)]
    
    data_ionDensity = pg.Data(open_path + '%s_ion_M0_%d.bp' %(filename, timeframe_start))
    dg_ionDensity = pg.GInterpModal(data_ionDensity)
    grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
    z_slice = int(len((grid_ionDensity[2][1:] + grid_ionDensity[2][:-1])/2)/2)  # slicing at midplane
    ionDensityGrid = values_ionDensity[:, :, z_slice, 0]
    ionDensityGrid_trimmed = ionDensityGrid[rowCutSize:(len(ionDensityGrid)-rowCutSize-1), rowCutSize:(len(ionDensityGrid[0])-rowCutSize-1)]
    
    #   shift coordinates to be cell-centered.
    CCC = []
    for j in range(0,len(grid_ionDensity)):
        CCC.append((grid_ionDensity[j][1:] + grid_ionDensity[j][:-1])/2)
    x_vals = CCC[0] # all x vals
    y_vals = CCC[1] # all y vals
    z_vals = CCC[2] # all z vals
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1]-y_vals[0]
    dz = z_vals[1]-z_vals[0]
    X, Y = np.meshgrid(x_vals, y_vals) # create mesh grids of y, x
    X = np.transpose(X)
    Y = np.transpose(Y)
    X_trimmed = X[rowCutSize:(len(X)-rowCutSize-1), rowCutSize:(len(X[0])-rowCutSize-1)] # trim to not include edges when searching for max/mins later.  
    Y_trimmed = Y[rowCutSize:(len(Y)-rowCutSize-1), rowCutSize:(len(Y[0])-rowCutSize-1)]

    for tf in range(timeframe_start, timeframe_end):
        
       data_ionUpar = pg.Data(open_path + '%s_ion_Upar_%d.bp' %(filename, tf))
       dg_ionUpar = pg.GInterpModal(data_ionUpar)
       grid_ionUpar, values_ionUpar = dg_ionUpar.interpolate()
       z_midplane = int(len((grid_ionUpar[2][1:] + grid_ionUpar[2][:-1])/2)/2) # slicing at midplane
       z_minimum = int(grid_ionUpar[2][0])
       ionUparGrid_midplane = values_ionUpar[:, :, z_midplane, 0]
       ionUparGrid_min = values_ionUpar[:, :, z_minimum, 0]
       ionUparGrid_min_trimmed = ionUparGrid_min[rowCutSize:(len(ionUparGrid_min)-rowCutSize-1), rowCutSize:(len(ionUparGrid_min[0])-rowCutSize-1)]
       ionUparGrid_midplane_trimmed = ionUparGrid_midplane[rowCutSize:(len(ionUparGrid_midplane)-rowCutSize-1), rowCutSize:(len(ionUparGrid_midplane[0])-rowCutSize-1)]

       data_ionDensity = pg.Data(open_path + '%s_ion_M0_%d.bp' %(filename, tf))
       dg_ionDensity = pg.GInterpModal(data_ionDensity)
       grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
       z_slice = int(len((grid_ionDensity[2][1:] + grid_ionDensity[2][:-1])/2)/2)  # slicing at midplane
       ionDensityGrid_midplane = values_ionDensity[:, :, z_midplane, 0]
       ionDensityGrid_min = values_ionDensity[:, :, z_minimum, 0]
       ionDensityGrid_min_trimmed = ionDensityGrid_min[rowCutSize:(len(ionDensityGrid_min)-rowCutSize-1), rowCutSize:(len(ionDensityGrid_min[0])-rowCutSize-1)]
       ionDensityGrid_midplane_trimmed = ionDensityGrid_midplane[rowCutSize:(len(ionDensityGrid_midplane)-rowCutSize-1), rowCutSize:(len(ionDensityGrid_midplane[0])-rowCutSize-1)]

       frameMax_density = ionDensityGrid_midplane_trimmed.max()
       maxLocation_density = np.where(ionDensityGrid_midplane_trimmed == frameMax_density) #location in TRIMMED frame.
       
       sigma_density = ionDensityGrid_midplane_trimmed[maxLocation_density[0][0], maxLocation_density[1][0] ] / ionDensityGrid_min_trimmed[maxLocation_density[0][0], maxLocation_density[1][0] ]
       flow = ionUparGrid_min_trimmed[maxLocation_density[0][0], maxLocation_density[1][0] ]/c_so
       normalized_flow_list.append(flow)
       flux_density = sigma_density*flow
       flux_density_list.append(flux_density)
    all_densityflux.append(flux_density_list)
    all_normalized.append(normalized_flow_list)

colors=cm.winter(np.linspace(0,1,len(all_normalized)))

point = 75
plt.figure()
for i in range(0, len(all_densityflux)):
    if(filename_list[i].find('nFac') == -1):
       plt.plot(np.arange(timeframe_start, timeframe_end)*timestep, all_densityflux[i], color=colors[i])
       plt.text(point*timestep,all_densityflux[i][point],filename_list[i],color=colors[i])
    else:
        colors=cm.plasma(np.linspace(0,1,len(all_normalized)))
        plt.plot(np.arange(timeframe_start, timeframe_end)*timestep, all_densityflux[i], color=colors[i])
        plt.text(point*timestep,all_densityflux[i][point],filename_list[i],color=colors[i])


plt.legend(filename_list)
plt.title("normalized density flux")
plt.xlabel("Time")
plt.ylabel("c_so")
plt.savefig("postprocess/densityflux_together.png")
plt.show()

