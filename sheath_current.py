import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pylab as pyplt
import os.path
import math

def PolyArea(x,y):
                return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
n0 = 2.5e19  #1/m^3
filename_list = ['050aPerp_1Lz_cs3_NEUTRAL_CX_nFac01', '050aPerp_1Lz_cs3_NEUTRAL_CX_nFac05', '050aPerp_1Lz_cs3_NEUTRAL_CX_nFac10', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac01', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac05', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac10', '025aPerp_1fourthLz_cs3', '050aPerp_1fourthLz_cs3', '100aPerp_1fourthLz_cs3','150aPerp_1fourthLz_cs3','200aPerp_1fourthLz_cs3','250aPerp_1fourthLz_cs3','025aPerp_1Lz_cs3', '050aPerp_1Lz_cs3','100aPerp_1Lz_cs3','150aPerp_1Lz_cs3','200aPerp_1Lz_cs3','250aPerp_1Lz_cs3']
timeframe_start = 0
elec_mass = 9.1093837015e-31 #kg
elec_charge = 1.60217662e-19 #coulombs
rho_s = 0.00062037219714627
sigma_v = 1
timestep = (10e-6)/100
mi = 3.368659976918e-27
Te0 = 6.408705948e-18
Ti0 = 6.408705948e-18
c_so = math.sqrt(Te0 / mi)
c_s3 = math.sqrt((Te0+3*Ti0)/mi)
c_s = math.sqrt(80*elec_charge/mi)#43617.079640066
L_c = 30
R = 2.3
omega_ci = 70307921.342553
rho_so = c_so / omega_ci
timestep = (10e-6)/100
lam = np.log(math.sqrt(mi/(2*math.pi*elec_mass)))
rowCutSize = 5
t = np.array(range(timeframe_start, 100))*timestep
 ##### Update parameters for graphing the blobs themselves. ###########
plt.rcParams.update({
                                 "text.usetex": True,
                                 "font.family": "serif",
                                 "font.serif": ["Palatino"],
                                 "font.size":12,
                                 "image.cmap": 'inferno', # Don't use rainbow cmaps
})
     
tf = 10

for filename in filename_list:
    if(filename.find("nFac")!=-1):
        open_path = 'final_scans/neutrals/' + filename + '/'
    else:
        open_path = 'final_scans/baseline/' + filename + '/'
    save_path = 'postprocess/' + filename + '_postprocess/'
    jpar_1_list = []
    jpar_2_list = []
    
    for tf in range(0,100):
        data_phi = pg.Data(open_path + '%s_phi_%d.bp' %(filename, tf))
        dg_phi = pg.GInterpModal(data_phi)
        grid_phi, values_phi = dg_phi.interpolate()
        z_slice = int(len((grid_phi[2][1:] + grid_phi[2][:-1])/2)/2)  # slicing at midplane
        phiGrid = values_phi[:, :, z_slice, 0]
        phiGrid_trimmed = phiGrid[rowCutSize:(len(phiGrid)-rowCutSize-1), rowCutSize:(len(phiGrid[0])-rowCutSize-1)]



        #   shift coordinates to be cell-centered.
        CCC = []
        for j in range(0,len(grid_phi)):
                CCC.append((grid_phi[j][1:] + grid_phi[j][:-1])/2)
        x_vals = CCC[0] # all x vals
        y_vals = CCC[1] # all y vals
        z_vals = CCC[2] # all z vals
        X, Y = np.meshgrid(x_vals, y_vals) # create mesh grids of y, x
        X = np.transpose(X)
        Y = np.transpose(Y)
        X_trimmed = X[rowCutSize:(len(X)-rowCutSize-1), rowCutSize:(len(X[0])-rowCutSize-1)]
        Y_trimmed = Y[rowCutSize:(len(Y)-rowCutSize-1), rowCutSize:(len(Y[0])-rowCutSize-1)]

        data_ionDensity = pg.Data(open_path  + '%s_electron_M0_%d.bp' %(filename, tf))
        dg_ionDensity = pg.GInterpModal(data_ionDensity)
        grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
        ionDensityGrid = values_ionDensity[:, :, z_slice, 0]
        ionDensityGrid_trimmed = ionDensityGrid[rowCutSize:(len(ionDensityGrid)-rowCutSize-1), rowCutSize:(len(ionDensityGrid[0])-rowCutSize-1)]

        data_elecTemp = pg.Data(open_path  + '%s_electron_Temp_%d.bp' %(filename, tf))
        dg_elecTemp = pg.GInterpModal(data_elecTemp)
        grid_elecTemp, values_elecTemp = dg_elecTemp.interpolate()
        elecTempGrid = values_elecTemp[:, :, z_slice, 0]
        elecTempGrid_trimmed = elecTempGrid[rowCutSize:(len(elecTempGrid)-rowCutSize-1), rowCutSize:(len(elecTempGrid[0])-rowCutSize-1)]

        data_ionTemp = pg.Data(open_path  + '%s_ion_Temp_%d.bp' %(filename, tf))
        dg_ionTemp = pg.GInterpModal(data_ionTemp)
        grid_ionTemp, values_ionTemp = dg_ionTemp.interpolate()
        ionTempGrid = values_ionTemp[:, :, z_slice, 0]
        ionTempGrid_trimmed = ionTempGrid[rowCutSize:(len(ionTempGrid)-rowCutSize-1), rowCutSize:(len(ionTempGrid[0])-rowCutSize-1)]

        z_slice_midplane = int(len((grid_ionDensity[2][1:] + grid_ionDensity[2][:-1])/2)/2)  # slicing at midplane
        z_slice_end = len(grid_ionDensity[2])-1

        n_mid = values_ionDensity[:, :, z_slice_midplane]
        n_sh = values_ionDensity[:, :, 0]
   
        p = ionDensityGrid_trimmed*(np.array(ionTempGrid_trimmed)  + np.array(elecTempGrid_trimmed))
        p0 = n0 * (Te0 + Ti0)
        pMax = np.where(p == p.max())

        sigma_v = (n_sh[pMax[0][0], pMax[1][0]] / n_mid[pMax[0][0], pMax[1][0]])[0]

        J_par = ((2*n0*sigma_v*(elec_charge**2)*c_s3) / (L_c*Te0))*(phiGrid_trimmed[pMax[0][0], pMax[1][0]]-(lam*Te0/elec_charge))
        jpar_1_list.append(J_par)
        data_M1_elec = pg.Data(open_path + '%s_electron_M1_%d.bp' %(filename, tf))
        dg_M1_elec = pg.GInterpModal(data_M1_elec)
        grid_M1_elec, values_M1_elec = dg_M1_elec.interpolate()
        z_slice = int(np.array(grid_M1_elec[2]).min())  # slicing at midplane
        M1_elecGrid = values_M1_elec[:, :, z_slice, 0]
        M1_elecGrid_trimmed = M1_elecGrid[rowCutSize:(len(M1_elecGrid)-rowCutSize-1), rowCutSize:(len(M1_elecGrid[0])-rowCutSize-1)]

        data_M1_ion = pg.Data(open_path + '%s_ion_M1_%d.bp' %(filename, tf))
        dg_M1_ion = pg.GInterpModal(data_M1_ion)
        grid_M1_ion, values_M1_ion = dg_M1_ion.interpolate()
        z_slice = int(np.array(grid_M1_ion[2]).min())  # slicing at midplane
        M1_ionGrid = values_M1_ion[:, :, z_slice, 0]
        M1_ionGrid_trimmed = M1_ionGrid[rowCutSize:(len(M1_ionGrid)-rowCutSize-1), rowCutSize:(len(M1_ionGrid[0])-rowCutSize-1)]


        J_par_2 = 2*(1/30)*elec_charge*(M1_ionGrid_trimmed[pMax[0][0], pMax[1][0]] - M1_elecGrid_trimmed[pMax[0][0], pMax[1][0]])
        jpar_2_list.append(J_par_2)
        
        if(tf == 10):
            print()
            print('For ' + str(filename) + ', first equation yields: ' + str(J_par) + ' and the second yields ' + str(J_par_2))

    plt.figure()
    plt.plot(t, jpar_1_list, color = 'maroon')
    plt.title(filename.replace('_', ' ') + " Jparallel, eq1")
    plt.xlabel("Time (s)")
    plt.ylabel("Parallel current (A)")
    plt.savefig(save_path + filename + "_jparallel_eq1.png")

    plt.figure()
    plt.plot(t, jpar_2_list, color = 'teal')
    plt.title(filename.replace('_', ' ') + " Jparallel, eq2")
    plt.xlabel("Time (s)")
    plt.ylabel("Parallel current (A)")
    plt.savefig(save_path + filename + "_jparallel_eq2.png")
