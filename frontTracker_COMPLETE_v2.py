import matplotlib.lines as mlines
import postgkyl as pg
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

filename_list =['050aPerp_1Lz_cs3_NEUTRAL_CX_nFac01', '050aPerp_1Lz_cs3_NEUTRAL_CX_nFac05', '050aPerp_1Lz_cs3_NEUTRAL_CX_nFac10', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac01', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac05', '100aPerp_1Lz_cs3_NEUTRAL_CX_nFac10', '025aPerp_1fourthLz_cs3', '050aPerp_1fourthLz_cs3', '100aPerp_1fourthLz_cs3','150aPerp_1fourthLz_cs3','200aPerp_1fourthLz_cs3','250aPerp_1fourthLz_cs3','025aPerp_1Lz_cs3', '050aPerp_1Lz_cs3','100aPerp_1Lz_cs3','150aPerp_1Lz_cs3','200aPerp_1Lz_cs3','250aPerp_1Lz_cs3']

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


############ which method do you want to use to track velocity? ###############
TRACK_centerOfMass = False
TRACK_peakDensity = False
TRACK_peakGradient = False
TRACK_raytraceFront = False
TRACK_compactness = False
TRACK_theoryVelocity = False
TRACK_ab = False
TRACK_regime = False
TRACK_sizeVvb = False
TRACK_thermalEnergy = False
TRACK_kineticEnergy = False
save_and_plot_tracks = True
PLOT_ALL_TRACKERS = False
PLOT_ALL_GRAPHS = False
TRACK_density_over_time = True

def PolyArea(x,y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

size_list = []
vb_comparisonlist = []
initial_contour_area = 0
sigma = 0.5*13*rho_s
heaviside_zero = 0
n_delta_zero = 0
cutoff = 100
rowCutSize = 5
delta_p_list = []
compactness_megalist = []

#################### Begin iteration through each file specified above in filename_list.

for filename in filename_list:
    aPerp_int = float(float(filename[0:3])*0.01)
    if(filename.find("fourth") != -1):
        aPar_int = 0.25
    else:
        aPar_int = float(filename[9])
    if(filename.find("nFac") != -1):
        nn_int = float(filename[(len(filename)-2):(len(filename))])*0.1
    if(filename.find("nFac") != -1):
                    open_path =  'final_scans/neutrals/'  + filename + '/'
    else:
                    open_path =  'final_scans/baseline/'  + filename + '/'
    save_path = 'postprocess/' + filename + '_postprocess/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)


            
    ########### opening first frame to get basic coordinates. ###############
    data_ionDensity = pg.Data(open_path + '%s_electron_M0_%d.bp' %(filename, timeframe_start))
    dg_ionDensity = pg.GInterpModal(data_ionDensity)
    grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
    z_slice = int(len((grid_ionDensity[2][1:] + grid_ionDensity[2][:-1])/2)/2)  # slicing at midplane

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


    #################### front tracking algorithms implimented. ###############
    front_track_CM = []
    front_track_Density = []
    front_track_Gradient = []
    front_track_raytrace = []
    front_track_tCompactness_noComp = []
    front_track_tCompactness_Comp = []
    front_track_cCompactness  = []
    ab_list = []
    vb_list = []
    Etherm_list_ion = []
    Etherm_list_elec = []
    Ekin_list = []
    delP_running = []
    average_density = []

    ##### iterate through each frame of data to postprocess.
    for tf in range(timeframe_start, timeframe_end):
        data_ionDensity = pg.Data(open_path  + '%s_electron_M0_%d.bp' %(filename, tf))
        dg_ionDensity = pg.GInterpModal(data_ionDensity)
        grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
        ionDensityGrid = values_ionDensity[:, :, z_slice, 0]
        ionDensityGrid_3D  = values_ionDensity[:, :, :, 0]
        ionDensityGrid_trimmed = ionDensityGrid[rowCutSize:(len(ionDensityGrid)-rowCutSize-1), rowCutSize:(len(ionDensityGrid[0])-rowCutSize-1)]
        ionDensityGrid_3D_trimmed = ionDensityGrid_3D[rowCutSize:(len(ionDensityGrid)-rowCutSize-1), rowCutSize:(len(ionDensityGrid[0])-rowCutSize-1), :]
        if(TRACK_peakDensity or TRACK_compactness):
            # density tracker section.
            frameMax_density = ionDensityGrid_trimmed.max()
            maxLocation_density = np.where(ionDensityGrid_trimmed == frameMax_density) #location in TRIMMED frame.
            if(len(maxLocation_density)==2):
                front_track_Density.append([maxLocation_density[0][0], maxLocation_density[1][0]])
            else:
                print("error finding max, inserting placeholder.")
                front_track_Density.append(NULL)
        if(TRACK_density_over_time):
            blob_density_grid = np.where(ionDensityGrid < 0.5*n0, 0, ionDensityGrid)
            average_density.append(np.average(blob_density_grid))
        if(TRACK_peakGradient):
            # gradient tracker section.
            M0_gradient = np.gradient(ionDensityGrid, dx, axis=0) #assuming axis=0 is x...
            M0_gradient_trimmed = M0_gradient[rowCutSize:(len(M0_gradient)-rowCutSize-1), rowCutSize:(len(M0_gradient[0])-rowCutSize-1)]
            M0_gradient_trimmed = np.where(M0_gradient_trimmed > 0, 0, M0_gradient_trimmed)
            frameMax_gradient = np.asarray(np.absolute(M0_gradient_trimmed).max())
            maxLocation_gradient = np.where(np.absolute(M0_gradient_trimmed) == frameMax_gradient) #location in TRIMMED frame.
            if(len(maxLocation_gradient)==2):
                front_track_Gradient.append([maxLocation_gradient[0][0], maxLocation_gradient[1][0]])
            else:
                print("error finding max, inserting placeholder.")
                front_track_Gradient.append(NULL)
        if(TRACK_centerOfMass or TRACK_sizeVvb):
            #n_equation = sc.interpolate.RectBivariateSpline(x_vals[10:46], y_vals[10:46], ionDensityGrid)
            n_delta = ionDensityGrid_trimmed - (n0*1.2)
            n_delta = np.where(n_delta < 0, 0, n_delta)
            first_moment_x = np.sum(n_delta * X_trimmed)*dx
            zero_moment_x = np.sum(n_delta)*dx
            x_cm = first_moment_x/zero_moment_x
            dy = y_vals[1]-y_vals[0]
            first_moment_y = np.sum(n_delta*Y_trimmed)*dy
            zero_moment_y = np.sum(n_delta)*dy
            y_cm = first_moment_y/zero_moment_y
            maxLocation_cm = (x_cm, y_cm)
            if(len(maxLocation_cm)==2):
                front_track_CM.append(maxLocation_cm)
            else:
                print("error finding max, inserting placeholder.")
                front_track_CM.append(NULL)
        if(TRACK_raytraceFront):
            if(os.path.isfile(open_path + 'raytrace/file_number' + str(tf) + '_contour_number_0.txt')):
                raytrace_data = np.loadtxt(open_path  + 'raytrace/file_number' + str(tf) + '_contour_number_0.txt', dtype='float')
                raytrace_maxX = raytrace_data[:, 0].max()
                raytrace_max_location = np.where(raytrace_data[:, 0] == raytrace_maxX)
                front_track_raytrace.append(raytrace_data[raytrace_max_location][0])
            else:
                print('file not found. i is ' + str(tf))
                front_track_raytrace.append(front_track_raytrace[len(front_track_raytrace)-1])

        if(TRACK_ab or TRACK_regime or TRACK_theoryVelocity or TRACK_sizeVvb):
            if(os.path.isfile(open_path + 'raytrace/file_number' + str(tf) + '_contour_number_0.txt')): 
                contour = np.loadtxt(open_path +  'raytrace/file_number' + str(tf) + '_contour_number_0.txt')
                y_max_contour = contour[:,1].max()
                y_min_contour = contour[:,1].min()
                ab_list.append((y_max_contour - y_min_contour)*0.5)
            else:
                ab_list.append(ab_list[len(ab_list)-1])
        if(TRACK_regime or TRACK_theoryVelocity or TRACK_sizeVvb):
            if(filename.find("nFac") != -1):
                data_vsigma = pg.Data(open_path + '%s_ion_vSigmaCX_0.bp' %(filename))
                dg_vsigma = pg.GInterpModal(data_vsigma)
                grid_vsigma, values_vsigma = dg_vsigma.interpolate()
                z_slice = int(len((grid_vsigma[2][1:] + grid_vsigma[2][:-1])/2)/2)  # slicing at midplane
                vsigmaGrid = values_vsigma[:, :, z_slice, 0]
                n_n = 2.5e19*float(filename[len(filename)-2] + '.' + filename[len(filename)-1])
                vsigma_normalized = np.where(vsigmaGrid > 1.1*vsigmaGrid[10][10], True, False)
                vsigma_blob = vsigmaGrid[np.where(vsigma_normalized == True)[0], np.where(vsigma_normalized == True)[1]]
                cx_frequency = np.average(vsigma_blob)*n_n
                if(tf%10 ==0):
                            print('cx_freq is: ' + str(cx_frequency))
                nu = ((L_c*(R**2)/(math.sqrt(8)*(rho_s**3))**0.2))*(cx_frequency*rho_s/c_s)
            else:
                nu = 0
                cx_frequency = 0
            a_star = ((4*(L_c**2)/rho_s*R)**0.2)*rho_s
            a_b_squiggle = np.array(ab_list)/a_star

        if(TRACK_theoryVelocity or TRACK_sizeVvb):
            z_slice_midplane = int(len((grid_ionDensity[2][1:] + grid_ionDensity[2][:-1])/2)/2)  # slicing at midplane
            z_slice_end = len(grid_ionDensity[2])-1

            n_mid = values_ionDensity[:, :, z_slice_midplane]
            n_sh = values_ionDensity[:, :, 0]

            n_mid = np.array(n_mid[rowCutSize:(len(n_mid)-rowCutSize-1), rowCutSize:(len(n_mid[0])-rowCutSize-1)])
            n_sh = np.array(n_sh[rowCutSize:(len(n_sh)-rowCutSize-1), rowCutSize:(len(n_sh[0])-rowCutSize-1)])
            data_elecTemp = pg.Data(open_path  + '%s_electron_Temp_%d.bp' %(filename, tf))
            dg_elecTemp = pg.GInterpModal(data_elecTemp)
            grid_elecTemp, values_elecTemp = dg_elecTemp.interpolate()
            elecTempGrid = values_elecTemp[:, :, z_slice_midplane, 0]
            elecTempGrid_trimmed = elecTempGrid[rowCutSize:(len(elecTempGrid)-rowCutSize-1), rowCutSize:(len(elecTempGrid[0])-rowCutSize-1)]
            data_ionTemp = pg.Data(open_path  + '%s_ion_Temp_%d.bp' %(filename, tf))
            dg_ionTemp = pg.GInterpModal(data_ionTemp)
            grid_ionTemp, values_ionTemp = dg_ionTemp.interpolate()
            ionTempGrid = values_ionTemp[:, :, z_slice_midplane, 0]
            ionTempGrid_trimmed = ionTempGrid[rowCutSize:(len(ionTempGrid)-rowCutSize-1), rowCutSize:(len(ionTempGrid[0])-rowCutSize-1)]

            p = ionDensityGrid_trimmed*(np.array(ionTempGrid_trimmed)  + np.array(elecTempGrid_trimmed))
            p0 = n0 * (Te0 + Ti0)
            pMax = np.where(p == p.max())
            
            sigma_v = (n_sh[pMax[0][0], pMax[1][0]] / n_mid[pMax[0][0], pMax[1][0]])[0]
            if(tf%10 == 0):
                print('at tf = ' + str(tf) + ', sigma_v is: ' + str(sigma_v))
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

            p = ionDensityGrid_trimmed*(np.array(ionTempGrid_trimmed)  + np.array(elecTempGrid_trimmed))
            p0 = n0 * (Te0 + Ti0)
            current_ab = ab_list[len(ab_list)-1]
            delta_p = p.max() - p0
            #print('current a_b is: ' + str(current_ab)

            delta_n = ionDensityGrid_trimmed.max() - n0
            #print('delta_n/n is: ' + str(delta_n/ionDensityGrid_trimmed.max()))
            #print('delta_p/p is: ' + str(delta_p/p.max()))
            #if(tf ==0):
                #print('At t=0, delta_n is: ' + str(delta_n))
            alpha = 1
            delP_running.append(delta_p/p.max())
            if(filename.find("nFac") == -1):
               v_b_upper = 2*math.sqrt(current_ab/R)*c_so*(delta_p/p.max())
               v_b_lower = 1+((alpha*2*math.sqrt(R)*sigma_v*(current_ab**(5/2)))/(L_c*(rho_so**2)))
               v_b = v_b_upper / v_b_lower
               vb_list.append([v_b])
               alpha=0.1
               v_b_lower = 1+((alpha*2*math.sqrt(R)*sigma_v*(current_ab**(5/2)))/(L_c*(rho_so**2)))
               v_b = v_b_upper / v_b_lower
               vb_list[len(vb_list)-1].append(v_b)
            else:
               v_b_upper = 2*math.sqrt(current_ab/R)*c_so*(delta_p/p.max())
               v_b_lower = 1+((alpha*2*math.sqrt(R)*sigma_v*(current_ab**(5/2)))/(L_c*(rho_so**2))) +(math.sqrt(R*current_ab)*cx_frequency)/(2*c_so) 
               v_b = v_b_upper / v_b_lower
               vb_list.append([v_b])
               alpha=0.1
               v_b_lower = 1+((alpha*2*math.sqrt(R)*sigma_v*(current_ab**(5/2)))/(L_c*(rho_so**2))) +(math.sqrt(R*current_ab)*cx_frequency)/(2*c_so)
               v_b = v_b_upper / v_b_lower
               vb_list[len(vb_list)-1].append(v_b)
        if(TRACK_thermalEnergy):
            data_ionTemp = pg.Data(open_path  + '%s_ion_Temp_%d.bp' %(filename, tf))
            dg_ionTemp = pg.GInterpModal(data_ionTemp)
            grid_ionTemp, values_ionTemp = dg_ionTemp.interpolate()
            ionTempGrid = values_ionTemp[:, :, :, 0]
            ionTempGrid_trimmed = ionTempGrid[rowCutSize:(len(ionTempGrid)-rowCutSize-1), rowCutSize:(len(ionTempGrid[0])-rowCutSize-1), :]

            data_ionDensity_2 = pg.Data(open_path  + '%s_ion_M0_%d.bp' %(filename, tf))
            dg_ionDensity_2 = pg.GInterpModal(data_ionDensity_2)
            grid_ionDensity_2, values_ionDensity_2 = dg_ionDensity_2.interpolate()
            ionDensityGrid_2 = values_ionDensity_2[:, :, z_slice, 0]
            ionDensityGrid_3D_2  = values_ionDensity_2[:, :, :, 0]
            ionDensityGrid_trimmed_2 = ionDensityGrid_2[rowCutSize:(len(ionDensityGrid_2)-rowCutSize-1), rowCutSize:(len(ionDensityGrid_2[0])-rowCutSize-1)]
            ionDensityGrid_3D_trimmed_2 = ionDensityGrid_3D_2[rowCutSize:(len(ionDensityGrid_2)-rowCutSize-1), rowCutSize:(len(ionDensityGrid_2[0])-rowCutSize-1), :]
            
            
            p_ion = ionDensityGrid_3D_trimmed_2*(np.array(ionTempGrid_trimmed))
            E_thermal_ion = (3/2)*np.sum(p_ion)*dx*dy*dz
            Etherm_list_ion.append(E_thermal_ion)

            data_elecTemp = pg.Data(open_path  + '%s_electron_Temp_%d.bp' %(filename, tf))
            dg_elecTemp = pg.GInterpModal(data_elecTemp)
            grid_elecTemp, values_elecTemp = dg_elecTemp.interpolate()
            elecTempGrid = values_elecTemp[:, :, :, 0]
            elecTempGrid_trimmed = elecTempGrid[rowCutSize:(len(elecTempGrid)-rowCutSize-1), rowCutSize:(len(elecTempGrid[0])-rowCutSize-1), :]

            
            p_elec = ionDensityGrid_3D_trimmed*(np.array(elecTempGrid_trimmed))
            E_thermal_elec = (3/2)*np.sum(p_elec*dx*dy*dz)#*dx*dy*dz
            Etherm_list_elec.append(E_thermal_elec)
  
            
        if(TRACK_kineticEnergy):
            data_phi = pg.Data(open_path  + '%s_phi_%d.bp' %(filename, tf))
            dg_phi = pg.GInterpModal(data_phi)
            grid_phi, values_phi = dg_phi.interpolate()
            phiGrid = values_phi[:, :, :, 0]
            phiGrid_trimmed = phiGrid[rowCutSize:(len(phiGrid)-rowCutSize-1), rowCutSize:(len(phiGrid)-rowCutSize-1), :]
            #for i in range(0, len(phiGrid))
            #dif_x = np.diff(phiGrid_trimmed[:,:,0], axis=0)/dx
            grad_x = np.gradient(phiGrid_trimmed, dx)[0]
            grad_y = np.gradient(phiGrid_trimmed, dy, axis=1)
            E_kinetic = ((0.5)*np.sum(grad_x**2 + grad_y**2)*dx*dy*dz)
            Ekin_list.append(E_kinetic)
            
        if(TRACK_compactness):
            if(os.path.isfile(open_path + 'raytrace/file_number' + str(tf) + '_contour_number_0.txt')): 
                contour = np.loadtxt(open_path +  'raytrace/file_number' + str(tf) + '_contour_number_0.txt')
                contour_area = PolyArea(contour[:,0], contour[:, 1])
                if(tf == 0):
                    initial_contour_area = PolyArea(contour[:,0], contour[:, 1])

                starting_point = contour[0]
                last_point = starting_point
                perimeter = 0
                for i in range(1, len(contour)):
                    if(i == 1):
                        perimeter  = perimeter + ( ((starting_point - contour[i])[0])**2 + ((starting_point - contour[i])[1])**2  )**0.5
                        last_point = contour[i]
                        #print('first point. perimeter: ' + str(perimeter))
                    elif (i == len(contour)-1):
                        perimeter  = perimeter + ( ((starting_point - contour[i])[0])**2 + ((starting_point - contour[i])[1])**2  )**0.5
                        perimeter  = perimeter + (((contour[i] - starting_point)[0])**2 + ((contour[i] - starting_point)[1])**2 )**0.5
                        #print('last point. perimeter: ' + str(perimeter))
                    else:
                        perimeter  = perimeter + ( ((last_point - contour[i])[0])**2 + ((last_point - contour[i])[1])**2  )**0.5
                        last_point = contour[i]
                        #print('middle point. perimeter: ' + str(perimeter))
                circle_circumference = 2*math.pi*math.sqrt(contour_area/math.pi)
                front_track_cCompactness.append(circle_circumference/perimeter)
            else:
                front_track_cCompactness.append(front_track_cCompactness[len(front_track_cCompactness)-1])
            n_delta = ionDensityGrid_trimmed - (n0)
            n_delta = np.where(n_delta < 0, 0, n_delta)
            if(tf == 0):
                n_delta_zero = n_delta
            dp_x = front_track_Density[len(front_track_Density)-1][0]
            dp_y = front_track_Density[len(front_track_Density)-1][1]
            if(tf == 0):
                heaviside_zero = (X_trimmed - X_trimmed[dp_x, dp_y])**2 + (Y_trimmed - Y_trimmed[dp_x, dp_y])**2
                heaviside_zero = np.where(heaviside_zero < (sigma**2), True, False)
                n_heaviside_zero = ionDensityGrid_trimmed[np.where(heaviside_zero == True)[0], np.where(heaviside_zero == True)[1]]
                n_heaviside = n_heaviside_zero 
            else:
                heaviside = (X_trimmed - X_trimmed[dp_x, dp_y])**2 + (Y_trimmed - Y_trimmed[dp_x, dp_y])**2
                heaviside = np.where(heaviside < (sigma**2), True, False)
                n_heaviside = ionDensityGrid_trimmed[np.where(heaviside == True)[0], np.where(heaviside == True)[1]]
            Icompactness = (np.sum(n_heaviside)*dx*dy) / (np.sum(n_heaviside_zero)*dx*dy)
            Sourcecompactness = Icompactness * ((np.sum(n_delta_zero)*dx*dy) / (np.sum(n_delta)*dx*dy)  )

            front_track_tCompactness_noComp.append(Icompactness)
            front_track_tCompactness_Comp.append(Sourcecompactness)
 
    corrected = []
    for i in range(0, len(front_track_Density)):
                realMax_x = X_trimmed[front_track_Density[i][0], front_track_Density[i][1]]
                realMax_y = Y_trimmed[front_track_Density[i][0], front_track_Density[i][1]]
                corrected.append([realMax_x, realMax_y])
    front_track_Density = corrected
    corrected_grad = []
    for i in range(0, len(front_track_Gradient)):
                realMax_x = X_trimmed[front_track_Gradient[i][0], front_track_Gradient[i][1]]
                realMax_y = Y_trimmed[front_track_Gradient[i][0], front_track_Gradient[i][1]]
                corrected_grad.append([realMax_x, realMax_y])
    front_track_Gradient = corrected_grad

    if(TRACK_sizeVvb):
        x_cmTrack = np.array(front_track_CM)[0:cutoff,0]
        t = np.array(range(timeframe_start, cutoff))*timestep
        t_short = np.arange(timeframe_start, 50)*timestep
        cm_track_spline = sc.interpolate.UnivariateSpline(t, x_cmTrack)
        cm_velocity_spline = cm_track_spline.derivative()
        vb_comparisonlist.append( [float(filename[0] + '.' + filename[1] + filename[2]), np.average(cm_velocity_spline(t_short)), filename, np.average(delP_running)])
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size":10,
            "image.cmap": 'inferno', # Don't use rainbow cmaps
        })    
    if(save_and_plot_tracks):
        t = np.array(range(timeframe_start, cutoff))*timestep
        t_long = np.array(range(timeframe_start, timeframe_end))*timestep
        dt = t[1]-t[0] # microsecond
        if(TRACK_peakDensity):
            x_densityTrack = np.array(front_track_Density)[0:cutoff,0]
            density_track_spline = sc.interpolate.UnivariateSpline(t, x_densityTrack)
            density_spline_error_pearsonR = sc.stats.pearsonr(x_densityTrack, density_track_spline(t))
            density_spline_error_spearmanR = sc.stats.spearmanr(x_densityTrack, density_track_spline(t))
            density_spline_error_kendallTau = sc.stats.kendalltau(x_densityTrack, density_track_spline(t))
            density_velocity_spline = density_track_spline.derivative()
            plt.figure()
            plt.plot(t, x_densityTrack, color = 'cyan')
            plt.plot(t, density_track_spline(t), color='tab:blue')
            if(filename.find("nFac") == -1):
                plt.title("Radial position peak density point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial position peak density point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Radial position (m)")
            plt.legend(["Raw distance travelled", "Interpolated path"])
            plt.savefig(save_path + filename + "density_distance_graph.png")
            #plt.show()
            plt.figure()
            plt.plot(t, density_velocity_spline(t), color='tab:blue')
            if(filename.find("nFac") == -1):
                plt.title("Radial velocity peak density point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial velocity peak density point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend(["Interpolated path"])
            plt.savefig(save_path + filename + "density_velocity_graph.png")
            #plt.show()

        if(TRACK_density_over_time):
            plt.figure()
            plt.plot(t, np.array(average_density)/n0, color='navy')
            if(filename.find("nFac") == -1):
                        plt.title("Averaged blob density over time [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Averaged blob density over time [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized blob density (1/n0)")
            plt.savefig(save_path + filename + "density_over_time.png")
            
        if(TRACK_peakGradient):
            x_gradientTrack = np.array(front_track_Gradient)[0:cutoff,0]
            gradient_track_spline = sc.interpolate.UnivariateSpline(t, x_gradientTrack)
            gradient_velocity_spline = gradient_track_spline.derivative()
            plt.figure()
            plt.plot(t, x_gradientTrack, color='lime')
            plt.plot(t, gradient_track_spline(t), color='green')
            if(filename.find("nFac") == -1):
                plt.title("Radial position peak gradient point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial position peak gradient point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Radial position (m)")
            plt.legend(["Raw distance travelled", "Interpolated path"])
            plt.savefig(save_path + filename + "gradient_distance_graph.png")
            #plt.show()
            plt.figure()
            plt.plot(t, gradient_velocity_spline(t), color = 'green')
            if(filename.find("nFac") == -1):
                plt.title("Radial velocity peak gradient point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial velocity peak gradient point [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend(["Interpolated path"])
            plt.savefig(save_path + filename + "gradient_velocity_graph.png")
            #plt.show()
        if(TRACK_centerOfMass or TRACK_sizeVvb):
            x_cmTrack = np.array(front_track_CM)[0:cutoff,0]
            cm_track_spline = sc.interpolate.UnivariateSpline(t, x_cmTrack)
            cm_velocity_spline = cm_track_spline.derivative()
            plt.figure()
            plt.plot(t, x_cmTrack, color='yellow')
            plt.plot(t, cm_track_spline(t), color='orange')
            if(filename.find("nFac") == -1):
                plt.title("Radial position center of mass [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial position center of mass [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Radial position (m)")
            plt.legend(["Raw distance travelled", "Interpolated path"])
            plt.savefig(save_path + filename + "cm_distance_graph.png")
            #plt.show()
            plt.figure()
            plt.plot(t, cm_velocity_spline(t), color='orange')
            if(filename.find("nFac") == -1):
                plt.title("Radial velocity center of mass [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial velocity center of mass [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend(["Interpolated path"])
            plt.savefig(save_path + filename + "cm_velocity_graph.png")
            #plt.show()
    
        if(TRACK_raytraceFront):
            raytrace_track_spline = sc.interpolate.UnivariateSpline(t, np.asarray(front_track_raytrace)[0:cutoff,0])
            raytrace_velocity_spline = raytrace_track_spline.derivative()
            plt.figure()
            plt.plot(t, np.asarray(front_track_raytrace)[0:cutoff,0], color = 'pink')
            plt.plot(t, raytrace_track_spline(t), color = 'red')
            if(filename.find("nFac") == -1):
                plt.title("Radial position ray traced front [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial position ray traced front [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Radial position (m)")
            plt.legend(["Raw distance travelled", "Interpolated path"])
            plt.savefig(save_path + filename + "raytrace_distance_graph.png")
            #plt.show()
            plt.figure()
            plt.plot(t, raytrace_velocity_spline(t), color = 'red')
            if(filename.find("nFac") == -1):
                plt.title("Radial velocity ray traced front [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Radial velocity ray traced front [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend(["Interpolated path"])
            plt.savefig(save_path + filename + "raytrace_velocity_graph.png")
            #plt.show()
        if(TRACK_ab):
            plt.figure()
            plt.plot(t, np.array(ab_list[0:cutoff])/a0, color='purple')
            if(filename.find("nFac") == -1):
                plt.title("Normalized blob size [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Normalized blob size [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Size (a0)")
            plt.savefig(save_path + filename + "_ab_size.png")
            #plt.show()
        if(TRACK_kineticEnergy):
            plt.figure()
            plt.plot(t_long, Ekin_list, color='navy')
            if(filename.find("nFac") == -1):
                plt.title("Kinetic Energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Kinetic Energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Energy (Joules)")
            plt.savefig(save_path + filename + "_kinenergy.png")
            #plt.show()
        if(TRACK_thermalEnergy):
            plt.figure()
            plt.plot(t_long, Etherm_list_elec, color='darkorange')
            if(filename.find("nFac") == -1):
                plt.title("Electron thermal energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Electron thermal energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Energy (Joules)")
            plt.savefig(save_path + filename + "_elec_thermenergy.png")
            #plt.show()

            plt.figure()
            plt.plot(t_long, Etherm_list_ion, color='springgreen')
            if(filename.find("nFac") == -1):
                plt.title("Ion thermal energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Ion thermal energy [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            
            plt.xlabel("Time (s)")
            plt.ylabel("Energy (Joules)")
            plt.savefig(save_path + filename + "_ion_thermenergy.png")
            #plt.show()
            
        if(TRACK_regime):
            plt.figure()
            for i in range(0, len(a_b_squiggle)):
                if(a_b_squiggle[i]**(5/2) < 1 and math.sqrt(a_b_squiggle[i])*nu < 1):
                    #regime 1
                    #print('regime 1')
                    plt.plot(i*timestep, a_b_squiggle[i], markersize = 10, color= "green", marker = '.')
                elif(a_b_squiggle[i]**(5/2) > 1 and a_b_squiggle[i]**(5/2) > nu*math.sqrt(a_b_squiggle[i])):
                    #regime 2        
                    plt.plot(i*timestep, a_b_squiggle[i], markersize = 10, color= "blue", marker = '.')
                elif(nu > (1/math.sqrt(a_b_squiggle[i])) and nu > a_b_squiggle[i]**2):
                    #regime 3
                    plt.plot(i*timestep, a_b_squiggle[i], markersize = 10, color= "red", marker = '.')
                    
                else:
                    print('no regime')        
                    plt.plot(tf, point, color = 'orange')
            if(filename.find("nFac") == -1):
                plt.title("Velocity regime over time [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Velocity regime over time [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            
            plt.xlabel("Time (s)")
            plt.ylabel("a sub b")



            mark1 = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=10, label='Inertial Regime')
            mark2 = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=10, label='Sheath Limited Regime')
            mark3 = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=10, label='Neutral Regime')
            plt.legend(handles=[mark1,mark2,mark3], loc=4)
            
            plt.savefig(save_path + filename + "_regimes.png")
            #plt.show()
        if(TRACK_compactness):
            compactness_megalist.append( [float(filename[0] + '.' + filename[1] + filename[2]), np.average(front_track_tCompactness_noComp[0:50]), np.average(front_track_cCompactness[0:50]), filename])
            plt.figure()
            plt.plot(t_long, front_track_tCompactness_noComp, color='orange')
            #plt.plot(t_long, front_track_tCompactness_Comp, color='red')
            plt.plot(t_long, front_track_cCompactness, color='blue')
            if(filename.find("nFac") == -1):
                plt.title("Blob compactness [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Blob compactness [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            
            plt.xlabel("Time (s)")
            plt.ylabel("Compactness")
            plt.legend(["Thrysoe equation, no source compensation", "circle comparison"])
            plt.savefig(save_path + filename + "compactness.png")
            #plt.show()
        if(TRACK_theoryVelocity):
            np.savetxt(save_path + filename + "vb_velocity.csv", np.array(vb_list)[:,0], fmt = "%s", delimiter=",")        
            plt.figure()
            plt.plot(np.arange(timeframe_start, timeframe_end)*timestep, np.array(vb_list)[:,0], markersize = 10, color = "magenta", marker = '.')
            if(filename.find("nFac") == -1):
                plt.title("Predicted theory velocity [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Predicted theory velocity [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            
            plt.xlabel("Time (s)")
            plt.ylabel("Vb (m/s)")
            plt.savefig(save_path + filename + "_predictedVb.png")
           # plt.show() 
        if(PLOT_ALL_GRAPHS):
            plt.figure()
            plt.plot(np.arange(timeframe_start, cutoff)*timestep, np.array(vb_list)[:,0], markersize = 5, color = "magenta", marker = '.')
            #plt.plot(np.arange(timeframe_start, cutoff)*timestep, np.array(vb_list)[:,1], markersize = 5, color = "pink", marker = '.')
            #plt.plot(np.arange(timeframe_start, timeframe_end)*timestep, np.array(vb_list)[:,2], markersize = 5, color = "violet", marker = '.')
            #plt.plot(np.arange(timeframe_start, timeframe_end)*timestep, np.array(vb_list)[:,3], markersize = 5, color = "hotpink", marker = '.')
            #plt.plot(t, density_velocity_spline(t), color='tab:blue')
            plt.plot(t, gradient_velocity_spline(t), color = 'green')
            plt.plot(t, cm_velocity_spline(t), color = 'orange')
            plt.plot(t, raytrace_velocity_spline(t), color = 'red')
            if(filename.find("nFac") == -1):
                plt.title("Blob radial velocity [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz]")
            else:
                plt.title("Blob radial velocity [a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0]")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.yticks(np.arange(-2000, 5000, step=1000))
            plt.legend(["Theory Vb", "Simulation peak positive gradient", "Simulation center of mass", "Simulation raytraced front"])
            outfile_name = save_path + filename + "_allmethods_velocity_graph.png"
            plt.savefig(outfile_name)
            #plt.show()
       
        
    ##### Update parameters for graphing the blobs themselves. ###########
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "font.size":10,
            "image.cmap": 'inferno', # Don't use rainbow cmaps
        })


    if(PLOT_ALL_TRACKERS):
        for i in range(timeframe_start, cutoff): # or timeframe_end
            # open each frame
            data_ionDensity = pg.Data(open_path  + '%s_electron_M0_%d.bp' %(filename, i))
            dg_ionDensity = pg.GInterpModal(data_ionDensity)
            grid_ionDensity, values_ionDensity = dg_ionDensity.interpolate()
            ionDensityGrid_plotting = values_ionDensity[:, :, z_slice, 0]
            # take gradient
            M0_gradient = np.gradient(ionDensityGrid_plotting, dx, axis=0)
            #M0_gradient = M0_gradient[10:46, 10:46]
            plt.figure()
            if(filename.find("nFac") == -1):
                plt.title("[a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz] time: " + str(f"{(i*timestep/(10e-6)):.2f}") + "*10e-6 (s)")
            else:
                plt.title("[a" + r"$_\perp$" + " = " + str(aPerp_int) + 'a0, a' + r"$_\parallel$" + " = " + str(aPar_int) + "Lz, n" + r"$_n$" + " = " + str(nn_int) + "*n0] time: " + str(f"{(i*timestep/(10e-6)):.2f}") + "*10e-6 (s)")
            pyplt.pcolormesh(np.transpose(X), np.transpose(Y), np.transpose(ionDensityGrid_plotting))
            pyplt.colorbar()
            pyplt.tight_layout()
            if(TRACK_centerOfMass):
                x_location_cmFront = front_track_CM[i][0]
                y_location_cmFront = front_track_CM[i][1]
                plt.plot(x_location_cmFront, y_location_cmFront , markersize = 10, color= "white", marker = 'X')
            if(TRACK_peakDensity):
                x_location_densityFront = front_track_Density[i][0]
                y_location_densityFront = front_track_Density[i][1]
                plt.plot(x_location_densityFront, y_location_densityFront , markersize = 10, color= "cyan", marker = 'X')
            if(TRACK_peakGradient):
                x_location_gradFront = front_track_Gradient[i][0]
                y_location_gradFront = front_track_Gradient[i][1]
                plt.plot(x_location_gradFront, y_location_gradFront , markersize = 10, color= "lime", marker = 'X')

            if(TRACK_raytraceFront):
                x_location_rayFront = front_track_raytrace[i][0]
                y_location_rayFront = front_track_raytrace[i][1]
                plt.plot(x_location_rayFront, y_location_rayFront , markersize = 10, color= "red", marker = 'X')

            if(i<10):
                plt.savefig(save_path + filename + '_ALLMARKS_fronttrack_M0_frame00' + str(i) + '.png')
            elif (i<100):
                plt.savefig(save_path + filename + '_ALLMARKS_fronttrack_M0_frame0' + str(i) + '.png')
            else:
                plt.savefig(save_path + filename + '_ALLMARKS_fronttrack_M0_frame' + str(i) + '.png')

        cmd = 'convert -delay 10 ' + save_path + '*ALLMARKS_fronttrack_M0_frame[0-9]*.png ' + save_path + filename + '_allmarks.gif'
        os.system(cmd)
        os.system('rm ' + save_path + '*ALLMARKS_fronttrack*')
    print('finished ' + str(filename))

"""
if(TRACK_sizeVvb):
    np.savetxt("postprocess/vb_comparisonlist_CMbase.csv", vb_comparisonlist, fmt = "%s", delimiter=",")
    

if(TRACK_compactness):
    np.savetxt("postprocess/compactness_averaged.csv", compactness_megalist, fmt = "%s", delimiter=",")
"""
