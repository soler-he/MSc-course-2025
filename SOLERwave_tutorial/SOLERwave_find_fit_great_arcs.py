import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Heliocentric
import numpy as np
from numba import njit, prange
from scipy.signal import find_peaks
import time as tm


def pixel_to_great_segments(coord_of_flare,m_0):
    ''' Assigns an azimuthal and radial angle to each pixel of a map, starting from an origin

    :param coord_of_flare:  SkyCoord object, center of the flare with coord_frame of base map
    :param m_0:             map object with wcs, defines the wcs of the calculation; normally first map in the sequence to investigate
    :return:
        pppixel_vectors_xyz: 3d np.array, carthesian coordinates of each pixel in [xyz,pixel_i,pixel_j] == [xyz,pixel_y,pixel_x]
        aangles_along_arc:   2d np.array, angles of concetric circles around the flare for each pixel
        ttheta:              2d np.array, mathematicaly positve angles starting from the north intersecting great arc for each pixel

    ###########################################
    Example how to use this function:

    Flare_coordinates = SkyCoord(Tx=292 * u.arcsec, Ty=127 * u.arcsec, frame=m_seq_base.maps[0].coordinate_frame)
    coords,aangles_along_arc,ttheta = pixel_to_great_segments(Flare_coordinates,m_seq[0])
    ###########################################
    '''

    py,px = m_0.data.shape #Note: This line was added in the submit 2024.11.11, older programms might require adaptiations
    px_range = np.arange(px)
    py_range = np.arange(py)


    ########## Handle inputs for only

    start = coord_of_flare.transform_to(Heliocentric)

    distance_unit = u.m #start.cartesian.xyz.unit

    center = SkyCoord(0 * distance_unit,
                      0 * distance_unit,
                      0 * distance_unit,
                      obstime=start.obstime,
                      observer=start.observer,
                      frame=Heliocentric)

    start_cartesian = start.cartesian.xyz.to(distance_unit).value
    # end_cartesian = end.cartesian.xyz.to(start.distance_unit).value
    center_cartesian = center.cartesian.xyz.to(distance_unit).value

    v1 = start_cartesian - center_cartesian

    r_sun = np.linalg.norm(v1)  # ToDO !!! Is that a sufficient approximation?

    # Defines the second Vector to point to the north pole
    # Form Heliocentric coordinates:
    # The Y-axis is aligned with the component of the vector to the Sun’s north pole that is perpendicular to the Z-axis.
    v2 = np.array([0, r_sun, 0])

    #Vector in the plane of v1,v2 perp to v1
    v3 = np.cross(np.cross(v1, v2), v1)
    v3 = r_sun * v3 / np.linalg.norm(v3)

    #Vector pero to v1 and perp to v3
    v4 = np.cross(v1, v3) / r_sun


    ppx, ppy = np.meshgrid(px_range, py_range)

    ppixel_vectors = m_0.wcs.array_index_to_world(ppy,ppx)      #From astropy doc: (i, j) order, where for an image i is the row and j is the column
    ppixel_vectors = ppixel_vectors.transform_to(Heliocentric)

    pppixel_vectors_xyz = ppixel_vectors.cartesian.xyz.value


    aangles_along_arc = np.arctan2(np.linalg.norm(np.cross(v1, pppixel_vectors_xyz, axisb=0,axisc=0),axis=0),np.einsum('i,ijk->jk',v1, pppixel_vectors_xyz))

    ttheta = np.arctan2( np.einsum('i,ijk->jk',v4, pppixel_vectors_xyz),np.einsum('i,ijk->jk',v3, pppixel_vectors_xyz))

    return pppixel_vectors_xyz,aangles_along_arc,ttheta

def find_segments_from_list(theta_range, ttheta, angles_along_arc_range, aangles_along_arc, m_data_list):
    '''Finds the segments between the theta_range and angles_along_arc_range vectors and produces a mask and mean/var

    :param theta_range: 1d np.array, vector of the segment borders along ttheta direction
    :param ttheta:      2d np.array, mathematically positive angles starting from the north intersecting great arc for each pixel, return of pixel_to_great_segments
    :param angles_along_arc_range: 1d np.array, vector of the segment borders along the arc direction
    :param aangles_along_arc:      2d np.array, angles of concentric circles around the flare for each pixel, return of pixel_to_great_segments
    :param m_data_list:      list of 2d np.array, maps to be investigated [time][py,px]
    :return:
        intensity_mean  3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
        intensity_var   3d np.array, variance of the intensity of each segment [angles_along_arc, theta, time]
        pixel_per_segment 3d np.array, pixel per segment (same for all timesteps) [angles_along_arc, theta, time]
        mask_3          2d np.array, same size as ppx/ppy, mask of the different segments, numeration starting with 1, than following arcs, than theta

    ###########################################
    Example how to use this function:

    map_data_list,m_base,time,m_seq_base = load_preprocessed_fits(path =path,LVL_0_directory = LVL_0_directory)

    coords,aangles_along_arc,ttheta = pixel_to_great_segments(...)
    intensity_mean,intensity_var,mask_3 = find_segments(theta_range,ttheta,angles_along_arc_range,aangles_along_arc,m_data_list)
    #############################################
    '''


    py, px = m_data_list[0][:, :].data.shape  # Note: This line was added in the submit 2024.11.11, older programms might require adaptiations
    px_range = np.arange(px)
    py_range = np.arange(py)

    ppx, ppy = np.meshgrid(px_range, py_range)

    from numba import jit, njit, prange
    # Find ranges to allow for faster processing later
    # the range is one shorter as the vector as the last value marks the outer boundary
    i_range = angles_along_arc_range.shape[0] - 1
    j_range = theta_range.shape[0] - 1
    k_range = aangles_along_arc.shape[0]
    l_range = aangles_along_arc.shape[1]

    mask_3 = np.zeros_like(ttheta)

    intensity_mean = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list)))
    intensity_var = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list)))
    pixel_per_segment = np.zeros((len(angles_along_arc_range) - 1, len(theta_range) - 1, len(m_data_list)))

    # Shifting all Theta Values to positive values
    # Segment bound assures the change in index is at a boundary
    segment_bound = theta_range[0]
    # ttheta[ttheta < segment_bound] = ttheta[ttheta < segment_bound] + 2 * np.pi - segment_bound
    # theta_range[theta_range <= segment_bound] = theta_range[theta_range <= segment_bound] + 2 * np.pi - segment_bound

    ttheta = ttheta - segment_bound
    ttheta[ttheta < 0] = ttheta[ttheta < 0] + 2 * np.pi
    theta_range = theta_range - segment_bound
    theta_range[theta_range < 0] = theta_range[theta_range < 0] + 2 * np.pi

    # njited paralell loop to find the values
    for_loop_for_list(theta_range, ttheta, angles_along_arc_range, aangles_along_arc, ppx, ppy, intensity_mean, intensity_var,
               pixel_per_segment, m_data_list, mask_3, i_range, j_range, k_range, l_range)

    return intensity_mean, intensity_var, pixel_per_segment, mask_3

@njit(parallel=True)
def for_loop_for_list(theta_range,ttheta,angles_along_arc_range,aangles_along_arc,ppx,ppy,intensity_mean,intensity_var,pixel_per_segment,m_data_list,mask_3,i_range,j_range,k_range,l_range):
    """Function to be called in find segments to speed up the process

    :param defined in the find_segments_from_list
    :return Cast to the input matrices
    """
    time_sequ_len = len(m_data_list)

    for i in prange(i_range):       # angle_along_arc
        for j in range(j_range):    # theta


            #data_segement = [[1] for _ in range(m_data.shape[2])]
            data_mean = np.zeros(time_sequ_len)
            data_mean_square = np.zeros(time_sequ_len)
            data_segment_count = 0#np.zeros(m_data.shape[2])

            #test  = np.where(angles_along_arc_range[i] < aangles_along_arc < angles_along_arc_range[i + 1],True,False)
            # 0.08 sec for px_mask, py_mask and mask_2


            # theta [x,y] = theta [k,l]
            for k in range(k_range):
                for l in range(l_range):
                    #mask[k, l] =
                    if bool((angles_along_arc_range[i] < aangles_along_arc[k, l] < angles_along_arc_range[i + 1]) & (theta_range[j] < ttheta[k, l] < theta_range[j + 1])):  # np.bool_(mask_i[:,:,i])
                        px = ppx[k,l]
                        py = ppy[k,l]


                        for index in prange(time_sequ_len):
                            data_mean[index] += m_data_list[index][py,px]
                            data_mean_square[index] += m_data_list[index][py,px]**2

                        data_segment_count +=1


                        mask_3[k,l] = j*(angles_along_arc_range.shape[0]-1) + i + 1


            for index in prange(time_sequ_len):
                intensity_mean[i, j, index] = data_mean[index]/data_segment_count
                intensity_var[i, j, index] = data_mean_square[index]/data_segment_count-(data_mean[index]/data_segment_count)**2
                pixel_per_segment[i, j,index] = data_segment_count

def find_segments_from_list_staggered(theta_range,ttheta,angles_along_arc_range,aangles_along_arc,map_data_list,times_staggered = 4,parameter_dict ={}):
    """ Wrapper to call the find_segments_from_list function multiple times for a staggered output for the segments
    between the theta_range and angles_along_arc_range. Staggering is done equidistant along the angles_along_arc
    direction.

    :param theta_range: 1d np.array, vector of the segment borders along ttheta direction
    :param ttheta:      2d np.array, mathematically positive angles starting from the north intersecting great arc for each pixel, return of pixel_to_great_segments
    :param angles_along_arc_range: 1d np.array, vector of the segment borders along the arc direction
    :param aangles_along_arc:      2d np.array, angles of concentric circles around the flare for each pixel, return of pixel_to_great_segments
    :param m_data_list:            list of 2d np.array, maps to be investigated [time][py,px]
    :param times_staggered:        int, number of steps between two angles in angle_along_arc_range including the first
    :param parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        intensity_mean_staggered  3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
        intensity_var_staggered   3d np.array, variance of the intensity of each segment [angles_along_arc, theta, time]
        distance_staggered        1d np.array, distance values in u.meter
        base_mask                 2d np.array, same size as ppx/ppy, mask of the different segments of the non-staggered
                                               map, numeration starting with 1, than following arcs, than theta
        base_pixel_per_segment    3d np.array, pixel per segment for non-staggered map [angles_along_arc, theta, time]
                                               same value for all time steps
        parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions
    """
    import sunpy
    import astropy.units as u

    intensity_mean, intensity_var, base_pixel_per_segment, base_mask = find_segments_from_list(theta_range, ttheta,
                                                                                       angles_along_arc_range,
                                                                                       aangles_along_arc, map_data_list)
    ts = times_staggered

    distance = (angles_along_arc_range[:-1]+np.diff(angles_along_arc_range)/2) * sunpy.sun.constants.equatorial_radius #Todo: is this a sufficient approximation

    ################### Staggered Plot (with 3 steps between each angle along arch ) ##############################
    #'''
    ##########################################################################################

    diff_angles = (angles_along_arc_range[1]-angles_along_arc_range[0])/ts
    diff_distance = (distance[1]-distance[0])/ts

    intensity_mean_staggered = np.zeros((intensity_mean.shape[0]*ts,intensity_mean.shape[1],intensity_mean.shape[2]))
    intensity_mean_staggered[::ts,:,:] = intensity_mean[:,:,:]

    intensity_var_staggered = np.zeros_like(intensity_mean_staggered)
    intensity_var_staggered[::ts,:,:] = intensity_var[:,:,:]/base_pixel_per_segment # The variance is given for the peak intensity: var_mean = var/N

    distance_staggered = np.zeros(((len(angles_along_arc_range)-1)*ts))
    distance_staggered[::ts] = distance

    max_pixel_per_segment = [np.max(base_pixel_per_segment)]
    min_pixel_per_segment = [np.min(base_pixel_per_segment)]

    for start_index in range(1,ts):
        nr_of_segments = (len(angles_along_arc_range)-1) * (len(theta_range)-1)

        ##############################evaluate Data ######################################################################

        intensity_mean_stag,intensity_var_stag,pixel_per_segment,mask_3_stagg = find_segments_from_list(theta_range,ttheta,angles_along_arc_range+start_index*diff_angles,aangles_along_arc,map_data_list)

        intensity_mean_staggered[start_index::ts,:, :] = intensity_mean_stag[:,:,:]
        intensity_var_staggered[start_index::ts, :, :] = intensity_var_stag[:, :, :]/pixel_per_segment # The variance is given for the peak intensity: var_mean = var/N

        distance_staggered[start_index::ts] = distance + diff_distance*start_index

        max_pixel_per_segment.append(np.max(pixel_per_segment))
        min_pixel_per_segment.append(np.min(pixel_per_segment))

    parameter_dict['general'] = ' '
    parameter_dict['general: aaa distance between segments unstaggered (in Mm)'] = round(np.diff(distance)[0].to_value(u.Mm),2)
    parameter_dict['general: aaa distance between segments unstaggered (in Degree)'] = round(np.diff(angles_along_arc_range)[0] *180/np.pi,2)
    parameter_dict['general: aaa max range unstaggered (in Degree)'] = round(angles_along_arc_range[-1] *180/np.pi,2)
    parameter_dict['general: aaa min range unstaggered (in Degree)'] = round(angles_along_arc_range[0] * 180 / np.pi,2)
    parameter_dict['general: aaa max range unstaggered (in Mm)'] = round(distance[-1].to_value(u.Mm),2)
    parameter_dict['general: aaa min range unstaggered (in Mm)'] = round(distance[0].to_value(u.Mm),2)
    parameter_dict['diagnostic: Maximum Nr of pixel per Segment'] = np.max(max_pixel_per_segment)
    parameter_dict['diagnostic: Minimum Nr of pixel per Segment'] = np.min(min_pixel_per_segment)
    parameter_dict['staggered'] = ' '
    parameter_dict['staggered: times staggered'] = times_staggered
    parameter_dict['staggered: distance between staggered segments (in Mm)'] = round(diff_distance.to_value(u.Mm),2)
    parameter_dict['staggered: aaa between staggered segments (in Degree)'] = round(diff_angles* 180 / np.pi,2)

    return intensity_mean_staggered,intensity_var_staggered,distance_staggered *u.meter,base_mask,base_pixel_per_segment,parameter_dict

@njit(parallel=True)
def extract_loop(segment_values,m_data_list,mask_3_argsort,lower,upper):
        for t in prange(len(m_data_list)):
            segment_values[:,t] = m_data_list[t].flatten()[mask_3_argsort[lower:upper]]


def evaluate_on_mask_from_list(mask_3_full,function,m_data_list,theta_range,angles_along_arc_range,pixel_per_segment,func_dimensions = 1,**kwargs):
    '''Evaluates the function on data of a map sequence using a mask created by the find_segments function.

    :param full_mask:    2d np.array size equal to map size, return of the find_segment function (as maps_data is given by [y,x])
    :param function:                function to evaluate data on f(mat,**kwarg) with mat being [data,time], shall return a np.array with [time, func_dim]
    :param nr_of_segments:          int, number of segments in the mask
    :param m_data:                  3d np.array, data of the map sequence to be evaluated [,,time]
    :param theta_range:             1d np.array, see find_segments function
    :param angles_along_arc_range:  1d np.array, see find_segments function
    :param func_dimensions:         int, nr. of output values of the function
    :param kwargs:                  keyword arguments to be passed to the function
    :return:
        4d np.array with [angles_along_arc-1,theta-1,time,func_dim]
    '''

    time_sequ_len = len(m_data_list)
    nr_of_segments = (len(theta_range) - 1) * (len(angles_along_arc_range) - 1)

    # Maps_data: The first index corresponds to the y direction and the second to the x direction in the two-dimensional pixel coordinate system
    #mask_3_full = full_mask

    #Remove the time axis in pixel per coordinate
    if len(pixel_per_segment.shape) == 2:
        pixel_per_segment = pixel_per_segment[:,0]
    elif len(pixel_per_segment.shape) == 3:
        pixel_per_segment = pixel_per_segment[:, :, 0]

    segment_mat = np.zeros((nr_of_segments, time_sequ_len,func_dimensions))

    #mask_3_sort = np.reshape(mask_3_plot,(mask_3_plot.size))

    mask_3_argsort = np.argsort(mask_3_full, axis = None)
    mask_3_sorted = np.sort(mask_3_full,axis = None)

    #test_dat = np.zeros((m_data_list[0].shape[0]*m_data_list[0].shape[0]))

    for i in range(nr_of_segments):
        print(f'{i} of {nr_of_segments}')

        segment_values = np.zeros([int(pixel_per_segment.flatten()[i]),time_sequ_len])
        lower = np.searchsorted(mask_3_sorted,i+1,side='left')
        upper = np.searchsorted(mask_3_sorted,i+2,side='left')

        extract_loop(segment_values,m_data_list,mask_3_argsort,lower,upper)
        #for t in range(len(m_data_list)):
        #    segment_values[:,t] = m_data_list[t].flatten()[mask_3_argsort[lower:upper]]

        segment_mat[i, :,:] = function(segment_values,**kwargs) #[segment,time,func_dim]

    result_mat = np.reshape(segment_mat, (len(theta_range) - 1, len(angles_along_arc_range) - 1, len(m_data_list),func_dimensions))
    result_mat = np.einsum('ijkl ->jikl' ,result_mat) #[angles_along_arc,theta,time,func_dim]
    print('finished function evaluation')

    return result_mat

def find_coord_from_angles(coord_of_flare, theta_mat, arc_angles_mat):
    ''' Finds the corresponding x,y,z coordinates of theta and arc angles given in a heliocentric view with respect to
    the coord_of_flare Skycoord object. Takes 2d matrices as input to allow for the calculation of multiple points in
    one function call. For use in plot functions, see the example below.

    :param coord_of_flare:  Skycoord Object of the Flare origin at the intended height
    :param theta_mat:       2d np.array; theta values with [theta, aaa]
    :param arc_angles_mat:  2d np.array; aaa   values with [theta, aaa]
    :return:
        3d np.array Matrix with values for vectors [xyu,theta,aaa

    Example:
            coords = find_coord_from_angles(Flare_coordinates, theta_mat, aaa_mat)

            coords_meter = coords * u.meter
            coords_sky = SkyCoord(coords_meter[0, :, :], coords_meter[1, :, :], coords_meter[2, :, :],
                                  obstime=Flare_coordinates.obstime,
                                  observer=Flare_coordinates.observer,
                                  frame=Heliocentric).transform_to(Flare_coordinates.frame)

            ax.plot_coord(coords_sky[j, i])
    '''
    import astropy.units as u
    from astropy.coordinates import BaseCoordinateFrame, SkyCoord
    from sunpy.coordinates import Heliocentric, HeliographicStonyhurst, get_body_heliographic_stonyhurst
    import numpy as np

    start = coord_of_flare.transform_to(Heliocentric)

    distance_unit = u.m # Define meter as the unite that is used #start.cartesian.xyz.unit

    center = SkyCoord(0 * distance_unit,
                      0 * distance_unit,
                      0 * distance_unit,
                      obstime=start.obstime,
                      observer=start.observer,
                      frame=Heliocentric)

    start_cartesian = start.cartesian.xyz.to(distance_unit).value
    #end_cartesian = end.cartesian.xyz.to(start.distance_unit).value
    center_cartesian = center.cartesian.xyz.to(distance_unit).value

    v1 = start_cartesian - center_cartesian
    r_sun = np.linalg.norm(v1)  #ToDO !!! Is that a sufficient approximation?

    # Defines the second Vector to point to the north pole
    # Form Heliocentric coordinates:
    # The Y-axis is aligned with the component of the vector to the Sun’s north pole that is perpendicular to the Z-axis.
    v2 = np.array([0,r_sun,0])

    # Initial v3 vektor pointing north and perpendicular to v1
    v3 = np.cross(np.cross(v1, v2), v1)
    v3 = r_sun * v3 / np.linalg.norm(v3)

    #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    # Temporary Vector needed for the rotating vector
    v4 = np.cross(v1,v3)/r_sun

    # Using einsum to creat the matrices
    # Note on the notation: the number of the first letter indicates the dimensions

    vvv1 = np.einsum('i,jk -> ijk', v1, np.ones_like(theta_mat))
    vvv3 = np.einsum('i,jk -> ijk', v3, np.ones_like(theta_mat))
    vvv4 = np.einsum('i,jk -> ijk', v4, np.ones_like(theta_mat))
    tttheta = np.einsum('i,jk -> ijk',  np.ones_like(v3),theta_mat)
    aaarcangles = np.einsum('i,jk -> ijk',  np.ones_like(v3), arc_angles_mat)

    vvv_rot = vvv3*np.cos(tttheta) + vvv4 * np.sin(tttheta) #+ v1*np.dot(v1,v3) / r_sun**2 * (1-np.cos(theta)) #The last part is redundant as <v1|v3> = 0

    gggreat_arc_points_carthesian = vvv1 *np.cos(aaarcangles) + vvv_rot * np.sin(aaarcangles)

    mask = gggreat_arc_points_carthesian[2,:,:] < 0
    gggreat_arc_points_carthesian[0,mask] = np.nan
    gggreat_arc_points_carthesian[1, mask] = np.nan
    gggreat_arc_points_carthesian[2, mask] = np.nan

    return gggreat_arc_points_carthesian

############################################################################################
#
# Peak fitting and wave tracing
#
#############################################################################################

def peak_finding_algorithm(intensity_mean, intensity_std, theta_range, distance, time, max_nr_peaks_const=5,
                           wavefront_cutof=0.4, min_peak_height=1.1, c_closest=0.03, parameter_dict={}):
    ''' Finds the peaks in the perturbation profiles given by intensty_mean and distance. Uses custom parameters to
    allow fine adjustment

    :param intensity_mean:      3d np.array, mean of the intensity of each segment [angles_along_arc, theta, time]
    :param intensity_std:       3d np.array, standard deviation (= sqrt of variance) of the intensity of each segment [angles_along_arc, theta, time]
    :param theta_range:         1d np.array, vector of the segment borders along ttheta direction
    :param distance:            1d np.array, Distance in length units (e.g. [1e6,2e6] * u.m) of astropy.units
    :param time:                list with strings, time string of the observation of images
    :param max_nr_peaks_const:  int                 ; upper Limit of peaks searched for
    :param wavefront_cutof:     np.float: [% of peak];  Percent of peak height defining the wavefront/trail
    :param min_peak_height:     np.float: [%]       ; Minimum peak height required
    :param c_closest:           np.float: [%]       ; Prominence, height between new peak to minimum to the closest other peak
    :param parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        d_peak_mat      3d np.array, distance of peaks to wave origin [theta,time,peak_nr]
        d_front_mat     3d np.array, distance of front edges to wave origin [theta,time,peak_nr]
        d_trail_mat     3d np.array, distance of trailing edges to wave origin [theta,time,peak_nr]
        peak_mat        3d np.array, amplitude of peaks  [theta,time,peak_nr]
        front_mat       3d np.array, amplitude of front edges  [theta,time,peak_nr]
        trail_mat       3d np.array, amplitude of trailing edges  [theta,time,peak_nr]
        delta_peak_mat  3d np.array, uncertainty of the peaks amplitude to wave origin [theta,time,peak_nr]
        t_sunpy_sec     1d np.array, sunpy time object in seconds
        max_nr_peaks_vec        1d np.array, maximum nr of peaks-found-per-timestep, in each sector [theta]
        max_nr_peaks_const      int, maximum nr of peaks searched for
                parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions

    '''

    from sunpy.time import parse_time

    assert len(intensity_mean.shape) == 3, "intensity_mean requires 3 dimensions [angles_along_arc, theta, time]"
    assert len(intensity_std.shape) == 3, "intensity_sqrt requires 3 dimensions [angles_along_arc, theta, time]"
    #assert intensity_mean.shape == intensity_std.shape, "intensity_mean and intensity_staggered must have the same shape"
    assert len(distance) == intensity_mean.shape[
        0], "distance needs to have the same length as axis 0 of intensity_mean"
    assert theta_range.shape[0] - 1 == intensity_mean.shape[
        1], "theta_range must be shorter by 1 value than the axis 1 of intensity_mean"
    assert len(time) == intensity_mean.shape[2], "the time list has to be equally long as axis 2 of intensity_mean"

    time_dateobj = np.array(time, dtype='datetime64[ns]')  #For Plotting

    time_sunpyobj = parse_time(time)
    # Time vector used for fitting operations to avoid unnecessary large numbers
    t_sunpy_sec = (time_sunpyobj - time_sunpyobj[0]).to_value('sec')

    # Matrices Saving the distance along the solar surface from its origin
    d_peak_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    d_front_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    d_trail_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan

    # Matrices Saving the hight value of the wave
    peak_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    front_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan
    trail_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2], int(max_nr_peaks_const))) * np.nan

    # Uncertainty Matrices
    delta_peak_mat = np.zeros_like(peak_mat)*np.nan

    # TODO Check the implementation of the following
    max_nr_peaks_mat = np.zeros((intensity_mean.shape[1], intensity_mean.shape[2]),
                                dtype=np.int64)  # Maximum number of Peaks found in each theta and timestep

    ##################################################################################
    # Peak Finding Algorithm
    ##################################################################################

    # Integrating over different sectors (e.g. theta range)
    for j in range(intensity_mean.shape[1]):
        # Check if there are nans in the vector indicating a sector over the horizion
        assert not np.any(np.isnan(intensity_mean[:, j, :])), ("The sector with index j = %.0f contains nan values, "
                                                               "likely due to angle_along_arc reaching over the solar "
                                                               "horizon" %(j))


        distance_MM = distance.to_value(u.Mm)
        diff_distance_MM = (distance_MM[1] - distance_MM[0])

        theta_angle = (theta_range[j] + theta_range[j + 1]) / 2

        peaks_in_segment = False

        for t in range(intensity_mean.shape[2]):
            mean_int_vector = np.copy(intensity_mean[:, j, t])
            mean_int_vector[mean_int_vector < 1] = 1
            std_int_vector = intensity_std[:, j, t]

            # find all lokal maxima
            lok_max_arg = find_peaks(mean_int_vector)[0]
            lok_min_arg = find_peaks(-mean_int_vector)[0]

            #Peaks that are below a certain threshold are excempted
            lok_max_arg = lok_max_arg[mean_int_vector[lok_max_arg] >= min_peak_height]

            if lok_max_arg.shape[0] != 0:
                lok_max = mean_int_vector[lok_max_arg]

                peak = np.max(lok_max)
                peak_arg = lok_max_arg[lok_max == peak][0]

                front_args_vec = np.zeros((max_nr_peaks_const + 1))  # Allways has the border argument as a value
                boxing_front_arg = 0
                # d_front_vec = np.ones_like(front_args_vec)
                trail_args_vec = np.ones((max_nr_peaks_const + 1)) * len(
                    mean_int_vector)  # Allways has the border argument as a value
                boxing_trail_arg = len(mean_int_vector)

                peaks_in_segment = True

                for i in range(max_nr_peaks_const):
                    if lok_max.shape[0] != 0:
                        max_nr_peaks_mat[j, t] = i + 1

                        d_peak_mat[j, t, i] = distance_MM[peak_arg]
                        peak_mat[j, t, i] = peak
                        delta_peak_mat[j, t, i] = std_int_vector[peak_arg]

                        cutoff_height = (peak - 1) * wavefront_cutof + 1
                        #cutoff_height = (min_peak_height - 1) * wavefront_cutof + 1 # The wavefront is allways at the same height given by the min peak height

                        smaller_cutoff_arg = np.where(cutoff_height > mean_int_vector)[0]
                        arg_diff = (peak_arg - smaller_cutoff_arg)

                        try:
                            trail_arg = np.max(smaller_cutoff_arg[arg_diff > 0])

                            if trail_arg < boxing_front_arg:  # If there is another front within the trail of the peak, seek min between other front and current peak
                                trail_arg = int(boxing_front_arg + np.argmin(mean_int_vector[
                                                                             int(boxing_front_arg):peak_arg]))
                                trail_args_vec[
                                    i] = trail_arg + 1  # Include the corner piller in the vector (e.g. 2,5,8,6,2) = > 5 is the trail corner

                                d_trail_mat[j, t, i] = distance_MM[trail_arg]
                                trail_mat[j, t, i] = mean_int_vector[trail_arg]
                            else:
                                trail_args_vec[
                                    i] = trail_arg + 1  # Include the corner piller in the vector (e.g. 2,5,8,6,2) = > 5 is the trail corner

                                # Interpolating
                                d_trail_mat[j, t, i] = distance_MM[trail_arg] + (
                                            distance_MM[trail_arg + 1] - distance_MM[trail_arg]) / (
                                                               mean_int_vector[trail_arg + 1] - mean_int_vector[
                                                           trail_arg]) * (
                                                               cutoff_height - mean_int_vector[trail_arg])
                                trail_mat[j, t, i] = cutoff_height
                        except:
                            trail_arg = 0
                            #print('no trail found')
                        try:
                            front_arg = np.min(smaller_cutoff_arg[arg_diff < 0])

                            if front_arg > boxing_trail_arg:  # If there is another trail within the front of the peak, seek min between other trail and current peak
                                front_arg = int(peak_arg + np.argmin(mean_int_vector[peak_arg:int(boxing_trail_arg)]))
                                front_args_vec[
                                    i] = front_arg - 1  # Include the corner piller in the vector,(e.g. 2,5,8,6,2) = > 6 is the front corner

                                d_front_mat[j, t, i] = distance_MM[front_arg]
                                front_mat[j, t, i] = mean_int_vector[front_arg]
                            else:
                                front_args_vec[
                                    i] = front_arg - 1  # Include the corner piller in the vector,(e.g. 2,5,8,6,2) = > 6 is the front corner

                                # Interpolating
                                d_front_mat[j, t, i] = distance_MM[front_arg - 1] + (
                                            distance_MM[front_arg] - distance_MM[front_arg - 1]) / (
                                                               mean_int_vector[front_arg] - mean_int_vector[
                                                           front_arg - 1]) * (
                                                               cutoff_height - mean_int_vector[front_arg - 1])
                                front_mat[j, t, i] = cutoff_height
                        except:
                            front_arg = len(mean_int_vector)
                            #print('no front found')

                        mask = np.ones_like(lok_max, dtype=bool)
                        mask[(lok_max_arg > trail_arg) & (lok_max_arg < front_arg)] = False
                        # np.where((not((lok_max_arg>trail_lok_min_arg)&lok_max_arg<front_lok_min_arg)),lok_max,None)

                        lok_max = lok_max[mask]
                        lok_max_arg = lok_max_arg[mask]

                        peak_valid = False

                        while ((lok_max.shape[0] != 0) & (peak_valid == False)):
                            peak = np.max(lok_max)
                            peak_arg = lok_max_arg[lok_max == peak][0]

                            boxing_front_arg = np.max(front_args_vec[front_args_vec < peak_arg])
                            boxing_trail_arg = np.min(trail_args_vec[trail_args_vec > peak_arg])

                            h_diff_front = peak - np.min(mean_int_vector[int(boxing_front_arg):peak_arg])
                            #arg_diff_front = peak_arg - boxing_front_arg - 1
                            d_diff_front = distance_MM[peak_arg] - distance_MM[
                                int(boxing_front_arg + 1)]  # The one is to correct for the inclusion of the pillar

                            h_diff_trail = peak - np.min(mean_int_vector[peak_arg:int(boxing_trail_arg)])
                            #arg_diff_trail = boxing_trail_arg - peak_arg - 1  #
                            d_diff_trail = distance_MM[int(boxing_trail_arg - 1)] - distance_MM[
                                peak_arg]  # The one is to correct for the inclusion of the pillar

                            ############################################################
                            # Implementation of c as a % height * distance parameter
                            ##########################################################
                            #if (h_diff_trail * d_diff_trail > c_closest) & (
                            #        h_diff_front * d_diff_front > c_closest):
                                # Peak is valid if its product of distance and prominance is above a certain threshold
                            #    peak_valid = True

                            # Implementation of c as % height prominence within the closest neighbours
                            if (h_diff_trail > c_closest) & (h_diff_front > c_closest):
                                # Peak is valid if its product of distance and prominance is above a certain threshold
                                peak_valid = True

                            else:
                                # Otherwise, peak gets removed from list
                                lok_max = lok_max[lok_max_arg != peak_arg]
                                lok_max_arg = lok_max_arg[lok_max_arg != peak_arg]

                if lok_max.shape[0] != 0:
                    print('More than %.0f peaks found in %.0f° theta at ' % (max_nr_peaks_const, theta_angle) + time[t])

    max_nr_peaks_vec = np.nanmax(max_nr_peaks_mat, axis=1)

    #parameter_dict['peak finding algorithm: Limit of peaks searched for'] = max_nr_peaks_const
    #parameter_dict['peak finding algorithm: Percent of peak height defining the wavefront/trail'] = wavefront_cutof
    #parameter_dict['peak finding algorithm: Minimum peak height required'] = min_peak_height
    #parameter_dict['peak finding algorithm: height between new peak to minimum to colosest other peak * distance new peak to other front/trail (% * mM)'] = c_closest

    parameter_dict['peak finding algorithm'] = ' '
    parameter_dict['peak finding algorithm: max_nr_peaks_const'] = max_nr_peaks_const
    parameter_dict['peak finding algorithm: wavefront_cutof (% of peak height)'] = wavefront_cutof
    parameter_dict['peak finding algorithm: min_peak_height (%)'] = min_peak_height
    parameter_dict['peak finding algorithm: c_closest (% * mM)'] = c_closest

    return d_peak_mat, d_front_mat, d_trail_mat, peak_mat, front_mat, trail_mat,delta_peak_mat, t_sunpy_sec, max_nr_peaks_vec, max_nr_peaks_const, parameter_dict


def wave_tracing_algorithm(d_wave_mat, d_wave_std_mat, t_sunpy_sec, max_nr_peaks_vec, peak_tracked=True, v_min_step=10,
                           v_max_step=2000, min_points_in_wave='default', fit_second_feature=False, d_feature2_mat=[],
                           d_feature2_std_mat=[], parameter_dict={}):
    ''' Traces the waves in the output of the peak_finding_algorithm. Can in principle work with any feature (peak,
    front edge, or custom combination of both). A linear fit is applied to all waves detected. The function allows
    the fitting of a secondary feature based on waves detected for the primary one.

    :param d_wave_mat:          3d np.array, distance of feature tracked, [theta,time,peak_index]
    :param d_wave_std_mat:      3d np.array, std of the d_wave_mat parameters
    :param t_sunpy_sec:         1d np.array, sunpy time object in seconds
    :param max_nr_peaks_vec:    1d np.array, maximum nr of peaks-found-per-timestep, in each sector [theta]
    :param v_min_step:          float, minimum velocity of waves between time steps accepted (km/s)
    :param v_max_step:          float, maximum velocity of waves between time steps accepted (km/s)
    :param min_points_in_wave:  string or int, minimum number of time steps with wave detections to accept a new wave
                                default: 'default' => nr is 4 for with average delta_t between observations of less than 60 s
                                                   => nr is 3 for all with more than 60 s
    :param fit_second_feature:  Bool, enables fitting of a secondary feature (e.g. trail if peak is the main feature)
    :param d_feature2_mat:      3d np.array, distance of secondary feature, [theta,time,peak_index]
    :param d_feature2_std_mat:  3d np.array, std of the of secondary feature
    :param parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    :return:
        wave_value_dict:    dictionary of the results of the wave tracing algorithm
        most important listed:
            waves_feature_index     1d np.arrays in list of lists, indices of the d_wave_mat corresponding to a wave [theta][wave_nr]
            waves_time_index        1d np.arrays in list of lists, indices of the time_mat corresponding to a wave [theta][wave_nr]
            waves_fit               1d np.arrays in list of lists, fit parameters corresponding to a wave [theta][wave_nr]
            waves_fit_cov           2d np.arrays inlist of lists, covariants of fit parameters corresponding to a wave [theta][wave_nr]
            wave_ind_mat_2          3d np.array, matrix with entries of wave_nr on the position corresponding  with features
                                                 in the d_wave_mat, 0 at every position without a wave association [theta,time,peak_index]
        parameter_dict:      dict, filled with the parameters, in and output of SOLERwave functions
    '''
    ################################################################################
    v_min = v_min_step / 1e3  #converts form km/s to Mm/s
    v_max = v_max_step / 1e3  #converts form km/s to Mm/s

    nr_of_theta_values = d_wave_mat.shape[0]

    waves_feature_index = [[] for _ in range(nr_of_theta_values)]
    waves_time_index = [[] for _ in range(nr_of_theta_values)]
    waves_fit = [[] for _ in range(nr_of_theta_values)]
    waves_fit_cov = [[] for _ in range(nr_of_theta_values)]

    #Lists should a second feature also be fitted
    waves_fit_2 = [[] for _ in range(nr_of_theta_values)]
    waves_fit_2_cov = [[] for _ in range(nr_of_theta_values)]

    nr_of_waves_vec = np.zeros(nr_of_theta_values, dtype=np.int64)

    max_nr_peaks_const = d_wave_mat.shape[2]

    t_diff_sec = np.diff(t_sunpy_sec)

    # Set the minimum peaks per wave to either 4 (default) or 3 in the case of low cadence data
    # Warns the user if they choose to set the min_points_in_waves themselves
    # Todo: Implement with warning module
    #
    if (min_points_in_wave == 'default') and (np.mean(t_diff_sec) >= 60):
        min_points_in_wave = 3
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' wave_tracing_algorithm: Low cadence data detected (> 60s ), minimum peaks per wave set to 3')
    elif min_points_in_wave == 'default':
        min_points_in_wave = 4
    elif (np.mean(t_diff_sec) >= 60) and (min_points_in_wave > 3):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' wave_tracing_algorithm: Warning: Low cadence data detected (> 60s ), It is recommend to set the '
                    'set the minimum peaks per wave (min_points_in_wave) to 3')
    ##########################################################################
    #  Wave Finding Algorithm
    ##########################################################################

    # max_nr_peaks_vec[j] = np.sum(~ np.isnan(d_peak_mat[j,:,:]),None)

    t_wave_mat = np.einsum('i,j -> ij', np.arange(d_wave_mat.shape[1], dtype=np.int64),
                           np.ones(max_nr_peaks_const, dtype=np.int64))
    wave_ind_mat = np.zeros_like(d_wave_mat)  # Note: Differnt theta values start all with nr_waves = 1

    # Matrix for all the waves long enough to be tracked
    wave_ind_mat_2 = np.zeros_like(wave_ind_mat)

    for j in range(nr_of_theta_values):
        nr_waves = 0

        if max_nr_peaks_vec[j] != 0:
            for t in range(d_wave_mat.shape[1] - 1):
                for i in range(max_nr_peaks_const):
                    d_next = d_wave_mat[j, t + 1, :] - d_wave_mat[j, t, i]

                    # Sort for all allowed options
                    mask = np.zeros(max_nr_peaks_const, dtype=bool)
                    mask[(d_next > v_min * t_diff_sec[t]) & (d_next < v_max * t_diff_sec[t])] = True

                    #if j == 3:
                    #    print('hold')

                    if np.sum(mask) != 0:
                        part_of_wave_mask = wave_ind_mat[j, t + 1, mask] != 0
                        if (wave_ind_mat[j, t, i] == 0):
                            #part_of_wave_mask = wave_ind_mat[j, t + 1, mask] != 0
                            unique_wave_nr = np.unique(wave_ind_mat[j, t + 1, mask][part_of_wave_mask])
                            if np.all(~part_of_wave_mask):
                                # Creates a new wave with wave_nr +1 on both step t and t+1
                                nr_waves += 1
                                wave_ind_mat[j, t, i] = nr_waves
                                wave_ind_mat[j, t + 1, mask] = nr_waves
                            elif len(unique_wave_nr) == 1:
                                wave_ind_mat[j, t, i] = unique_wave_nr
                            else:
                                now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                                print(now + 'wave_tracing_algorithm: mulitple upstream waves claim to be origin,'
                                            'the highest wave_nr was chosen')
                                wave_ind_mat[j, t, i] = unique_wave_nr[-1] #Todo: might actually want to check the lenght of wave and decide then
                        elif np.any(part_of_wave_mask):
                            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                            print(now + 'wave_tracing_algorithm: upstream and downstream are already part of '
                                        'waves, upstream is chosen')
                            wave_ind_mat[j, t + 1, mask] = wave_ind_mat[j, t, i] #Todo: might actually want to check the lenght of wave and decide then
                        else:
                            # Adds the values at mask to the wave of the wavepoint
                            wave_ind_mat[j, t + 1, mask] = wave_ind_mat[j, t, i]

            long_enough_waves_ind_vec = []

            # Sorts all waves which are shorter than min_pints_in_wave:
            for nr_w in range(nr_waves):
                # Find the indices of the wave
                ind_vec = np.where(wave_ind_mat[j, :, :].flatten() == nr_w + 1)[0]

                # Find the corresponding time_vec indices
                time_vec = t_wave_mat.flatten()[ind_vec]

                # Check if there are enough unique time indices
                if np.unique(time_vec).size >= min_points_in_wave:
                    long_enough_waves_ind_vec.append(nr_w)

            waves_feature_index[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_time_index[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit_cov[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]

            # Lists if a second feature shall be also fited
            waves_fit_2[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]
            waves_fit_2_cov[j] = [[] for _ in range(len(long_enough_waves_ind_vec))]

            for w_i, nr_w in enumerate(long_enough_waves_ind_vec):
                #waves_feature[j][nr_w].append(d_wave_mat[j, :, :][wave_ind_mat[j, :, :] == nr_w + 1])  #
                #waves_time[j][nr_w].append(t_wave_mat[wave_ind_mat[j, :, :] == nr_w + 1])
                #waves_feature_std[j][nr_w].append(d_wave_std_mat[j, :, :][wave_ind_mat[j, :, :] == nr_w + 1])

                ###############################################
                #
                # Fitting the wave algorithm (added 30.12.2024)
                #
                ###############################################

                # Matrix with only waves marked deemed long enough
                wave_ind_mat_2[wave_ind_mat == nr_w + 1] = nr_w + 1

                ind_vec = np.where(wave_ind_mat[j, :, :].flatten() == nr_w + 1)[0]
                wave_p_vec = d_wave_mat[j, :, :].flatten()[ind_vec]
                time_vec = t_wave_mat.flatten()[ind_vec]
                std_vec = d_wave_std_mat[j, :, :].flatten()[ind_vec]

                try_fitting_prim_feature = False
                try:
                    a, fit_cov = np.polyfit(t_sunpy_sec[time_vec], wave_p_vec, 1,
                                            w=1 / std_vec,
                                            cov='unscaled')

                    waves_feature_index[j][w_i] = ind_vec
                    waves_time_index[j][w_i] = time_vec
                    waves_fit[j][w_i] = a
                    waves_fit_cov[j][w_i] = fit_cov.flatten()

                    try_fitting_prim_feature = True
                except:
                    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                    print(
                        now + ' wave_tracing_algorithm: fitting of a wave theta_index = %.0f , wave_nr = %.0f failed' % (
                        j, nr_w))

                # Tries to fit the second feature if fitting of the primary feature was successful and if a
                # secondary feature shall be fitted
                if fit_second_feature and try_fitting_prim_feature:
                    try:
                        wave_p2_vec = d_feature2_mat[j, :, :].flatten()[ind_vec]
                        std_2_vec = d_feature2_std_mat[j, :, :].flatten()[ind_vec]

                        a, fit_cov = np.polyfit(t_sunpy_sec[time_vec], wave_p2_vec, 1,
                                                w=1 / std_2_vec,
                                                cov='unscaled')

                        #waves_feature_index[j][w_i].append(ind_vec)
                        #waves_time_index[j][w_i].append(time_vec)
                        waves_fit_2[j][w_i] = a
                        waves_fit_2_cov[j][w_i] = fit_cov.flatten()
                    except:
                        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
                        print(now + 'wave_tracing_algorithm: fitting 2nd feature of a wave theta_index = %.0f , '
                                    'wave_nr = %.0f failed' % (j, nr_w))

            nr_of_waves_vec[j] = len(long_enough_waves_ind_vec)

    # Packing the wave values in a dict to allow for easier further us
    wave_value_dict = {}
    #name_list = ['waves_mu','waves_time','waves_std','waves_mu_std','waves_std_std','max_nr_peaks_vec','nr_of_waves_vec','distance_MM','peak_mat','d_range','mu_std_mat','std_std_mat']
    name_list = ['waves_feature_index', 'waves_time_index', 'nr_of_waves_vec', 'peak_tracked', 'max_nr_peaks_vec',
                 'nr_of_waves_vec',
                 'waves_fit', 'waves_fit_cov', 'wave_ind_mat_2', 'fit_second_feature', 'waves_fit_2', 'waves_fit_2_cov']
    for i in name_list:
        wave_value_dict[i] = eval(i)

    parameter_dict['wave tracing algorithm'] = ' '
    parameter_dict['wave tracing algorithm: v_min_step (km/s)'] = v_min_step
    parameter_dict['wave tracing algorithm: v_max_step (km/s)'] = v_max_step
    parameter_dict['wave tracing algorithm: min_points_in_wave'] = min_points_in_wave

    return wave_value_dict, parameter_dict

################################################################################
#
# End Fitting Algorithms (06.12.2024)
#
################################################################################