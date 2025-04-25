import os
import numpy as np
import astropy.units as u
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import Heliocentric,BaseHeliographic,propagate_with_solar_surface
from numba import njit,prange
import time as tm

from sunpy.net.fido_factory import UnifiedResponse


############################################################################################################
#
# Functions called in the Load new event libary file
#
############################################################################################################

def create_folder_structure(path,base_time,instrument_wavelength,tool_name = 'Coronal-Wave-Tool'):
    # !!! The tool name shall NOT contain "_" !!!
    # Creats folder structure, called in "serch new event" for a single function call
    # Can be used if data is already on hard drive

    LVL_0_directory = tool_name+'_' + base_time.replace(':', '_') +'-'+ instrument_wavelength
    path_LVL_0 = os.path.join(path, LVL_0_directory)
    os.makedirs(path_LVL_0,exist_ok=True)

    path_LVL_1_0 = os.path.join(path_LVL_0,'Input_Fits' )
    os.makedirs(path_LVL_1_0,exist_ok=True)

    path_LVL_1_1 = os.path.join(path_LVL_0,'Preprocesed_Fits' )
    os.makedirs(path_LVL_1_1,exist_ok=True)

    path_LVL_1_2 = os.path.join(path_LVL_0, 'Results')
    os.makedirs(path_LVL_1_2,exist_ok=True)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + f' custom_sunpy_file_handler: folder successfully created')
    return path_LVL_0


@njit(parallel=True)
def base_ratio(map_data,m_base_data,i_map,j_map,m_temp2):
    for i in prange(i_map):
        for j in range(j_map):
            m_temp2[i,j] = map_data[i,j] / m_base_data[i,j]
    return m_temp2

def create_preprocessed_input(path_LVL_0,fits_path,base_image_name,end_image_name,binning = 1,min_exposure_time = 1,Instrument_Name=[]):
    """ Differentially rotates, bins and creates the base ratio images of fits input data,
     saves them in path_LVL_/preprocessed_fits. Called in "search new event" for a single function call.
     Can be used with an external path.

    :param path_LVL_0: string, path to the top level directory
    :param fits_path: string, path to the fits files
    :param base_image_name: string, full name of the start/base/reference image, (e.g. 'test.fits')
    :param end_image_name: string, full name of the start/base/reference image, (e.g. 'test.fits')
    :param binning: int, image gets binned quadratically
    :param min_exposure_time: int, minimum exposure time in seconds. !! NOTE needs to be None for observations without exposure time
    :param Instrument_Name: string, instrument name (supported are 'aia', 'secchi','gong',gong_observatories,'ha2')
    :return:
    """

    from astropy.io import fits
    from sunpy.time import parse_time
    import pathlib
    import glob

    #all_file_paths = sorted(glob.glob('*.fits',root_dir=fits_path))
    all_file_paths = sorted(glob.glob(fits_path+'/*'))
    all_file_names = sorted(os.listdir(fits_path))

    start_index = -10
    end_index = -10

    for index,file_name in enumerate(all_file_names):
        if file_name == base_image_name:
            start_index = index
        if file_name == end_image_name:
            end_index = index+1

    assert start_index != -10, "The base_image_name could not be found"
    assert end_index != -10, "The end_image_name could not be found"

    #############################################################
    # Derotate files
    #############################################################

    file_paths = all_file_paths[start_index:end_index]
    file_names = all_file_names[start_index:end_index]

    match Instrument_Name.lower():
        case 'gong': #TODO can currently not be reached (But apparently it still works ?)
            data, header = fits.getdata(file_paths[0], header=True)
            # fix header
            header['cunit1'] = 'arcsec'
            header['cunit2'] = 'arcsec'
            header['cdelt1'] = header['SOLAR-R'] / header['Radius']
            header['cdelt2'] = header['cdelt1']

            m_reference = sunpy.map.Map(data, header)
        case 'ha2':
            from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
            data, header = fits.getdata(file_paths[0], header=True)

            earth = get_body_heliographic_stonyhurst('earth', header['DATE-OBS'])

            header['dsun_obs'] = earth.radius.to_value('m')
            header['hgln_obs'] = earth.lon.to_value('degree')
            header['hglt_obs'] = earth.lat.to_value('degree')

            # Define the rotation angle
            angle_rad = np.deg2rad(header["ANGLE"])

            # Compute the CD matrix values
            cd1_1 = np.cos(angle_rad) * header["CDELT1"]
            cd1_2 = np.sin(angle_rad) * header["CDELT1"]
            cd2_1 = -np.sin(angle_rad) * header["CDELT2"]
            cd2_2 = np.cos(angle_rad) * header["CDELT2"]

            # Update header for SunPy compatibility
            header["CTYPE1"] = "HPLN-TAN"
            header["CTYPE2"] = "HPLT-TAN"
            header["CD1_1"] = cd1_1
            header["CD1_2"] = cd1_2
            header["CD2_1"] = cd2_1
            header["CD2_2"] = cd2_2
            header["EXTEND"] = True

            m_reference = sunpy.map.Map(data, header)
        case _:
            m_reference = sunpy.map.Map(file_paths[0])

    # Checks if there is a minimum exposure time required
    if min_exposure_time is not None:
        assert m_reference.exposure_time.to_value('s') > min_exposure_time, "The reference image exposure time is below the min_exposure_time argument"
        m_reference = m_reference / m_reference.exposure_time

    out_wcs = m_reference.wcs

    path_LVL_1_1 = os.path.join(path_LVL_0, 'Preprocesed_Fits')
    path_LVL_1_1_ref = os.path.join(path_LVL_1_1,base_image_name[:-5]+'_reference.fits')

    if binning != 1:
        m_reference = m_reference.superpixel((binning, binning) * u.pixel)

        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + f' custom_sunpy_file_handler: binning active {binning}x{binning} pixel')
    m_reference.save(path_LVL_1_1_ref, overwrite=True)

    m_base_data = np.array(m_reference.quantity, dtype=np.float64)
    m_base_data[m_base_data == 0] = 100000  # np.nan  #Todo !!: Check if this iduces an error

    for i in range(len(file_paths)-1):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' custom_sunpy_file_handler: process image: '+ '%.f of %.f' % (i+1,len(file_paths)-1))

        match Instrument_Name.lower():
            case 'gong': #'bigbear','cerrotololo','elteideo','learmonth','maunaloa','Udaipur'
                data, header = fits.getdata(file_paths[i+1], header=True)
                # fix header
                header['cunit1'] = 'arcsec'
                header['cunit2'] = 'arcsec'
                header['cdelt1'] = header['SOLAR-R'] / header['Radius']
                header['cdelt2'] = header['cdelt1']

                m_temp = sunpy.map.Map(data, header)
            case 'ha2':
                data, header = fits.getdata(file_paths[i+1], header=True)

                earth = get_body_heliographic_stonyhurst('earth', header['DATE-OBS'])

                header['dsun_obs'] = earth.radius.to_value('m')
                header['hgln_obs'] = earth.lon.to_value('degree')
                header['hglt_obs'] = earth.lat.to_value('degree')

                # Define the rotation angle
                angle_rad = np.deg2rad(header["ANGLE"])

                # Compute the CD matrix values
                cd1_1 = np.cos(angle_rad) * header["CDELT1"]
                cd1_2 = np.sin(angle_rad) * header["CDELT1"]
                cd2_1 = -np.sin(angle_rad) * header["CDELT2"]
                cd2_2 = np.cos(angle_rad) * header["CDELT2"]

                # Update header for SunPy compatibility
                header["CTYPE1"] = "HPLN-TAN"
                header["CTYPE2"] = "HPLT-TAN"
                header["CD1_1"] = cd1_1
                header["CD1_2"] = cd1_2
                header["CD2_1"] = cd2_1
                header["CD2_2"] = cd2_2
                header["EXTEND"] = True

                m_temp = sunpy.map.Map(data, header)
            case _:
                m_temp= sunpy.map.Map(file_paths[i+1])

        # Checks if there is a minimum exposure time required
        if min_exposure_time is not None:
            # Check if the exposure time exceeds the minimum requirement value
            exp_time_check = m_temp.exposure_time.to_value('s') > min_exposure_time
        else:
            exp_time_check = True

        if exp_time_check:
            # Checks if there is a minimum exposure time required
            if min_exposure_time is not None:
                m_temp = m_temp/m_temp.exposure_time

            # Differentialy rotate the images
            with propagate_with_solar_surface():
                m_temp2 = m_temp.reproject_to(out_wcs)


            #########################################################
            # Base Ratio
            ########################################################

            i_map, j_map = m_base_data.shape

            if binning != 1:
                m_temp2 = m_temp2.superpixel((binning, binning) * u.pixel)


            # Float64 is required as NJIT cannot read <8f format.
            map_data = m_temp2.data.astype('float64')

            # Define Array to be filled
            m_temp3 = np.zeros_like(map_data)

            # Calculate Base ratio with a non python parallelized function.
            m_temp3 = base_ratio(map_data, m_base_data, i_map, j_map, m_temp3)

            # Create map to be saved
            m_temp4 = sunpy.map.GenericMap(m_temp3, m_temp2.fits_header)

            # Create new file name
            path_LVL_1_1_derot = os.path.join(path_LVL_1_1,file_names[i+1][:-5]+'_derot_bin_base.fits') #Todo: Change so it splits at the last.fits and append the rest

            # Save derotated, binned and base ratio calculated map
            m_temp4.save(path_LVL_1_1_derot,overwrite=True)## Produces not a full map object, so exposure time etc is lost

            # Adding a new fits keyword
            # https://stackoverflow.com/questions/57611913/how-to-save-and-add-new-fits-header-in-fits-file
            with fits.open(path_LVL_1_1_derot, mode='update') as hdul:
                hdr = hdul[0].header
                hdr['T_ROT'] = (m_temp.fits_header['date-obs'])

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + ' custom_sunpy_file_handler: Preprocessed input saved')
    return


def search_new_event(path,start_time, end_time,Instrument_Name,Wavelength_,custom_binning = None,min_exposure_time = 1,jsoc_notify_mail = ' ',Stereo_Source='STEREO_A'):
    """ Searches and downloads new events using sunpys Fido search request, generates a new folder structure and
    preprocessed data downloaded

    :param path: string, path the folder shall be created
    :param start_time: string, in the format 'JJJJ-MM-DDThh:mm:ss', start time of event
    :param end_time: string, in the format 'JJJJ-MM-DDThh:mm:ss',
    :param Instrument_Name: string, instrument name (supported are 'aia', 'secchi','gong',gong_observatories,'ha2')
    :param Wavelength_: float * u.AA, wavelength in microns
    :param custom_binning: int, image gets binned quadratically
    :param min_exposure_time: int, min exposure time in seconds, set to None by instruments without exposure time
    :param jsoc_notify_mail: string, Example 1: SDO/AIA, 211$\AA$, 2011 Sept 06
    :param Stereo_Source: string, only needed for STEREO SECCHI observations, default 'STEREO_A'
    :return:
    """
    if (Instrument_Name.lower() == 'aia'):
        res_initial: UnifiedResponse = Fido.search(a.Time(start_time, end_time),a.jsoc.Series('aia.lev1_euv_12s'),a.Wavelength(Wavelength_),a.jsoc.Segment('image'),a.jsoc.Notify(jsoc_notify_mail))

        exp_time = res_initial[0,:]['EXPTIME']

        # The first image with sufficiently long exposure time is taken as reference
        reference_image = res_initial[:,exp_time > min_exposure_time][0]

        # Base time
        base_time = reference_image['T_REC'][0][:-1]

        # Second download request with the first (base) image having sufficient exposure time
        start_time_ = np.array(base_time, dtype='datetime64[ns]') - np.timedelta64(1, 's')
        res = Fido.search(a.Time(start_time_, end_time), a.jsoc.Series('aia.lev1_euv_12s'),
                              a.Wavelength(Wavelength_), a.jsoc.Segment('image'), a.jsoc.Notify(jsoc_notify_mail))

        # Instrument plus wavelength in on string
        instrument_wavelength = Instrument_Name.upper() + '_%.f' % Wavelength_.value + 'AA'

    # Case for Secchi data from Stero
    elif (Instrument_Name.lower() == 'secchi'):
        res = Fido.search(a.Time(start_time, end_time), a.Instrument(Instrument_Name), a.Wavelength(Wavelength_),a.Source(Stereo_Source))

        start_time = res[0,:]['Start Time']
        end_time = res[0,:]['End Time']

        # Calculate Exposure time
        dt_sec = (end_time - start_time).to_value('sec')

        # The first image with sufficiently long exposure time is taken as reference
        reference_image = res[:, dt_sec > min_exposure_time][0,0]

        # Creates the correct base time string from the start date
        base_time = str(reference_image['Start Time'].to_value('datetime64'))[:19]

        # Instrument plus wavelenght in on string
        instrument_wavelength = Instrument_Name.upper() + '_%.f' % Wavelength_.value + 'AA'

    # H alpha observations:
    if (Wavelength_.to_value('AA') >= 6562 ) and (Wavelength_.to_value('AA') <= 6563):

        # Gong is the source, not the instrument. The tool will accept it as instrument name if the wavelength
        # corresponds to h_alpha
        if (Instrument_Name.lower() == 'gong'):
            min_exposure_time = None

            res = Fido.search(a.Time(start_time, end_time), a.Wavelength(6562.8 * u.AA),a.Source('gong'))

            Instrument_names = res[0,:]['Instrument']
            Inst_dict = {}

            for index,Inst in enumerate(Instrument_names):
                if Inst in Inst_dict.keys():
                    Inst_dict[Inst] = Inst_dict[Inst] + 1

                    # Logs the Observation as last, should there be no more following
                    Inst_dict[Inst + '_last'] = res[0, :]['Start Time'][index]
                else:
                    Inst_dict[Inst] = 1

                    # Loggs the first observation and the last, should there be no more images
                    Inst_dict[Inst+'_first'] = res[0, :]['Start Time'][index]
                    Inst_dict[Inst + '_last'] = res[0, :]['Start Time'][index]


            now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
            print(now + ' search_new_event: To use gong data, please use the name of the observatory as instrument name')
            print('Following instruments were found in the search range: ')

            for key, value in Inst_dict.items():
                print('%s: %s times' % (key, value))

            # No further action is taken, as the user shall enter the Instrument of interest
            return


            # Gong individual Observatories:
        elif (Instrument_Name.lower() == 'bigbear') or (Instrument_Name.lower() == 'big bear'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('Big bear'),a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_BigBear'

        elif (Instrument_Name.lower() == 'cerrotololo') or (Instrument_Name.lower() == 'cerro tololo'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('Cerro Tololo'),a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_CerroTololo'

        elif (Instrument_Name.lower() == 'elteideo') or (Instrument_Name.lower() == 'el teide'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('El Teide'), a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_ElTeide'

        elif (Instrument_Name.lower() == 'learmonth'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('Learmonth'),a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_Learmonth'

        elif (Instrument_Name.lower() == 'maunaloa') or (Instrument_Name.lower() == 'mauna loa'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('Mauna Loa'),a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_MaunaLoa'

        elif (Instrument_Name.lower() == 'Udaipur'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('Udaipur'),a.Wavelength(6562.8 * u.AA))
            base_time = str(res[0, 0]['Start Time'].to_value('datetime64'))[:19]
            instrument_wavelength = 'GONG_H_alpha_Udaipur'

        # Kanzelhöhe H_alpha
        elif (Instrument_Name.lower() == 'ha2'):
            min_exposure_time = None
            res = Fido.search(a.Time(start_time, end_time), a.Instrument('ha2'), a.Wavelength(6562.8 * u.AA))
            instrument_wavelength ='Kanzelhoehe_H_alpha'

    # Create the folder Structure for the observation
    path_LVL_0 = create_folder_structure(path,base_time,instrument_wavelength,tool_name = 'SOLERwave')

    path_LVL_1_0_0 = os.path.join(path_LVL_0, 'Input_Fits/{file}')

    downloaded_files = Fido.fetch(res[0,:], path=path_LVL_1_0_0,overwrite=False)

    # Find the first image in the download folder, it is the base image
    # Input for the create_preprocessed_input function
    path_LVL_1_0 = os.path.join(path_LVL_0, 'Input_Fits')

    base_image_name = sorted(os.listdir(path_LVL_1_0))[0]
    end_image_name = sorted(os.listdir(path_LVL_1_0))[-1]

    if (custom_binning is None) and (Instrument_Name.lower() == 'aia'):
        binning = 2
    elif custom_binning is None:
        binning = 1
    else:
        binning = custom_binning

    create_preprocessed_input(path_LVL_0, path_LVL_1_0, base_image_name, end_image_name, binning=binning,min_exposure_time = min_exposure_time,Instrument_Name = Instrument_Name)

    print('LVL_0_directory: ' + path_LVL_0)

    return

############################################################################################################
#
# Functions called in the Event file
#
############################################################################################################

def create_folder_structure_result(path, LVL_0_directory, wave_origin_coordinates, direction, width):
    """Creates folder structure for a single wave analysis

    :param path: string, path where LVL_0_directory is found
    :param LVL_0_directory: string, name of LVL_0_directory
    :param wave_origin_coordinates: Skycord object with the coordinates of presumed wave origen
    :param direction: float, direction of wave in degree
    :param width: float, width of wave in degree
    :return: file_path_dict: dictionary used as default input for the SOLWERwave plotting functions
    """

    path_LVL_0 = os.path.join(path, LVL_0_directory)
    os.makedirs(path_LVL_0,exist_ok=True) #Creates new Main directory if not allready defined

    path_LVL_0_Input = os.path.join(path_LVL_0,'Input_Fits' )

    path_LVL_0_Preprocessed = os.path.join(path_LVL_0,'Preprocesed_Fits' )

    path_LVL_0_Results = os.path.join(path_LVL_0, 'Results')

    shortend_LVL0 = ''.join(LVL_0_directory.split('_')[1:])

    result_directory = (shortend_LVL0 + '_Lon%.0f' % (wave_origin_coordinates.Tx.to_value()) + '_Lat%.0f' % (
        wave_origin_coordinates.Ty.to_value()) + 'Dir%.0f' % (direction) + 'W%.0f' % (width))

    filename_appendix = result_directory + '_'+ ''.join(LVL_0_directory.split('_')[1:])
    filename_appendix = ''

    path_LVL_0_Results_0 = os.path.join(path_LVL_0_Results,result_directory)
    #str1_unicode = path_LVL_0_2_0.encode('unicode_escape').decode()  # Step 1: Unicode # https://stackoverflow.com/questions/29557760/long-paths-in-python-on-windows
    #path_LVL_0_2_0 = os.path.abspath(os.path.normpath(str1_unicode))
    os.makedirs(path_LVL_0_Results_0, exist_ok=True)

    path_LVL_0_Results_0_Output = os.path.join(path_LVL_0_Results_0,'Output')
    os.makedirs(path_LVL_0_Results_0_Output,exist_ok=True)

    path_LVL_0_Results_0_Diagnostics = os.path.join(path_LVL_0_Results_0, 'Diagnostics')
    os.makedirs(path_LVL_0_Results_0_Diagnostics, exist_ok=True)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + f' create_folder_structure_result: folder successfully created')

    file_path_dict = {}
    # name_list = ['waves_mu','waves_time','waves_std','waves_mu_std','waves_std_std','max_nr_peaks_vec','nr_of_waves_vec','distance_MM','peak_mat','d_range','mu_std_mat','std_std_mat']
    name_list = ['path_LVL_0','path_LVL_0_Input','path_LVL_0_Results_0','path_LVL_0_Results_0_Output','path_LVL_0_Results_0_Diagnostics','filename_appendix']
    for i in name_list:
        file_path_dict[i] = eval(i)

    return file_path_dict

def load_preprocessed_fits(path,LVL_0_directory):
    """Loads the preprocessed fits files into different list outputs

    :param path: string, path where LVL_0_directory is found
    :param LVL_0_directory: string, name of LVL_0_directory
    :return:
        map_data_list, list with all image data as 2d numpy arrays
        m_reference, sunpy.map.Map object, reference image
        time, list with strings, time string of the observation of images
        sunpy_seq, sunpy.map.Map sequence object of files loaded
    """
    import glob
    import os
    path_LVL_0 = os.path.join(path, LVL_0_directory)
    os.makedirs(path_LVL_0, exist_ok=True)

    path_LVL_0_Preprocessed = os.path.join(path_LVL_0,'Preprocesed_Fits' )


    #files_paths = sorted(glob.glob( '*derot_bin_base.fits',root_dir=path_LVL_0_Preprocessed))
    files_paths = sorted(glob.glob(path_LVL_0_Preprocessed+'/*derot_bin_base.fits'))

    #ref_file = sorted(glob.glob('*reference.fits',root_dir=path_LVL_0_Preprocessed))
    ref_file = sorted(glob.glob(path_LVL_0_Preprocessed+'/*reference.fits'))
    m_reference = sunpy.map.Map(ref_file)

    map_data_list = [[] for _ in range(len(files_paths))]
    time = [[] for _ in range(len(files_paths))]
    sunpy_seq = [[] for _ in range(len(files_paths))]

    for index, file_path in enumerate(files_paths):
        map_temp = sunpy.map.Map(file_path)
        map_data = map_temp.data.astype('float64')  # Float64 is required as NJIT cannot read <8f format.

        # Cast values to lists
        map_data_list[index] = map_data
        time[index] = map_temp.fits_header['T_ROT']
        sunpy_seq[index] = map_temp

    sunpy_seq = sunpy.map.Map(sunpy_seq, sequence=True)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + ' load_preprocessed_fits: files loaded')

    return map_data_list, m_reference, time, sunpy_seq

###############################################################################
#
# Output generator
#
###############################################################################

def create_numerical_output(j,
                            intensity_mean_staggered, intensity_var_staggered, distance_staggered,
                            d_peak_mat, d_front_mat, d_trail_mat, peak_mat, front_mat, trail_mat,delta_peak_mat,time,
                            instr_dir_width_title_string,
                            wave_value_dict, file_path_dict=[], save_path=[], filename_appendix=[]):
    """ Function to create the numerical output files

    :param j: int, sector to be used
    :param **matrices, standard matrix output of the SOLERwave functions
    :param file_path_dict: dict, holds default file paths
    :param save_path: string, custom file path, overruled by file_path_dict if both are given
    :param filename_appendix: string, custom filename_appendix, overruled by file_path_dict if both are given
    :return:
    """
    import pandas as pd #https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html#min-tut-01-tableoriented

    decimals_distance = 1
    decimals_amplitude = 3
    #time_dateobj = np.array(time, dtype='datetime64[ns]')

    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(
            now + ' Print Parameter Dict : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                  'Only the file_path_dict was used')  # TODO: might ues an actual waring package

    if len(file_path_dict) != 0:
        save_path = file_path_dict['path_LVL_0_Results_0_Output']
        filename_appendix = file_path_dict['filename_appendix']

    if len(filename_appendix) == 0:
        filename_appendix = ''


    #########################################################################
    # Save perturbation profiles
    #########################################################################
    distance_Mm = np.round(distance_staggered.to_value('Mm'),decimals_distance)
    intensity_mean = np.concatenate((np.array([distance_Mm]).T,np.round(intensity_mean_staggered[:,j,:],decimals_amplitude)),axis = 1)
    intensity_std = np.concatenate((np.array([distance_Mm]).T,np.round(np.sqrt(intensity_var_staggered[:,j,:]),decimals_amplitude)),axis = 1)

    int_mean_dict_path = os.path.join(save_path, 'perturbation_profile' + filename_appendix + '.csv')
    np.savetxt(int_mean_dict_path, intensity_mean, header='distance in Mm, '+','.join(time),delimiter=',',fmt='%.3f')

    int_mean_dict_path = os.path.join(save_path, 'perturbation_profile_std' + filename_appendix + '.csv')
    np.savetxt(int_mean_dict_path, intensity_std, header='distance in Mm, '+', '.join(time),delimiter=',',fmt='%.3f')

    # Loading options
    # intensity_mean = np.loadtxt(int_mean_dict_path,delimiter=',')[:,1:]
    # distance = np.loadtxt(int_mean_dict_path,delimiter=',')[:,0]
    # df_time = pd.read_csv(int_mean_dict_path)
    # time = list(df_time)[1:]

    ########################################################################
    # Kinematics matrix
    #######################################################################
    nr_of_waves_vec= wave_value_dict['nr_of_waves_vec']
    waves_time_index = wave_value_dict['waves_time_index']
    waves_feature_index = wave_value_dict['waves_feature_index']

    time_dateobj = np.array(time, dtype='datetime64[s]')
    delta_d = np.diff(distance_staggered)[0].to_value('Mm')



    for nr_w in range(nr_of_waves_vec[j]):
        # Creating vectors corresponding to the entries of one identified wave using the
        # index of individual features

        ind_vec = waves_feature_index[j][nr_w]
        wave_d_peak_vec = d_peak_mat[j].flatten()[ind_vec]
        wave_d_front_vec = d_front_mat[j].flatten()[ind_vec]
        wave_d_trail_vec = d_trail_mat[j].flatten()[ind_vec]
        wave_peak_vec = peak_mat[j].flatten()[ind_vec]
        delta_wave_peak_vec = delta_peak_mat[j].flatten()[ind_vec]

        time_index_vec = np.array(waves_time_index[j][nr_w])

        kinematics_dict = {}
        wave_width = wave_d_front_vec-wave_d_trail_vec
        kinematics_dict['time'] = time_dateobj[time_index_vec]
        kinematics_dict['peak distance in Mm'] = np.round(wave_d_peak_vec,decimals_distance)
        kinematics_dict['delta peak distance in Mm'] = np.round(delta_d*np.ones_like(wave_d_peak_vec),decimals_distance)
        kinematics_dict['front distance in Mm'] = np.round(wave_d_front_vec,decimals_distance)
        kinematics_dict['delta front distance in Mm'] = np.round(delta_d*np.ones_like(wave_d_peak_vec),decimals_distance)
        kinematics_dict['wave width in Mm'] = np.round(wave_width,decimals_distance)
        kinematics_dict['delta wave width in Mm'] = np.round(2* delta_d*np.ones_like(wave_width),decimals_distance)
        kinematics_dict['peak amplitude in percent'] = np.round(wave_peak_vec,decimals_amplitude)
        kinematics_dict['delta peak amplitude in percent'] =np.round(delta_wave_peak_vec,decimals_amplitude)

        wave_dict_path = os.path.join(save_path, 'Kinematics_Wave_%.0f'%(nr_w) + filename_appendix + '.csv')

        df_kinematics = pd.DataFrame(kinematics_dict)
        df_kinematics.to_csv(wave_dict_path, index=False)


    ##################################################################
    # Saving directory to be used in other project
    ################################################################
    #https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

    all_wave_values_dict = wave_value_dict.copy()
    all_wave_values_dict['time'] = time

    all_wave_values_dict['intensity_mean'] = intensity_mean_staggered
    all_wave_values_dict['intensity_var'] = intensity_var_staggered
    all_wave_values_dict['distance'] = distance_staggered
    all_wave_values_dict['d_peak_mat'] = d_peak_mat
    all_wave_values_dict['d_front_mat'] = d_front_mat
    all_wave_values_dict['d_trail_mat'] = d_trail_mat
    all_wave_values_dict['peak_mat'] = peak_mat
    all_wave_values_dict['trail_mat'] = trail_mat
    all_wave_values_dict['front_mat'] = front_mat

    all_wave_values_dict['instr_dir_width_title_string'] = instr_dir_width_title_string

    all_wave_values_dict['j'] = j

    import pickle
    wave_values_dict_path = os.path.join(save_path, 'wave_value_dict' + filename_appendix + '.pkl')

    with open(wave_values_dict_path, 'wb') as f:
        pickle.dump(all_wave_values_dict, f)

    # Loading options :
    #with open(wave_values_dict_path, 'rb') as f:
    #    loaded_dict = pickle.load(f)

    now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
    print(now + 'path for wave values dictornary')
    print((wave_values_dict_path))

    return

def print_parameter_dict(parameter_dict,file_path_dict = [],save_path = [],filename_appendix = []):
    """Saves all the parameter of the parameter_dict in a text file

    :param parameter_dict: dict, filled with the parameters, in and output of SOLERwave functions
    :param file_path_dict: dict, holds default file paths
    :param save_path: string, custom file path, overruled by file_path_dict if both are given
    :param filename_appendix: string, custom filename_appendix, overruled by file_path_dict if both are given
    :return:
    """
    if len(file_path_dict) != 0 and ((len(filename_appendix) != 0) or (len(save_path) != 0)):
        now = tm.strftime("%H:%M:%S", tm.localtime(tm.time()))
        print(now + ' Print Parameter Dict : Warning: both a file_path_dict and a save_path and/or name appendix where given. '
                    'Only the file_path_dict was used') # TODO: might ues an actual waring package

    if len(file_path_dict) != 0:
        save_path = file_path_dict['path_LVL_0_Results_0_Diagnostics']
        filename_appendix = file_path_dict['filename_appendix']

    if len(filename_appendix) == 0:
        filename_appendix = ''

    if len(save_path) != 0:
        Parameter_dict_path = os.path.join(save_path, 'Parameter_Dictionary' + filename_appendix+'.txt')

        with open(Parameter_dict_path, 'w') as f:
            for key, value in parameter_dict.items():
                f.write('%s: %s\n' % (key, value))

    return



