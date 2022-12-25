# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:09:40 2022

@author: Luke Brown
"""

#import required modules
import glob, datetime, math, rawpy
import numpy as np
from skimage import filters, measure
import imageio as iio
from uncertainties import unumpy, umath

def process(img_dir, zenith, azimuth, date,
            direction = 'up', mask = 90, min_zenith = 0, max_zenith = 60, zenith_bin = 10, azimuth_bin = 10,
            pai_sat = 8, down_factor = 3, solar_time = 10, lat = None, fcover_zenith = 10, use_miller_rings = False, pre_process_raw = True, ignore_zeros = False):
    
# =============================================================================
# INPUT VALIDATION CHECKS
# =============================================================================
    
    #check img_dir is a string and raise error if not
    if not isinstance(img_dir, str):
        raise ValueError('Variable \'img_dir\' must be a string')
    #check zenith and azimuth are arrays
    if not isinstance(zenith, np.ndarray) & isinstance(azimuth, np.ndarray):
        raise ValueError('Variables \'zenith\' and \'azimuth\' must be arrays')
    
    #check direction is either 'up' or 'down' and raise error if not
    if not (direction == 'up' or direction == 'down'):
        raise ValueError('Variable \'direction\' must be either \'up\' or \'down\'')
    #check mask is less than or equal to 180°
    if mask > 180:
        raise ValueError('Variable \'mask\' must be less than or equal to 180°')
    #check min_zenith, max_zenith, zenith_bin, azimuth_bin, down_factor, solar_time, lat, fcover_zenith and use_miller_rings have single values and raise error if not
    if not (np.size(min_zenith) == 1 &
            np.size(max_zenith) == 1 &
            np.size(zenith_bin) == 1 &
            np.size(azimuth_bin) == 1 &
            np.size(pai_sat) == 1 &
            np.size(down_factor) == 1 &
            np.size(solar_time) == 1 &
            np.size(lat) == 1 &
            np.size(fcover_zenith) == 1 &
            np.size(use_miller_rings) == 1):
        raise ValueError('Variables \'min_zenith\', \'max_zenith\', \'zenith_bin\', \'azimuth_bin\', \'pai_sat\', \'down_factor\', \'solar_time\', \'lat\', \'fcover_zenith\' and \'use_miller_rings\' must have single values only')
    #check min_zenith is less than max_zenith and raise error if not
    if min_zenith >= max_zenith:
        raise ValueError('Variable \'min_zenith\' must be less than \'max_zenith\'')
    
    #check down_factor is an integer and raise error if not
    if not isinstance(down_factor, int):
        raise ValueError('Variable \'down_factor\' must be an integer')
    #check fcover_zenith is less than max_zenith and raise error if not
    if fcover_zenith > max_zenith:
        raise ValueError('Variable \'fcover_zenith\' must be less than or equal to \'max_zenith\'')
    #check lat is less than 90 and greater than -90 and raise error if not
    if not (lat >= -90)  & (lat <= 90):
        raise ValueError('Variable \'lat\' must be less than or equal to 90 and greater than or equal to -90')
    #check use_miller_rings is boolean:
    if not isinstance(use_miller_rings, bool):
        raise ValueError('Variable \'use_miller_rings\' must be a boolean')
    #check pre_process_raw is boolean:
    if not isinstance(pre_process_raw, bool):
        raise ValueError('Variable \'pre_process_raw\' must be a boolean')
    #check ignore_zeros is boolean:
    if not isinstance(ignore_zeros, bool):
        raise ValueError('Variable \'ignore_zeros\' must be a boolean')
	
    #determine the number of zenith rings and azimuth cells
    n_rings = (max_zenith - min_zenith) / float(zenith_bin)
    n_cells = 360 / azimuth_bin
    #check difference between max_zenith and min_zenith is divisble by zenith_bin and raise error if not
    if not float(n_rings).is_integer():
        raise ValueError('Variable \'zenith_bin\' must divide into the difference between \'max_zenith\' and \'min_zenith\' without a remainder')
    #check 360 is divisible by azimuth_bin and raise error if not
    if not float(n_cells).is_integer():
        raise ValueError('Variable \'azimuth_bin\' must divide into 360 without a remainder')
        
# =============================================================================
# LOCATION OF IMAGES
# =============================================================================
        
    #locate images
    images = glob.glob(img_dir + '/*.NEF')
    images.extend(glob.glob(img_dir + '/*.CR2'))
    images.extend(glob.glob(img_dir + '/*.JPG'))
    images.extend(glob.glob(img_dir + '/*.JPEG'))
    images.extend(glob.glob(img_dir + '/*.PNG'))
    images.extend(glob.glob(img_dir + '/*.GIF'))
    images.extend(glob.glob(img_dir + '/*.BMP'))
    images.extend(glob.glob(img_dir + '/*.TIF'))
    images.extend(glob.glob(img_dir + '/*.TIFF'))
    n_images = len(images)
    #check images have been found
    if n_images < 1:
        raise ValueError('No images could be located in \'img_dir\'')
                
# =============================================================================
# CREATION OF ARRAYS TO STORE INTERMEDIATE RESULTS
# =============================================================================
    
    #determine image dimensions
    rows = np.size(zenith, axis = 0)
    cols = np.size(zenith, axis = 1)
    
    #create array to store stack of binary images
    bin_img = np.zeros((rows, cols, n_images))
    
    if direction == 'down':
        #create arrays to store stack of RGB bands
        r = np.zeros((rows, cols, n_images))
        g = np.zeros((rows, cols, n_images))
        b = np.zeros((rows, cols, n_images))
    
    #convert n_rings and n_cells to integer
    n_rings = int(n_rings)
    n_cells = int(n_cells)
    
    #determine DOY from date
    doy = datetime.datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday
    #determine solar declination angle
    dec = 23.45 * math.sin(math.radians(360.0 / 365.0 * (284 + doy)))
    #determine hour angle at specified solar time
    hour_angle = abs(360.0 / 24.0 * (solar_time - 12))
    #determine SZA at specified solar time
    sza = 90 - math.degrees(math.asin(math.sin(math.radians(dec)) * math.sin(math.radians(lat)) + math.cos(math.radians(dec)) * math.cos(math.radians(lat)) * math.cos(math.radians(hour_angle))))
    
    if not use_miller_rings:
        #create arrays to store gap fraction of each azimuth cell within FCOVER, FIPAR and Warren-Wilson zenith rings
        fcover_ring_gf = np.zeros((n_images, n_cells))
        fipar_ring_gf = np.zeros((n_images, n_cells))
        warren_ring_gf = np.zeros((n_images, n_cells))
        
        #determine pixels correspodning to FCOVER zenith ring
        fcover_index = (zenith <= fcover_zenith)
        #determine pixels corresponding to FIPAR zenith ring
        fipar_index = (zenith >= sza - zenith_bin / 2.0) & (zenith <= sza + zenith_bin / 2.0)
        #determine pixels corresponding to Warren-Wilson zenith ring
        warren_index = (zenith >= 57.5 - zenith_bin / 2.0) & (zenith <= 57.5 + zenith_bin / 2.0)
        
    #create array to store intermediate results
    miller_rings_gf = np.zeros((n_images, n_cells, n_rings))
    miller_rings_theta = np.zeros(n_rings)
    
# =============================================================================
# IMAGE BINARISATION AND MASKING
# =============================================================================
    
    #loop through images
    for i in range(n_images):
        #read in image
        if ('.NEF' in images[i]) | ('.CR2' in images[i]):
            if pre_process_raw:
                data = rawpy.imread(images[i]).postprocess()
            else:
                data = rawpy.imread(images[i]).postprocess(gamma = (1, 1), no_auto_bright = True, output_bps = 16) 
        else:
            data = iio.imread(images[i])
        
        #resize according to the downsampling factor
        if down_factor != 1:
            data = measure.block_reduce(data, (down_factor, down_factor, 1), np.mean)
       
        if direction == 'up':
            #threshold blue band
            b = data[:,:,2]
            if ignore_zeros:
                bin_img[:,:,i] = b > filters.threshold_isodata(b[b != 0])
            else:
                bin_img[:,:,i] = b > filters.threshold_isodata(b)
                
        elif direction == 'down':         
            #split red, green and blue bands into individual arrays, casting as float
            r[:,:,i] = data[:,:,0].astype(float)  
            g[:,:,i] = data[:,:,1].astype(float)  
            b[:,:,i] = data[:,:,2].astype(float)  
			
            #calculate excess green and red indices
            ex_green = 2 * g[:,:,i] - r[:,:,i] - b[:,:,i]
            ex_red = 1.4 * r[:,:,i] - g[:,:,i]
            #binarise image
            bin_img[:,:,i] = ex_green - ex_red < 0
			
            if mask > 0:
                #apply angular mask
                bin_img[:,:,i][(azimuth >= 180 - (mask / 2.0)) & (azimuth <= 180 + (mask / 2.0))] = np.nan
                            
# =============================================================================
# COMPUTATION OF GAP FRACTION FOR FCOVER, FIPAR AND WARREN-WISLON ZENITH RINGS    
# =============================================================================
        
        #create azimuth angle counter
        a = 0
        #loop through desired number of azimuth cells, splitting up zenith ring
        for j in range(n_cells):
            #determine maximum azimuth angle of current cell
            cell_azimuth_max = a + azimuth_bin
            #determine pixels corresponding to current azimuth cell
            azimuth_index = (azimuth >= a) & (azimuth <= cell_azimuth_max)

            if not use_miller_rings:          
                #create new variables for current FCOVER, FIPAR and Warren-Wilson cells and determine size
                fcover_cell = bin_img[:,:,i][fcover_index & azimuth_index]
                fcover_cell_size = np.size(fcover_cell)
                fipar_cell = bin_img[:,:,i][fipar_index & azimuth_index]
                fipar_cell_size = np.size(fipar_cell)
                warren_cell = bin_img[:,:,i][warren_index & azimuth_index]
                warren_cell_size = np.size(warren_cell)
                
                #calculate gap fraction for current FCOVER, FIPAR and Warren-Wilson cells
                if fcover_cell_size > 0:
                    fcover_ring_gf[i,j] = float(np.sum(fcover_cell)) / float(fcover_cell_size)
                else:
                    fcover_ring_gf[i,j] = np.nan
                if fipar_cell_size > 0:
                    fipar_ring_gf[i,j] = float(np.sum(fipar_cell)) / float(fipar_cell_size)
                else:
                    fipar_ring_gf[i,j] = np.nan
                if warren_cell_size > 0:
                    warren_ring_gf[i,j] = float(np.sum(warren_cell)) / float(warren_cell_size)
                else:
                    warren_ring_gf[i,j] = np.nan
                
# =============================================================================
# COMPUTATION OF GAP FRACTION FOR MILLER ZENITH RINGS
# =============================================================================
              
            #create zenith angle counter
            z = min_zenith
            for k in range(n_rings):
                #determine central zenith angle of current ring and store in intermediate results array
                miller_rings_theta[k] = z + zenith_bin / 2.0

                #determine pixels corresponding to current zenith ring
                zenith_index = (zenith >= z) & (zenith <= z + zenith_bin)
                
                #create new variable for current cell and calculate size
                miller_cell = bin_img[:,:,i][zenith_index & azimuth_index]
                miller_cell_size = np.size(miller_cell)
            
                #calculate gap fraction for current cell
                if miller_cell_size > 0:
                    miller_rings_gf[i,j,k] = float(np.sum(miller_cell)) / float(miller_cell_size)
                else:
                    miller_rings_gf[i,j,k] = np.nan
                
                #increment zenith angle counter
                z = z + zenith_bin
            #increment azimuth counter
            a = cell_azimuth_max
    
# =============================================================================
# COMPUTATION OF PAIe/GAIe
# =============================================================================
    
    #create dictionary to store final results
    results = {}
    
    #determine the mean and standard error of gap fraction values over all azimuth cells in each zenith ring (per image)
    mean_miller_gf_each_image = np.nanmean(miller_rings_gf, axis = 1)
    se_miller_gf_each_image = np.nanstd(miller_rings_gf, axis = 1) / np.sqrt(np.sum(np.isfinite(miller_rings_gf), axis = 1))
    #pack into uncertainties array
    mean_miller_gf_each_image = unumpy.uarray(mean_miller_gf_each_image, se_miller_gf_each_image)
    
    #determine the mean of gap fraction values in each zenith ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
    mean_miller_gf=np.nanmean(mean_miller_gf_each_image, axis = 0)
    #add uncertainty due to variability between images (i.e. outside FOV)
    outside_fov_miller_gf_u = np.nanstd(unumpy.nominal_values(mean_miller_gf_each_image), axis = 0) / math.sqrt(n_images)
    mean_miller_gf = mean_miller_gf + unumpy.uarray(0, outside_fov_miller_gf_u)
    
    #calculate weights for each zenith ring
    weights = np.sin(np.radians(miller_rings_theta)) * zenith_bin
    weights = weights / np.sum(weights)    
    #calculate cosine of central zentih angle for each zenith ring
    cos_theta = np.cos(np.radians(miller_rings_theta))
    
    #calculate PAIe according to Miller
    results['paie_miller'] = 2 * np.sum(-unumpy.log(mean_miller_gf) * cos_theta * weights)
    
    if use_miller_rings:
        #determine zenith ring closest to 57.5°
        diff_57 = np.abs(miller_rings_theta - 57.5)
        miller_rings_57 = diff_57 == np.min(diff_57)
        #calculate PAIe using Warren-Wilson's method
        results['paie_warren'] = -umath.log(np.mean(mean_miller_gf[miller_rings_57])) / 0.93
    else:
        #determine the mean and standard error of gap fraction values in Warren-Wilson ring over all azimuth cells (per image)
        mean_warren_gf_each_image = np.nanmean(warren_ring_gf, axis = 1)
        se_warren_gf_each_image = np.nanstd(warren_ring_gf, axis = 1) / np.sqrt(np.sum(np.isfinite(warren_ring_gf), axis = 1))
        #pack into uncertainties array
        mean_warren_gf_each_image = unumpy.uarray(mean_warren_gf_each_image, se_warren_gf_each_image)
        
        #determine the mean of gap fraction values in Warren-Wilson ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
        mean_warren_gf=np.nanmean(mean_warren_gf_each_image)
        #add uncertainty due to variability between images (i.e. outside FOV)
        outside_fov_warren_gf_u = np.nanstd(unumpy.nominal_values(mean_warren_gf_each_image)) / math.sqrt(n_images)
        mean_warren_gf = mean_warren_gf + unumpy.uarray(0, outside_fov_warren_gf_u)
        
        #calculate PAIe using Warren-Wilson's method
        results['paie_warren'] = -umath.log(mean_warren_gf) / 0.93
            
# =============================================================================
# COMPUTATION OF PAI/GAI
# =============================================================================
    
    #calculate gap fraction for saturated cells in all zenith rings using Poisson model (specific to each zenith ring)
    miller_sat_gf = np.zeros((n_images, n_cells, n_rings))
    #repeat array for all azimuth cells and all images
    miller_sat_gf[:,:,:] = np.exp(-0.5 * pai_sat / cos_theta)
    
    #substitute gap fractions of zero
    miller_rings_gf[miller_rings_gf == 0] = miller_sat_gf[miller_rings_gf == 0]

    #determine the mean, standard deviation and standard error of the natural logarithm of gap fraction values over all azimuth cells in each zenith ring (per image)
    mean_ln_gf_each_image = np.nanmean(np.log(miller_rings_gf), axis = 1)
    se_ln_gf_each_image = np.nanstd(np.log(miller_rings_gf), axis = 1) / np.sqrt(np.sum(np.isfinite(miller_rings_gf), axis = 1))
    #pack into uncertainties array
    mean_ln_gf_each_image = unumpy.uarray(mean_ln_gf_each_image, se_ln_gf_each_image)
    
    #determine the mean of the natural logarithm of gap fraction values in each zenith ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
    mean_ln_gf = np.nanmean(mean_ln_gf_each_image, axis = 0)
    #add uncertainty due to variability between images (i.e. outside FOV)
    outside_fov_ln_gf_u = np.nanstd(unumpy.nominal_values(mean_ln_gf_each_image), axis = 0) / math.sqrt(n_images)
    mean_ln_gf = mean_ln_gf + unumpy.uarray(0, outside_fov_ln_gf_u)
       
    #calculate PAI according to Miller
    results['pai_miller'] = 2 * np.sum(-mean_ln_gf * cos_theta * weights)
    
    if use_miller_rings:
        #calculate PAI according to Warren-Wilson
        results['pai_warren'] = np.mean(- mean_ln_gf[miller_rings_57]) / 0.93
    else:
        #calculate gap fraction for saturated cells in Warren-Wilson ring using Poisson model
        warren_sat_gf = np.zeros((n_images, n_cells))
        #repeat array for all azimuth cells and images
        warren_sat_gf[:,:] = np.exp(-0.5 * pai_sat / np.cos(np.radians(57.5)))
        
        #substitute gap fractions of zero
        warren_ring_gf[warren_ring_gf == 0] = warren_sat_gf[warren_ring_gf == 0]
        
        #determine the mean and standard error of the natural logarithm of gap fraction values in Warren-Wilson ring over all azimuth cells (per image)
        mean_ln_warren_gf_each_image = np.nanmean(np.log(warren_ring_gf), axis = 1)
        se_ln_warren_gf_each_image = np.nanstd(np.log(warren_ring_gf), axis = 1) / np.sqrt(np.sum(np.isfinite(warren_ring_gf), axis = 1))
        #pack into uncertainties array
        mean_ln_warren_gf_each_image = unumpy.uarray(mean_ln_warren_gf_each_image, se_ln_warren_gf_each_image)
        
        #determine the mean of gap fraction values in Warren-Wilson ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
        mean_ln_warren_gf=np.nanmean(mean_ln_warren_gf_each_image)
        #add uncertainty due to variability between images (i.e. outside FOV)
        outside_fov_ln_warren_gf_u = np.nanstd(unumpy.nominal_values(mean_ln_warren_gf_each_image)) / math.sqrt(n_images)
        mean_ln_warren_gf = mean_ln_warren_gf + unumpy.uarray(0, outside_fov_ln_warren_gf_u)
        
        #calculate PAI according to Warren-Wilson
        results['pai_warren'] = - mean_ln_warren_gf / 0.93
    
# =============================================================================
# COMPUTATION OF CLUMPING INDEX
# =============================================================================
    
    #calculate clumping index as ratio of PAIe to PAI
    results['clumping_miller'] = results['paie_miller'] / results['pai_miller']
    results['clumping_warren'] = results['paie_warren'] / results['pai_warren']

# =============================================================================
# COMPUTATION OF FCOVER AND FIPAR
# =============================================================================
    
    if use_miller_rings:
        #calculate FCOVER using gap fraction of zenith rings of less than or equal to the defined zenith angle
        nadir_rings = miller_rings_theta <= fcover_zenith
        results['fcover'] = 1 - np.mean(mean_miller_gf[nadir_rings])
        #determine zenith rings closest to SZA
        sza_diff = np.abs(miller_rings_theta - sza)
        sza_rings = sza_diff == np.min(sza_diff)
        #calculate instantaneous black sky FIPAR using gap fraction of zenith rings closest to SZA
        results['fipar'] = 1 - np.mean(mean_miller_gf[sza_rings])
    else:
        #determine the mean and standard error of gap fraction values in FCOVER ring over all azimuth cells (per image)
        mean_fcover_gf_each_image = np.nanmean(fcover_ring_gf, axis = 1)
        se_fcover_gf_each_image = np.nanstd(fcover_ring_gf, axis = 1) / np.sqrt(np.sum(np.isfinite(fcover_ring_gf), axis = 1))
        #pack into uncertainties array
        mean_fcover_gf_each_image = unumpy.uarray(mean_fcover_gf_each_image, se_fcover_gf_each_image)
        
        #determine the mean of gap fraction values in FCOVER ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
        mean_fcover_gf=np.nanmean(mean_fcover_gf_each_image)
        #add uncertainty due to variability between images (i.e. outside FOV)
        outside_fov_fcover_gf_u = np.nanstd(unumpy.nominal_values(mean_fcover_gf_each_image)) / math.sqrt(n_images)
        mean_fcover_gf = mean_fcover_gf + unumpy.uarray(0, outside_fov_fcover_gf_u)
        
        #calculate FCOVER
        results['fcover'] = 1 - np.nanmean(mean_fcover_gf)
        
        #determine the mean and standard error of gap fraction values in FIPAR ring over all azimuth cells (per image)
        mean_fipar_gf_each_image = np.nanmean(fipar_ring_gf, axis = 1)
        se_fipar_gf_each_image = np.nanstd(fipar_ring_gf, axis = 1) / np.sqrt(np.sum(np.isfinite(fipar_ring_gf), axis = 1))
        #pack into uncertainties array
        mean_fipar_gf_each_image = unumpy.uarray(mean_fipar_gf_each_image, se_fipar_gf_each_image)
        
        #determine the mean of gap fraction values in FIPAR ring over all images, propagating uncertainty of individual measurements (i.e. within FOV) through calculation
        mean_fipar_gf=np.nanmean(mean_fipar_gf_each_image)
        #add uncertainty due to variability between images (i.e. outside FOV)
        outside_fov_fipar_gf_u = np.nanstd(unumpy.nominal_values(mean_fipar_gf_each_image)) / math.sqrt(n_images)
        mean_fipar_gf = mean_fipar_gf + unumpy.uarray(0, outside_fov_fipar_gf_u)
        
        #calculate instantaneous black sky FIPAR
        results['fipar'] = 1 - np.nanmean(mean_fipar_gf)
        
        #return results dictionary
        return results
        
# =============================================================================
# COMPUTATION OF EUCLIDEAN DISTANCE TO A GIVEN CENTRE
# =============================================================================

def dist_centre(img_size, centre):
    #check img_size and opt_cen are two-element arrays and raise error if not
    if not (np.size(img_size) == 2) & (np.size(centre) == 2):
        raise ValueError('Variables \'img_size\' and \'centre\' must be two-element arrays')
    
    #check opt_cen is less than or equal to img_size and raise error if not
    if np.sum(centre <= img_size) < 2:
        raise ValueError('Variable \'centre\' must be less than or equal to than \'img_size\'')
    
    #calculate squared x and y distances
    x2 = (np.arange(img_size[1]) - centre[1]) ** 2
    y2 = (np.arange(img_size[0]) - centre[0]) ** 2
    
    #create output array
    dist_img = np.zeros(img_size)
    
    #loop through rows and calculate euclidean distances
    for i in range(img_size[0]):
        dist_img[i,:] = np.sqrt(x2 + y2[i])
    
    #return output array
    return dist_img

# =============================================================================
# COMPUTATION OF THE ZENITH ANGLE OF EACH PIXEL
# =============================================================================

def zenith(img_size, opt_cen, cal_fun, down_factor = 3):
    #check img_size and opt_cen are two-element arrays and cal_fun is a three-element array and raise error if not
    if not (np.size(img_size) == 2) & (np.size(opt_cen) == 2) & (np.size(cal_fun) == 3):
        raise ValueError('Variables \'img_size\' and \'opt_cen\' must be two-element arrays and \'cal_fun\' must be a three-element array')
    #check opt_cen is less than or equal to img_size and raise error if not
    if np.sum(opt_cen <= img_size) < 2:
        raise ValueError('Variable \'opt_cen\' must be less than or equal to than \'img_size\'')
    
    #subtract 1 from optical centre
    opt_cen = opt_cen - 1
    
    #calculate distance of each pixel from optical centre
    distance = dist_centre(img_size, opt_cen)
    #calculate zenith angle of each pixel using lens calibration function
    zenith_array = cal_fun[0] * distance ** 3 + cal_fun[1] * distance ** 2 + cal_fun[2] * distance
    
    #return zenith array, resized according to the downsampling factor
    if down_factor == 1:
        return zenith_array
    else:
        return measure.block_reduce(zenith_array, (down_factor, down_factor), np.mean)

# =============================================================================
# COMPUTATION OF THE AZIMUTH ANGLE OF EACH PIXEL
# =============================================================================

def azimuth(img_size, opt_cen, down_factor = 3):
	#check img_size and opt_cen are two-element arrays and raise error if not
    if not (np.size(img_size) == 2) & (np.size(opt_cen) == 2):
        raise ValueError('Variables \'img_size\' and \'opt_cen\' must be two-element arrays')
    #check opt_cen is less than or equal to img_size and raise error if not
    if np.sum(opt_cen <= img_size) < 2:
        raise ValueError('Variable \'opt_cen\' must be less than or equal to than \'img_size\'')
    
    #subtract 1 from optical centre
    opt_cen = opt_cen - 1

    #define constant vector
    ax = 0
    ay = opt_cen[0]
    
    #create azimuth array
    azimuth_array = np.zeros(img_size)
    
    #loop through rows of azimuth array and define dynamic vector     
    for i in range(len(azimuth_array)):
        if i > opt_cen[0]:
            by = (i - opt_cen[0]) * -1
        if i < opt_cen[0]:
            by = opt_cen[0] - i
        if i == opt_cen[0]:
            by = 0
            
        #loop through columns of azimuth array and define dynamic vector
        for j in range(len(azimuth_array[i])):
            if j > opt_cen[1]:
                bx = j - opt_cen[1]
            if j < opt_cen[1]:
                bx = (opt_cen[1] - j) * -1
            if j == opt_cen[1]:
                bx = 0
            
            #calculate dot product and determinant
            dot = ax * bx + ay * by
            det = ax * by - ay * bx
            
            #calculate angle of current element, converting negative values
            angle = (math.degrees((math.atan2(det, dot)))) * -1
            if angle < 0:
                angle = 360 + angle
            azimuth_array[i,j] = angle
        
    #return azimuth array, resized according to the downsampling factor
    if down_factor == 1:
        return azimuth_array
    else:
        return measure.block_reduce(azimuth_array, (down_factor, down_factor), np.mean)
