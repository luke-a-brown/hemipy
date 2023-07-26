# HemiPy: A Python module for automated estimation of forest biophysical variables and uncertainties from digital hemispherical photographs

## Introduction

`hemipy` is an open-source Python module for deriving forest biophysical variables and uncertainties from digital hemispherical photographs in an automated manner. `hemipy` is well-suited to batch processing and supports a wide range of image formats. 

From multi-angular gap fraction determined by an automated binary image classification<sup>[1]</sup>, the following canopy biophysical variables (and their uncertainties) are computed by `hemipy`:

* Effective plant area index (PAI<sub>e</sub>), plant area index (PAI), and the clumping index (Ω)<sup>[2,3]</sup>;
* The instantaneous black-sky fraction of intercepted photosynthetically active radiation (FIPAR);
* The fraction of vegetation cover (FCOVER).

<sup>[1]</sup>For upwards-facing images, Ridler and Calvard's (1978) clustering algorithm is used to separate sky and canopy pixels, as it was shown to be the most robust of 35 algorithms tested by Jonckheere et al. (2005). In this case, only the blue band is used to maximise contrast and minimise the confounding effects of chromatic aberration and within-canopy multiple scattering (Leblanc et al., 2005; Macfarlane et al., 2014, 2007; Zhang et al., 2005). For downwards- facing images, the approach proposed by Meyer and Neto (2008) is adopted, which separates green vegetation from the underlying soil background on the basis of two colour indices.

<sup>[2]</sup>Two alternative approaches are implemented to estimate PAI<sub>e</sub>, in which a random distribution of leaves is assumed: a method derived from Warren-Wilson's (1963) approach, which considers gap fraction at the hinge region surrounding the zenith angle of 57.5° only (where gap fraction is nearly independent of leaf angle distribution), and a generalised version of Miller's (1967) integral, which makes use of a fuller range of multi-angular observations. In both cases, to account for the effects of foliage clumping and derive PAI as opposed to PAI<sub>e</sub>, Lang and Yueqin's (1986) logarithm averaging method is adopted. Ω is computed as the ratio of PAI<sub>e</sub> to PAI.

<sup>[3]</sup>For downwards-facing images, effective green area index (GAI<sub>e</sub>) and green area index (GAI) are provided as opposed to PAI<sub>e</sub> and PAI.

## Additional information and citation

More detailed description, demonstration, and verification of `hemipy` is provided in the accompanying *Methods in Ecology and Evolution* paper. **Please cite the paper in any work making use of the module:**

Brown, L.A., Morris, H., Leblanc, S., Bai, G., Lanconelli, C., Gobron, N., Meier, C., Dash, J. HemiPy: A Python module for automated estimation of forest biophysical variables and uncertainties from digital hemispherical photographs, *Methods Ecol. Evol.*

## Installation

The latest release of `hemipy` can easily be installed using `pip`:

`pip install https://github.com/luke-a-brown/hemipy/archive/refs/tags/v0.1.2.zip`

## Dependencies

`hemipy` makes use of the `imageio`, `numpy`, `rawpy`, `scikit-image` and `uncertainties` modules, as well as several modules included in the Python Standard Library (`datetime`, `glob` and `math`).

## Overview

`hemipy` consists of three main functions. The typical workflow for processing a set of digital hemispherical photographs from a single measurement plot is as follows:

1.	Use the `hemipy.zenith()` and `hemipy.azimuth()` functions to determine the zenith and azimuth angle represented by each pixel of the image (based on the characteristics of the digital camera and lens);
2.	Pass these arrays to the `hemipy.process()` function, along with the directory of images to be processed, the direction (i.e. upwards- or downwards-facing) of these images, and the date and latitude at which the images were acquired (necessary for FIPAR computation):

All images within a directory are processed together to provide a single value (and uncertainty) for each canopy biophysical variable. Therefore, each directory should correspond to a single measurement plot, which typically contains between 5 and 15 images.

## Processing options

The `hemipy.process()` function is highly configurable, and includes the ability to:
* Specify the angular resolution at which gap fraction should be computed (`zenith_bin`, `azimuth_bin`, default is 10°);
* Define the maximum zenith angle to use for the computation of FCOVER, from 0° to the chosen value (`fcover_zenith`, default is 10°);
* Specify the local solar time at which to compute instantaneous black-sky FIPAR (`solar_time`, default is 10:00);
* Use the zenith rings defined in the computation of PAI according to Miller (1967) for deriving FIPAR, FCOVER, and PAI according to the hinge approach (`use_miller_rings`, default is False, i.e. to use dedicated rings centred at the solar zenith angle, nadir, and 57.5°, which is more accurate, but will increase computation time). If True, the rings with the closest central zenith angle are used;
* Set a minimum and maximum zenith angle of analysis for the computation of PAI according to Miller (1967), e.g. to avoid the effects of mixed pixels at the extremes of the image (Jonckheere et al., 2004) (`min_zenith`, default is 0°, `max_zenith`, default is 60°);
* Apply a mask of a specified size to the bottom of downwards-facing images (`mask`, useful for removing the operator’s legs, default is a 90° mask);
* Specify a downsampling factor to speed up the computation (`down_factor`, default is 3);
* Define the ‘saturated’ PAI value used to compute the gap fraction of cells with no gaps (`pai_sat`, only applicable to the computation of PAI, default is 8) (Chianucci, 2013; Weiss and Baret, 2017).
* Specify whether to pre-process RAW images (e.g. as recommended by Macfarlane et al. (2014)) (`pre_process_raw`, default is True);
* Specify whether zeros should be ignored by Ridler and Calvard's (1978) clustering algorithm, which may be useful if processing circular fisheye images (`ignore_zeros`, default is False);
* Specify whether to save the binarised image to the same directory as the input image as an 8-bit PNG (canopy = 0, gaps = 255), which may be useful for quality control purposes (`save_bin_img`, default is False).

## Processing example

The example below demonstrates how `hemipy` can be used to process a set of images with the following directory structure:

* example_data
  - plot_a
    - overstory
      - image_1, image_2, ..., image_14, image_15
    - understory
      - image_1, image_2, ..., image_14, image_15
	  
**Note that the images in the `example_data` folder of this repository do not reflect best-case illumination conditions (i.e. uniform overcast skies or close to sunrise/sunset), and are provided purely to demonstrate the operation of the code!**

```
#import required modules
import numpy as np
import hemipy, glob, exifread

#define input directory (sub-directories correspond to measurement plots and contain images from that plot)
input_dir = 'example_data/'
#define latitude of the site (necessary for FIPAR computation)
lat = 51.7734

#define image size and optical centre
img_size = np.array([3465, 5202])
opt_cen = np.array([1754, 2595])
#define calibration function coefficients (^3, ^2 and ^1)
cal_fun = np.array([0,0,0.0548543])

#calculate the zenith and azimuth angle of each pixel
zenith = hemipy.zenith(img_size, opt_cen, cal_fun)
azimuth = hemipy.azimuth(img_size, opt_cen)

#open output file and write header
output_file = open('example_output.csv', 'w')
output_file.write('Date,Plot,Direction,PAIe_Hinge,PAI_Hinge,Clumping_Hinge,PAIe_Miller,PAI_Miller,Clumping_Miller,FIPAR,FCOVER\n')

#locate and loop through measurement plots
plots = glob.glob(input_dir + '/*')
for i in range(len(plots)):
    layers = glob.glob(plots[i] + '/*')
    
    #locate and loop through understory/overstory layers
    for j in range(len(layers)):       
        #open first image in folder and retrieve date and time from EXIF data
        image = open(glob.glob(layers[j]+'/*.cr2')[0], 'rb')
        tags = exifread.process_file(image)
        #determine date of plot acquisition
        date = str(tags['EXIF DateTimeOriginal'])[0:10].replace(':', '-')
        
        #determine image direction
        if 'overstory' in layers[j]:
            direction = 'up'
        elif 'understory' in layers[j]:
            direction = 'down'
            
        #run the main function and write results to output file
        results = hemipy.process(layers[j], zenith, azimuth, date = date, lat = lat, direction = direction)     
        output_file.write(date + ',' +\
                          layers[j].split('\\')[-2] + ',' +\
                          direction + ',' +\
                          str(results['paie_hinge']) + ',' +\
                          str(results['pai_hinge']) + ',' +\
                          str(results['clumping_hinge']) + ',' +\
                          str(results['paie_miller']) + ',' +\
                          str(results['pai_miller']) + ',' +\
                          str(results['clumping_miller']) + ',' +\
                          str(results['fipar']) + ',' +\
                          str(results['fcover']) + '\n')

#close output file
output_file.close()
```

## References

Chianucci, F., 2013. *Canopy Properties Estimation in Deciduous Forests with Digital Photography*. Università degli Studi della Tuscia.

Jonckheere, I., Fleck, S., Nackaerts, K., Muys, B., Coppin, P., Weiss, M., Baret, F., 2004. Review of methods for in situ leaf area index determination. *Agric. For. Meteorol.* 121, 19–35. https://doi.org/10.1016/j.agrformet.2003.08.027

Lang, A.R.G., Yueqin, X., 1986. Estimation of leaf area index from transmission of direct sunlight in discontinuous canopies. *Agric. For. Meteorol.* 37, 229–243. https://doi.org/10.1016/0168-1923(86)90033-X

Leblanc, S.G., Chen, J.M., Fernandes, R., Deering, D.W., Conley, A., 2005. Methodology comparison for canopy structure parameters extraction from digital hemispherical photography in boreal forests. *Agric. For. Meteorol.* 129, 187–207. https://doi.org/10.1016/j.agrformet.2004.09.006

Macfarlane, C., Grigg, A., Evangelista, C., 2007. Estimating forest leaf area using cover and fullframe fisheye photography: Thinking inside the circle. *Agric. For. Meteorol.* 146, 1–12. https://doi.org/10.1016/j.agrformet.2007.05.001

Macfarlane, C., Ryu, Y., Ogden, G.N., Sonnentag, O., 2014. Digital canopy photography: Exposed and in the raw. *Agric. For. Meteorol.* 197, 244–253. https://doi.org/10.1016/j.agrformet.2014.05.014

Miller, J., 1967. A formula for average foliage density. *Aust. J. Bot.* 15, 141–144. https://doi.org/10.1071/BT9670141

Warren-Wilson, J., 1963. Estimation of foliage denseness and foliage angle by inclined point quadrats. *Aust. J. Bot.* 11, 95–105.

Weiss, M., Baret, F., 2017. *CAN-EYE V6.4.91 User Manual*. Institut National de la Recherche Agronomique, Avignon, France.

Zhang, Y., Chen, J.M., Miller, J.R., 2005. Determining digital hemispherical photograph exposure for leaf area index estimation. *Agric. For. Meteorol.* 133, 166–181. https://doi.org/10.1016/j.agrformet.2005.09.009