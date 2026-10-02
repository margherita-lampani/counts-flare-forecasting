"""
Utility functions for magnetogram preprocessing, calibration, and feature extraction.

This module provides the core data-processing routines used to transform
raw MDI and HMI active-region magnetograms into standardized machine-learning
inputs. It includes FITS handling, quality control, instrument calibration,
magnetic-flux computation, and extraction of physically motivated solar
activity features.

Main functionalities:
    - FITS metadata and timestamp extraction
    - Magnetogram quality-control checks
    - Instrument-specific calibration and smoothing
    - Signed and unsigned magnetic flux estimation
    - Active-region masking and magnetic-field thresholding
    - Magnetic feature computation for flare forecasting
    - Catalogue and label utilities

Implemented magnetic features:
    - Total signed magnetic flux
    - Total unsigned magnetic flux
    - Schrijver R parameter
    - Log-transformed Schrijver R
    - Weighted length of strong-gradient neutral lines (WLsg)
    - Mean and maximum magnetic-field gradient
    - Magnetic-field skewness
    - Magnetic-field kurtosis

The implementation is specifically designed for MDI and HMI active-region
cutouts, where geometrical reprojection and limb-correction procedures are
not required. All derived quantities are computed using physically meaningful
units whenever possible to ensure consistency across instruments.

This module serves as the low-level preprocessing layer used throughout
the flare forecasting pipeline.
"""

from astropy.convolution import convolve, Gaussian2DKernel
from datetime import datetime
import numpy as np
from flare_forecast.utils.sunpy_utils import mapPixelArea, makeBMask
import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.stats import skew, kurtosis
from flare_forecast.utils.sunpy_utils import makeBMask

def read_catalog(filename, **kwargs):
    """
    Reads a csv file into a pandas dataframe and converts any columns
    to datetime that can be successfully converted

    Parameters:
        filename (str):     location of csv file
        kwargs:             optional keyword arguments to pass to pd.read_csv
    
    Returns:
        df:                 converted pandas dataframe
    """
    df = pd.read_csv(filename, **kwargs)
    for col in df.columns[df.dtypes == 'object']:
        try:
            df[col] = pd.to_datetime(df[col])
        except (pd.errors.ParserError, ValueError):
            pass
    return df


def add_label_data(flare_data):
    """
    Adds flare labeling data to a list

    Parameters:
        flare_data (dataframe):     relevant catalog of flares

    Returns:
        file_data (list):           list with labels for C, M, X flares 
                                    and flare intensity
    """
    file_data = []
    for flare_class in ['C', 'M', 'X']:
        if sum(flare_data['CMX'] == flare_class) > 0:
            file_data.append(1)
        else:
            file_data.append(0)
    if len(flare_data) > 0:
        file_data.append('{:0.1e}'.format(max(flare_data['intensity'])))
    else:
        file_data.append(0)
    return file_data


def calculate_flaring_rate(flare_data, window):
    """
    Calculate the rate of flaring over the number of days covered by flare_data

    Parameters: 
        flare_data (dataframe):    catalog of relevant flares 
        window (int):              number of days to calculate flaring rate over
    
    Returns: 
        flare_rate (float):        rate of flaring over window
    """
    # filter only M flares
    flare_data = flare_data[flare_data['intensity'] >= 1e-5]
    # filter unique days
    peak_days = flare_data['peak_time'].dt.date
    unique_days = pd.unique(peak_days)
    # calculate flaring rate
    flare_rate = len(unique_days) / window
    return flare_rate


def extract_date_time(file, data=None, year=None):
    """
    Extract date and time from homogenized MDI/HMI filenames.

    Accepted examples:
    - 20141231_235809_I-12253_HMI.fits
    - 20141231_235809_I-12253_HMI_SIDE1.fits
    - 20100512_120000_QS-0_MDI.fits
    """
    if not file.endswith('.fits'):
        return None, None

    try:
        import os
        base = os.path.basename(file).replace('.fits', '')
        parts = base.split('_')

        # Always trust first two fields
        date = parts[0]
        time = parts[1]

        timestamp = datetime.strptime(date + time, "%Y%m%d%H%M%S")
        return date, timestamp

    except Exception:
        return None, None


def check_quality(data, header):
    """
    Unified quality check for homogenized MDI and HMI datasets.
    """
    q = header.get('QUALITY', None)
    if q is not None and not isinstance(q, (int, float)):
        return False
    return True


def compute_tot_flux(map, Blim=30, area_threshold=16):
    """
    Calculate total signed and unsigned flux for magnetogram by multiplying with 
    pixel area and masking only larger regions of flux
    
    For AR cutouts, uses uniform pixel area since cutouts are already in local projection.

    Parameters:
        map (sunpy map):    LOS magnetogram
        Blim (int):         min threshold for masking (Gauss)
        area_threshold (int): min size of region to mask (pixels)
    
    Returns:
        tot_flux (float):   total signed magnetic flux 
        tot_us_flux (float): total unsigned magnetic flux 
    """
    try:
        # For AR cutouts, use uniform pixel area
        # Cutouts are in local projection, no geometric corrections needed
        pixel_area = np.abs(map.scale[0].value * map.scale[1].value)
        
        # Sanity check
        if pixel_area == 0 or not np.isfinite(pixel_area):
            pixel_area = 1.0  # fallback
        
        area_map_data = np.ones_like(map.data) * pixel_area
        
        # Create mask on Gauss data
        Bmask = makeBMask(map.data, Blim=Blim, area_threshold=area_threshold, 
                         connectivity=2, dilationR=8)
        
        # Multiply by area after masking
        Bdata = map.data * area_map_data
        
        tot_flux = np.nansum(Bdata * Bmask)
        tot_us_flux = np.nansum(np.abs(Bdata * Bmask))
        
    except Exception as e:
        print(f"Warning in compute_tot_flux: {e}, using simplified calculation")
        # Fallback: no area correction
        Bmask = makeBMask(map.data, Blim=Blim, area_threshold=area_threshold, 
                         connectivity=2, dilationR=8)
        tot_flux = np.nansum(map.data * Bmask)
        tot_us_flux = np.nansum(np.abs(map.data * Bmask))
    
    return tot_flux, tot_us_flux


# ---------------------------------------------------------------------------
# Conversion constants
# 1 arcsec on the Sun ≈ 725.27 km  (R☉ ≈ 696 Mm, angular diameter ≈ 1919 arcsec)
# ---------------------------------------------------------------------------
ARCSEC_TO_CM   = 725.27e5           # cm per arcsec
ARCSEC_TO_MM   = 0.72527            # Mm per arcsec  (725.27 km / 1000)
ARCSEC2_TO_CM2 = ARCSEC_TO_CM ** 2  # cm² per arcsec²  ≈ 5.26e15 cm²
ARCSEC2_TO_MM2 = ARCSEC_TO_MM ** 2  # Mm² per arcsec²  ≈ 0.526 Mm²


def compute_magnetic_features(map_obj, Blim=30, area_threshold=16):
    """
    Compute all magnetic features from a LOS magnetogram cutout for flare prediction.
    ...
    (docstring already in English — unchanged)
    """

    nan_result = {
        'tot_flux':         np.nan,
        'tot_us_flux':      np.nan,
        'schrijver_R':      np.nan,
        'schrijver_R_log':  np.nan,
        'wlsg':             np.nan,
        'grad_mean':        np.nan,
        'grad_max':         np.nan,
        'bz_skew':          np.nan,
        'bz_kurt':          np.nan,
    }

    try:
        data = map_obj.data.astype(float)

        # ------------------------------------------------------------------ #
        # Pixel scale in various units
        # map_obj.scale → arcsec/pixel, product → arcsec²/pixel
        # ------------------------------------------------------------------ #
        pixel_area_arcsec2 = np.abs(map_obj.scale[0].value * map_obj.scale[1].value)
        if pixel_area_arcsec2 == 0 or not np.isfinite(pixel_area_arcsec2):
            pixel_area_arcsec2 = 1.0  # fallback without physical units
        pixel_area_cm2     = pixel_area_arcsec2 * ARCSEC2_TO_CM2   # cm²/pixel
        pixel_area_Mm2     = pixel_area_arcsec2 * ARCSEC2_TO_MM2   # Mm²/pixel
        pixel_scale_arcsec = np.sqrt(pixel_area_arcsec2)            # arcsec/pixel
        pixel_scale_Mm     = pixel_scale_arcsec * ARCSEC_TO_MM      # Mm/pixel

        # ------------------------------------------------------------------ #
        # 1. AR mask
        # ------------------------------------------------------------------ #
        ar_mask = makeBMask(data, Blim=Blim, area_threshold=area_threshold,
                            connectivity=2, dilationR=8).astype(bool)
        if ar_mask.sum() < 10:
            return nan_result

        # ------------------------------------------------------------------ #
        # 2. Total signed and unsigned flux  [Mx = G·cm²]
        # ------------------------------------------------------------------ #
        Bdata       = data * pixel_area_cm2
        tot_flux    = float(np.nansum(Bdata[ar_mask]))
        tot_us_flux = float(np.nansum(np.abs(Bdata[ar_mask])))

        # ------------------------------------------------------------------ #
        # 3. PIL mask — shared by Schrijver R and WLsg
        #    Threshold: 150 G  (Schrijver 2007, Falconer 2008)
        #    Dilation radius: 12 physical arcsec (consistent MDI/HMI)
        # ------------------------------------------------------------------ #
        R_BLIM               = 150.0   # G
        R_PHYS_RADIUS_ARCSEC = 12.0    # arcsec
        dil_radius_pix = max(1, int(np.round(R_PHYS_RADIUS_ARCSEC / pixel_scale_arcsec)))
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity

        pos_pil = data > R_BLIM
        neg_pil = data < -R_BLIM
        for _ in range(dil_radius_pix):
            pos_pil = ndimage.binary_dilation(pos_pil, structure=struct)
            neg_pil = ndimage.binary_dilation(neg_pil, structure=struct)
        pil_mask = pos_pil & neg_pil

        # ------------------------------------------------------------------ #
        # 4. Schrijver R
        #    schrijver_R     [Mx]           sum |B|·dA_cm²  → physical units
        #    schrijver_R_log [log10(G·Mm²)] uses dA_Mm² to reproduce the
        #                                   original Schrijver (2007) scale: range ~3–6
        # ------------------------------------------------------------------ #
        if pil_mask.sum() > 0:
            abs_B_pil       = np.nansum(np.abs(data[pil_mask]))   # G·pixel
            schrijver_R     = float(abs_B_pil * pixel_area_cm2)   # Mx
            flux_pil_GMm2   = abs_B_pil * pixel_area_Mm2          # G·Mm²
            schrijver_R_log = float(np.log10(flux_pil_GMm2)) if flux_pil_GMm2 > 0 else 0.0
        else:
            schrijver_R     = 0.0
            schrijver_R_log = 0.0   # quiescent AR, ML-safe

        # ------------------------------------------------------------------ #
        # 5. Gradient features in physical units  [G/Mm]
        #    Sobel output in G/pixel → divided by pixel_scale_Mm → G/Mm
        #    This way MDI and HMI are directly comparable
        # ------------------------------------------------------------------ #
        grad_x   = ndimage.sobel(data, axis=1)
        grad_y   = ndimage.sobel(data, axis=0)
        grad_mag = np.hypot(grad_x, grad_y)   # G/pixel

        grad_mean = float(np.nanmean(grad_mag[ar_mask])) / pixel_scale_Mm   # G/Mm
        grad_max  = float(np.nanmax(grad_mag[ar_mask]))  / pixel_scale_Mm   # G/Mm

        # ------------------------------------------------------------------ #
        # 6. WLsg in physical units  [G·Mm]
        #    grad_mag in G/pixel, pixel_scale_Mm in Mm/pixel
        #    → product: G/pixel * Mm/pixel * n_pixel = G·Mm  ✓
        # ------------------------------------------------------------------ #
        if pil_mask.sum() > 0:
            wlsg = float(np.nansum(grad_mag[pil_mask]) * pixel_scale_Mm)   # G·Mm
        else:
            wlsg = 0.0

        # ------------------------------------------------------------------ #
        # 7. Moments of the B_LOS distribution  [dimensionless]
        # ------------------------------------------------------------------ #
        bz_ar   = data[ar_mask]
        bz_skew = float(skew(bz_ar, nan_policy='omit'))
        bz_kurt = float(kurtosis(bz_ar, fisher=True, nan_policy='omit'))

        return {
            'tot_flux':        tot_flux,
            'tot_us_flux':     tot_us_flux,
            'schrijver_R':     schrijver_R,
            'schrijver_R_log': schrijver_R_log,
            'wlsg':            wlsg,
            'grad_mean':       grad_mean,
            'grad_max':        grad_max,
            'bz_skew':         bz_skew,
            'bz_kurt':         bz_kurt,
        }

    except Exception as e:
        print(f"Warning in compute_magnetic_features: {e}")
        return nan_result



def calibrate_img(img, dataset):
    """
    Calibrate and smooth magnetogram image based on instrument
    
    SIMPLIFIED FOR HMI/MDI ONLY

    Parameters:
        img:            magnetogram data extracted from fits
        dataset (str):  which magnetogram instrument (HMI or MDI)
    
    Returns:
        img:            calibrated magnetogram data
    """
    if dataset == 'MDI':
        img = img / 1.3
        kernel = Gaussian2DKernel(1)
        img = convolve(img, kernel)
    elif dataset == 'HMI':
        kernel = Gaussian2DKernel(4)
        img = convolve(img, kernel)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Only HMI and MDI are supported.")
    
    return img


def extract_fits(data_fits, dataset):
    """
    Extract image and header from fits data according to magnetogram dataset
    
    SIMPLIFIED FOR HMI/MDI ONLY

    Parameters:
        data_fits:      opened fits file
        dataset (str):  which type of magnetogram (HMI or MDI)
    
    Returns:
        img:        observations from fits file
        header:     header dictionary from fits file
    """
    if dataset in ['HMI', 'MDI']:
        if len(data_fits) > 1 and data_fits[1].data is not None:
            hdu = data_fits[1]
        else:
            hdu = data_fits[0]
        
        if hdu.data is None:
            raise ValueError(f"No data found in HDU for {dataset}")
        
        img    = hdu.data
        header = hdu.header
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Only HMI and MDI are supported.")
    
    return img, header