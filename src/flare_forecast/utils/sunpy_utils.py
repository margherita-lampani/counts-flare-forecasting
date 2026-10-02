"""
Solar magnetogram utilities based on SunPy coordinate transformations.

This module provides low-level routines for computing physically meaningful
geometric and magnetic-field quantities from solar magnetograms. The
functions are used during preprocessing and feature extraction to estimate
photospheric areas and identify magnetically active regions.

Implemented utilities:
    - Pixel-area estimation from solar coordinate transformations
    - Strong-field region masking
    - Morphological filtering of magnetic structures
    - Active-region area selection based on magnetic-field thresholds

Main functions:
    mapPixelArea()
        Computes the photospheric surface area represented by each pixel
        using SunPy coordinate transformations and solar geometry.

    makeBMask()
        Generates a mask of magnetically significant regions by applying
        field-strength thresholds together with morphological filtering
        and spatial dilation.

These utilities provide the geometric and magnetic masks required for
the computation of flux-based and morphology-based predictors used in
solar flare forecasting.
"""

import numpy as np
import astropy.units as u
from skimage.morphology import dilation, area_opening, disk
from sunpy.coordinates.frames import HeliographicStonyhurst
from sunpy.map import Map


def mapPixelArea(map):
    """Function to calculate each pixel's photospheric area for a given map. 

    Parameters
    ----------
    map : sunpy map
        Map containing observations

    Returns
    -------
    area : sunpy map
        Map containing the area of each pixel.
    """
    
    x, y = np.meshgrid(*[np.arange(v) for v in map.data.shape])* u.pixel

    # Calculate position of three of the pixel's corners
    a = map.pixel_to_world(x+0.5*u.pixel, y+0.5*u.pixel).transform_to(HeliographicStonyhurst)
    b = map.pixel_to_world(x+0.5*u.pixel, y-0.5*u.pixel).transform_to(HeliographicStonyhurst)
    c = map.pixel_to_world(x-0.5*u.pixel, y-0.5*u.pixel).transform_to(HeliographicStonyhurst)
    d = map.pixel_to_world(x-0.5*u.pixel, y+0.5*u.pixel).transform_to(HeliographicStonyhurst)

    ## Stacking latitudes and longitudes

    lats = np.concatenate((a.data.lat[:,:,None], b.data.lat[:,:,None], c.data.lat[:,:,None], d.data.lat[:,:,None], a.data.lat[:,:,None]), axis=2)
    lons = np.concatenate((a.data.lon[:,:,None], b.data.lon[:,:,None], c.data.lon[:,:,None], d.data.lon[:,:,None], a.data.lon[:,:,None]), axis=2)


    # Get colatitude (a measure of surface distance as an angle)
    an = np.sin(lats/2)**2 + np.cos(lats)* np.sin(lons/2)**2
    colat = 2*np.arctan2( np.sqrt(an), np.sqrt(1-an) )

    #azimuth of each point in segment from the arbitrary origin
    az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2*np.pi*u.rad)

    # Calculate step sizes
    daz = np.diff(az, axis=2)
    daz = (daz + np.pi*u.rad) % (2 * np.pi*u.rad) - np.pi*u.rad

    # Determine average surface distance for each step
    deltas=np.diff(colat, axis=2)/2
    colat=colat[:,:,0:-1]+deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1-np.cos(colat)) * daz

    # Integrate and save the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a factor of 4pi.
    area = np.abs(np.nansum(integrands, axis=2))/(4*np.pi*u.rad) # Could be area of inside or outside
    area = np.concatenate((area[:,:,None], 1-area[:,:,None]), axis=2)
    area = np.min(area, axis=2)

    # Convert fractional area to real area
    # # Solar Radius
    rsun = (map.meta['RSUN_REF']*u.m).to(u.km)

    area = 4*np.pi*(area*rsun*rsun).to(u.Mm*u.Mm)
    return Map(area, map.meta)



def makeBMask(data, 
            Blim=30, 
            area_threshold=128,
            connectivity=2,
            dilationR=8):
    """Function to greate mask surrounding strong fields.

    Parameters
    ----------
    data : numpy array
        data used to create the mask
    Blim : float
        Magnetic field theshold used to determine the mask kernels
    area_threshold : int
        area_threshold passed to area_opening operation
    connectivity : int
        (1) for using only immediate neighbors (vertical and horizontal).
        (2) for using also diagonals
    dilationR : 8
        Radius of dilation disk

    Returns
    -------
    BMask : sunpy map
        Map containing the Bmask.
    """

    fieldMask = np.abs(data)>Blim           
    step1 = area_opening(fieldMask, area_threshold=area_threshold, connectivity=connectivity)
    footprint = disk(dilationR)
    mask = dilation(step1, footprint).astype(float)

    return mask
