"""
Index, clean, and standardize active-region magnetogram cutouts from MDI and HMI.

This script processes year-organized FITS magnetograms, applies quality-control
checks and instrument-specific calibration, computes magnetic flux properties,
converts observations to compressed HDF5 files, and generates metadata indices
for downstream machine-learning applications.

Main operations:
    - Validate FITS quality flags
    - Extract active-region identifiers from filenames
    - Calibrate MDI and HMI magnetograms
    - Compute total and unsigned magnetic flux
    - Save cleaned magnetograms as HDF5 files
    - Generate per-instrument metadata indices
    - Merge MDI and HMI indices by date and active region

Output:
    - Cleaned HDF5 magnetograms
    - Per-instrument index files (index_MDI.csv, index_HMI.csv)
    - Optional merged catalogue indexed by observation date and active region

Designed specifically for active-region cutouts and therefore does not
perform limb correction, reprojection, or image reorientation.
"""

import sys
import os

from astropy.io import fits
import csv
from datetime import datetime
import h5py
from pathlib import Path
from src.flare_forecast.utils.helper import *
from src.flare_forecast.utils.sunpy_utils import mapPixelArea
import argparse
from sunpy.map import Map
from multiprocessing import Pool
import pandas as pd
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def extract_region_info(filename):
    """
    Extract region type and identifier from filename
    
    Parameters:
        filename (str): name of fits file
    
    Returns:
        region_type (str): 'QS', 'I', or 'IA'
        region_id (str): identifier after the dash (e.g., '0', '7957', '9189')
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.fits', '').split('_')
    
    # Expected format: YYYYMMDD_HHMMSS_REGION-ID_INSTRUMENT
    # e.g., 19960421_002945_QS-0_MDI.fits or 20001019_235843_IA-9189_MDI.fits
    if len(parts) >= 3:
        region_full = parts[2]
        if '-' in region_full:
            region_type, region_id = region_full.split('-', 1)
            return region_type, region_id
    
    return None, None


class Indexer:
    def __init__(self, data: str, data_dir: str = 'Data-cut-flare-number-padded', 
                 save_dir: str = 'Data-cut-flare-number-padded/hdf5', 
                 index_dir: str = 'Data-cut-flare-number-padded', 
                 metadata_cols: list = [], 
                 nworkers: int = 4):
        """
        Initialize an indexing class to iterate through data, cleaning and indexing
        
        Parameters:
            data (str):             magnetogram instrument (HMI or MDI)
            data_dir (str):         root dir where the data is stored 
            save_dir (str):         dir for new hdf5 files to be saved
            index_dir (str):        location for generated index to be saved
            metadata_cols (list):   list of fits header keys to save 
        """
        if data not in ['HMI', 'MDI']:
            raise ValueError(f"Only HMI and MDI are supported, got {data}")
        
        self.data = data
        self.root_dir = Path(data_dir)
        self.new_dir = Path(save_dir) / self.data
        if not os.path.exists(self.new_dir):
            os.makedirs(self.new_dir, exist_ok=True)
        self.file = index_dir + '/index_' + data + '.csv'
        self.metadata_cols = metadata_cols
        self.nworkers = nworkers

        # header data for csv file
        header = ['filename', 'fits_file', 'timestamp', 'region_type', 'region_id']
        header.extend(metadata_cols)
        header.extend(['tot_us_flux', 'tot_flux', 'datamin', 'datamax'])
        #NEW EXTRA FEATURES
        #header.extend(['tot_flux', 'tot_us_flux', 'datamin', 'datamax',
        #               'schrijver_R', 'schrijver_R_log', 'wlsg', 'grad_mean', 'grad_max',
        #               'bz_skew', 'bz_kurt'])
        header = [key + '_' + data for key in header]
        header.insert(2, 'date')

        # write header to csv file
        with open(self.file, 'w') as out_file:
            out_writer = csv.writer(out_file, delimiter=',')
            out_writer.writerow(header)
    
    def index_data(self):
        """
        Clean and index data in root dir by year. Save index to csv file.
        
        Parameters:
            nworkers (int):     Number of processes to index years in parallel
        """
        # open csv file writer
        out_file = open(self.file, 'a')
        out_writer = csv.writer(out_file, delimiter=',')

        # iterate through files and add to index
        years = sorted(os.listdir(self.root_dir / self.data))
        args = [(year, False, False) for year in years]

        # index years in parallel and write results to csv
        self.error_files = []
        with Pool(self.nworkers) as pool:
            for result in pool.starmap(self.index_year, args):
                index = result[0]
                self.error_files.extend(result[1])
                out_writer.writerows(index) 

        out_file.close()
        print('Finished indexing', self.data)    
        print('Errors on:', self.error_files)

    def index_year(self, year, onefileperday=False, test=False):
        """
        Indexes fits files within a specified directory by writing metadata
        to a csv writer. Nothing will be written to the index if the file has 
        a bad quality flag.

        Parameters:
            year (str):         subdirectory containing files, sorted by year
            onefileperday (bool):   option to only index the first file from each day
            test (bool):        option to stop indexing after 5 files
        
        Returns:
            index:              list of index data for all files in year
            error_files:        list of any files which threw errors
        """
        n = 0
        index = []
        t0 = time.time()
        error_files = []

        if not os.path.isdir(self.root_dir / self.data / year):
            return index, error_files
        if not os.path.exists(self.new_dir / year):
            os.makedirs(self.new_dir / year, exist_ok=True)
        
        all_files = sorted(os.listdir(self.root_dir / self.data / year))
        total_files = len(all_files)
        print(f'Starting {self.data} {year}: {total_files} files to process')
        
        for idx, file in enumerate(all_files, 1):
            # extract date and time from filename
            date, timestamp = extract_date_time(file, self.data, year)

            if date is None:
                continue

            try:
                # open file
                with fits.open(self.root_dir / self.data / year / file, cache=False) as data_fits:
                    data_fits.verify('fix')
                    img, header = extract_fits(data_fits, self.data)           

                # index_file
                index_data = self.index_item(
                    self.root_dir / self.data / year / file,
                    img, header, date, timestamp, 
                    self.new_dir / year
                )
            except Exception as e:
                error_files.append(str(self.root_dir / self.data / year / file))
                print(f"Error processing {file}: {e}")
                continue

            if index_data is None:
                continue

            # add metadata to list
            index.append(index_data)
            n += 1
            
            # Progress update every 10%
            if idx % max(1, total_files // 10) == 0:
                print(f'{self.data} {year}: {idx}/{total_files} processed, {n} indexed')

            # cut off indexing after 5 files if testing
            if test and n >= 5:
                break

        t1 = time.time()
        print(n, 'files indexed for', self.data, year, 'in', round((t1 - t0) / 60, 2), 'minutes')
        
        return index, error_files
    
    def index_item(self, file, img, header, date, timestamp, new_dir): 
        # clean by checking quality flag
        if not check_quality(self.data, header):
            return None

        # Extract region info ONCE at the beginning
        region_type, region_id = extract_region_info(str(file))
        
        if region_type is None or region_id is None:
            print(f"Warning: Could not extract region info from {file}")
            return None

        # calibrate and smooth
        img = calibrate_img(img, self.data)

        # create sunpy map - NO REPROJECTION for AR cutouts
        map_obj = Map(img, header)
        
        # Calculate flux directly on the calibrated cutout
        tot_flux, tot_us_flux = compute_tot_flux(map_obj, Blim=30, area_threshold=16)
        #NEW EXTRA FEATURES
        #features = compute_magnetic_features(map_obj, Blim=30, area_threshold=16)

        #if tot_us_flux == 0.0:
        #    print(f"Warning: zero flux in {file}, skipping")
        #    return None

        #if features['tot_us_flux'] == 0.0 or np.isnan(features['tot_us_flux']):
        #    print(f"Warning: zero or NaN flux in {file}, skipping")
        #    return None

        # save new hdf5 file with region info in filename
        new_file = new_dir / (self.data + '_magnetogram.' + 
                        datetime.strftime(timestamp, '%Y%m%d_%H%M%S') + 
                        f'_{region_type}-{region_id}_TAI.h5')
        
        with h5py.File(new_file, 'w') as h5:
            h5.create_dataset('magnetogram', data=map_obj.data, compression='gzip')

        # store metadata
        index_data = [str(new_file), str(file), date, timestamp, region_type, region_id]
        index_data.extend([header[key] for key in self.metadata_cols])
        index_data.extend([tot_us_flux,tot_flux,np.nanmin(map_obj.data), np.nanmax(map_obj.data)])
        #index_data.extend([features['tot_flux'], features['tot_us_flux'],
        #                np.nanmin(map_obj.data), np.nanmax(map_obj.data),
        #                features['schrijver_R'], features['schrijver_R_log'], features['wlsg'], features['grad_mean'], features['grad_max'],
        #                features['bz_skew'], features['bz_kurt']])

        return index_data


def parse_args(args=None):
    """
    Parses command line arguments to script. Sets up argument for which 
    dataset to index.

    Parameters:
        args (list):    defaults to parsing any command line arguments
    
    Returns:
        parser args:    Namespace from argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        type=str,
                        nargs='*',
                        default=['MDI', 'HMI'],
                        help='dataset to index (MDI and/or HMI)')
    parser.add_argument('-r', '--root',
                        type=str,
                        default='Data-cut-flare-number-padded',
                        help='root directory for fits files')
    parser.add_argument('-n', '--newdir',
                        type=str,
                        default='Data-cut-flare-number-padded/hdf5',
                        help='directory to save cleaned hdf5 files')
    parser.add_argument('-i', '--indexdir',
                        type=str,
                        default='Data-cut-flare-number-padded',
                        help='directory to save index file')
    parser.add_argument('-w', '--workers',
                        type=int,
                        default=8,
                        help='number of cores for parallelization')
    return parser.parse_args(args)


def merge_indices_by_date_AR(root_dir, datasets):
    """
    Merge generated indices across datasets by date and active region
    
    Parameters:
        root_dir (str):     location of index files
        datasets (list):    names of data being indexed and merged
    
    Returns:
        df_merged (DataFrame): merged dataframe with common keys
    """
    dfs = []
    
    for data in datasets:
        data = data.upper()
        filename = Path(root_dir) / f'index_{data}.csv'
        df = pd.read_csv(filename)
        
        # Rename columns to standardize common ones
        df = df.rename(columns={
            f'region_type_{data}': 'region_type',
            f'region_id_{data}': 'region_id'
        })
        
        dfs.append(df)
    
    # Merge on common fields: date, region_type, region_id
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(
            df, 
            how='outer',
            on=['date', 'region_type', 'region_id'],
            suffixes=('', '_dup')
        )
    
    # Sort by date and region
    df_merged = df_merged.sort_values(['date', 'region_type', 'region_id']).reset_index(drop=True)
    
    # Reorder columns: put date, region_type, region_id first
    cols = df_merged.columns.tolist()
    key_cols = ['date', 'region_type', 'region_id']
    other_cols = [col for col in cols if col not in key_cols]
    df_merged = df_merged[key_cols + other_cols]
    
    print(f'{len(df_merged)} entries in merged index')
    print(f'Unique dates: {df_merged["date"].nunique()}')
    print(f'Unique regions: {df_merged["region_type"].nunique()} types')
    print('\nFirst few rows:')
    print(df_merged.head(10))
    
    return df_merged


def main():
    parser = parse_args()
    datasets = parser.data
    root_dir = parser.root
    new_dir = parser.newdir
    index_dir = parser.indexdir
    nworkers = parser.workers

    if not os.path.exists(new_dir):
        os.makedirs(new_dir, exist_ok=True)

    for data in datasets:
        data = data.upper()
        indexer = Indexer(data, root_dir, new_dir, index_dir, ['t_obs'], nworkers=nworkers)
        indexer.index_data()

    if len(datasets) > 1:
        df_merged = merge_indices_by_date_AR(index_dir, datasets)
        filename_merged = '_'.join([data for data in datasets])
        df_merged.to_csv(index_dir + '/index_' + filename_merged + '.csv', index=False)


if __name__ == '__main__':
    main()