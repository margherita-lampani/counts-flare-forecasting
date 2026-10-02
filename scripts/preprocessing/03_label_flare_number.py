"""
Label active-region magnetogram observations with future flare activity.

The script queries the Heliophysics Event Knowledgebase (HEK) and retrieves
only flare events reported by the NOAA Space Weather Prediction Center (SWPC).
For each magnetogram observation, flare occurrences within a configurable
forecast window (24 h, 48 h, or 72 h) are identified and converted into
supervised learning labels.

Generated labels:
    C_count
    M_count
    X_count

Additional metadata:
    C_peak_times, M_peak_times, X_peak_times
    C_hours_from_obs, M_hours_from_obs, X_hours_from_obs
    C_details, M_details, X_details

The script performs one HEK query per unique active region, parallelizes
requests using multiple workers, and caches results locally to minimize
repeated network access.

Output:
    CSV file containing magnetogram features together with flare-count
    and flare-timing labels for supervised forecasting experiments.
"""

import csv
import pickle
import os
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from sunpy.net import hek
from sunpy.net import attrs as a

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLARE_CLASS_FLUX = {
    'A': 1e-8,
    'B': 1e-7,
    'C': 1e-6,
    'M': 1e-5,
    'X': 1e-4,
}


# ---------------------------------------------------------------------------
# Flux helpers
# ---------------------------------------------------------------------------

def class_to_flux(flare_class: str) -> float:
    """Convert a GOES flare class string (e.g. 'C4.5') to flux in W/m^2."""
    if not flare_class or pd.isna(flare_class):
        return 0.0
    s = str(flare_class).strip().upper()
    if not s:
        return 0.0
    try:
        letter = s[0]
        if letter not in FLARE_CLASS_FLUX:
            return 0.0
        numeric = float(s[1:]) if len(s) > 1 else 1.0
        return FLARE_CLASS_FLUX[letter] * numeric
    except (ValueError, IndexError):
        return 0.0


def get_class_letter(flare_class: str) -> Optional[str]:
    """Return the GOES class letter (A/B/C/M/X), or None if unparseable."""
    if not flare_class or pd.isna(flare_class):
        return None
    letter = str(flare_class).strip().upper()[:1]
    return letter if letter in FLARE_CLASS_FLUX else None


# ---------------------------------------------------------------------------
# HEK query
# ---------------------------------------------------------------------------

def query_swpc_flares(hek_client: hek.HEKClient,
                      ar_num: int,
                      t_start: Time,
                      t_end: Time,
                      verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Query HEK for flare events associated with ar_num, filtered to events
    catalogued by SWPC (FRM.Name == 'SWPC') and attributed to that specific AR
    (NOAANum == ar_num).

    No fallback to full-disk queries: if nothing is found for this AR,
    returns None (= no flares for this AR in this window).

    Returns:
        pd.DataFrame with at least columns ['fl_goescls', 'event_peaktime'],
        or None if nothing found.
    """
    try:
        result = hek_client.search(
            a.Time(t_start, t_end),
            a.hek.EventType('FL'),
            a.hek.AR.NOAANum == int(ar_num),
            a.hek.FRM.Name == 'SWPC'
        )
        if result is None or len(result) == 0:
            return None
        # Drop multidimensional columns (HEK quirk)
        names = [n for n in result.colnames if len(result[n].shape) <= 1]
        df = result[names].to_pandas()
        return df if not df.empty else None

    except Exception as e:
        if verbose:
            print(f"      [HEK] Query warning for AR {ar_num}: {e}")
        return None


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------
'''
def compute_labels(df_flares: Optional[pd.DataFrame],
                   obs_time: datetime,
                   window_hours: int) -> Dict:
    """
    Compute label columns from a HEK flare DataFrame.

    Filters to the [obs_time, obs_time + window_hours] window,
    then counts only C+ events (flux >= 1e-6 W/m^2).

    Returns a dict with all label keys set to 0 / 0.0 if no C+ flares found.
    """
    empty = {
        'C_count': 0,
        'M_count': 0,
        'X_count': 0,
        #'binary_C+': 0,
        #'binary_M+': 0,
        #'binary_X+': 0,
        #'total_count': 0,
        #'total_flux_integrated': 0.0,
        #f'xrsb_max_in_{window_hours}h': 0.0,
    }

    if df_flares is None or df_flares.empty:
        return empty
    if 'fl_goescls' not in df_flares.columns:
        return empty

    # Filter to the forecast time window
    if 'event_peaktime' in df_flares.columns:
        try:
            peak_times   = pd.to_datetime(df_flares['event_peaktime'], utc=True)
            window_start = pd.Timestamp(obs_time, tz='UTC')
            window_end   = pd.Timestamp(obs_time + timedelta(hours=window_hours), tz='UTC')
            mask   = (peak_times >= window_start) & (peak_times <= window_end)
            df_win = df_flares[mask].copy()
        except Exception:
            df_win = df_flares.copy()
    else:
        df_win = df_flares.copy()

    if df_win.empty:
        return empty

    # Keep only C+ flares
    cplus = []
    for cls in df_win['fl_goescls'].dropna():
        letter = get_class_letter(str(cls))
        flux   = class_to_flux(str(cls))
        if letter in ('C', 'M', 'X') and flux >= 1e-6:
            cplus.append((letter, flux))

    if not cplus:
        return empty

    C = sum(1 for lt, _ in cplus if lt == 'C')
    M = sum(1 for lt, _ in cplus if lt == 'M')
    X = sum(1 for lt, _ in cplus if lt == 'X')

    #binary_C_plus = 1 if C > 0 or M > 0 or X > 0 else 0
    #binary_M_plus = 1 if M > 0 or X > 0 else 0
    #binary_X_plus = 1 if X > 0 else 0

    return {
        'C_count':               C,
        'M_count':               M,
        'X_count':               X,
        #'binary_C+':            binary_C_plus,
        #'binary_M+':            binary_M_plus,
        #'binary_X+':            binary_X_plus,
        #'total_count':           C + M + X,
        #'total_flux_integrated': sum(fl for _, fl in cplus),
        #f'xrsb_max_in_{window_hours}h': max(fl for _, fl in cplus),
    }
'''
# ---------------------------------------------------------------------------
# Label computation  (MODIFIED)
# ---------------------------------------------------------------------------

def compute_labels(df_flares: Optional[pd.DataFrame],
                   obs_time: datetime,
                   window_hours: int) -> Dict:

    empty = {
        'C_count': 0, 'M_count': 0, 'X_count': 0,
        'C_peak_times': '', 'C_hours_from_obs': '', 'C_details': '',
        'M_peak_times': '', 'M_hours_from_obs': '', 'M_details': '',
        'X_peak_times': '', 'X_hours_from_obs': '', 'X_details': '',
    }

    if df_flares is None or df_flares.empty:
        return empty
    if 'fl_goescls' not in df_flares.columns:
        return empty

    # Filter to the forecast time window
    if 'event_peaktime' in df_flares.columns:
        try:
            peak_times   = pd.to_datetime(df_flares['event_peaktime'], utc=True)
            window_start = pd.Timestamp(obs_time, tz='UTC')
            window_end   = pd.Timestamp(obs_time + timedelta(hours=window_hours), tz='UTC')
            mask   = (peak_times >= window_start) & (peak_times <= window_end)
            df_win = df_flares[mask].copy()
            df_win['_peak_dt'] = peak_times[mask].values
        except Exception:
            df_win = df_flares.copy()
            df_win['_peak_dt'] = pd.NaT
    else:
        df_win = df_flares.copy()
        df_win['_peak_dt'] = pd.NaT

    if df_win.empty:
        return empty

    # Separate buckets per class
    buckets = {'C': [], 'M': [], 'X': []}
    obs_ts  = pd.Timestamp(obs_time, tz='UTC')

    for _, frow in df_win.iterrows():
        cls    = frow['fl_goescls']
        letter = get_class_letter(str(cls))
        flux   = class_to_flux(str(cls))
        if letter in ('C', 'M', 'X') and flux >= 1e-6:
            peak_dt = frow['_peak_dt']
            if pd.notna(peak_dt):
                pt      = pd.Timestamp(peak_dt, tz='UTC')
                time_str = pt.strftime('%Y-%m-%dT%H:%M:%S')
                delta_h  = f'{(pt - obs_ts).total_seconds() / 3600.0:.2f}'
            else:
                time_str = ''
                delta_h  = ''
            buckets[letter].append({
                'time':    time_str,
                'delta_h': delta_h,
                'detail':  str(cls).strip(),
            })

    def join_field(bucket, key):
        return '|'.join(f['key'] for f in bucket).replace("'key'", f"'{key}'") \
            if bucket else ''

    # Simpler and cleaner:
    def jf(bucket, key):
        return '|'.join(f[key] for f in bucket)

    return {
        'C_count':          len(buckets['C']),
        'M_count':          len(buckets['M']),
        'X_count':          len(buckets['X']),
        'C_peak_times':     jf(buckets['C'], 'time'),
        'C_hours_from_obs': jf(buckets['C'], 'delta_h'),
        'C_details':        jf(buckets['C'], 'detail'),
        'M_peak_times':     jf(buckets['M'], 'time'),
        'M_hours_from_obs': jf(buckets['M'], 'delta_h'),
        'M_details':        jf(buckets['M'], 'detail'),
        'X_peak_times':     jf(buckets['X'], 'time'),
        'X_hours_from_obs': jf(buckets['X'], 'delta_h'),
        'X_details':        jf(buckets['X'], 'detail'),
    }

# ---------------------------------------------------------------------------
# Main labeler
# ---------------------------------------------------------------------------

class ARFlareLabeler:
    """Labels AR data rows with SWPC-sourced flare counts and flux via HEK."""

    def __init__(self, input_file: str, output_file: str, window_hours: int,
                 verbose: bool = True, n_workers: int = 8):
        self.input_file   = input_file
        self.output_file  = output_file
        self.window_hours = window_hours
        self.verbose      = verbose
        self.n_workers    = n_workers
        self.hek_client   = hek.HEKClient()

        self.stats = {
            'total_rows':         0,
            'processed_ars':      0,
            'ars_with_flares':    0,
            'ars_without_flares': 0,
            'errors':             0,
        }

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def read_input_data(self) -> pd.DataFrame:
        self.log(f"Reading input file: {self.input_file}")
        df = pd.read_csv(self.input_file)
        self.stats['total_rows'] = len(df)
        self.log(f"  Total rows: {len(df)}")
        return df

    def get_observation_time(self, row: pd.Series) -> Optional[datetime]:
        """Extract observation timestamp. Prefers HMI, falls back to MDI."""
        for col in ('timestamp_HMI', 'timestamp_MDI'):
            if pd.notna(row.get(col)):
                try:
                    return pd.to_datetime(row[col]).to_pydatetime()
                except Exception:
                    pass
        try:
            return datetime.strptime(str(row['date']), '%Y%m%d')
        except Exception:
            return None

    def get_dataset_preference(self, row: pd.Series) -> Tuple[str, str, str]:
        """Return (dataset_name, filename, timestamp_str). Prefers HMI over MDI."""
        if pd.notna(row.get('filename_HMI')):
            return 'HMI', row['filename_HMI'], str(row.get('timestamp_HMI', ''))
        if pd.notna(row.get('filename_MDI')):
            return 'MDI', row['filename_MDI'], str(row.get('timestamp_MDI', ''))
        return 'Unknown', '', ''

    def get_magnetogram_features(self, row: pd.Series, dataset: str) -> Dict:
        """Extract magnetogram scalar features for the given dataset suffix."""
        features = {}
        for feat in ('tot_us_flux', 'tot_flux', 'datamin', 'datamax'):
            col = f'{feat}_{dataset}'
            features[feat] = row[col] if col in row.index else np.nan
        return features

    # ------------------------------------------------------------------
    # HEK prefetch  (one query per unique AR, parallelized)
    # ------------------------------------------------------------------

    def _build_ar_time_ranges(self, df: pd.DataFrame) -> Dict[int, Tuple[datetime, datetime]]:
        """
        For each unique AR number, compute the (min_time, max_time) range
        across all observations so that a single HEK query covers them all.
        """
        ar_time_ranges: Dict[int, Tuple[datetime, datetime]] = {}
        for ar_num, group in df.groupby('region_id'):
            times = [
                t for t in (self.get_observation_time(row) for _, row in group.iterrows())
                if t is not None
            ]
            if times:
                ar_time_ranges[int(ar_num)] = (min(times), max(times))
        return ar_time_ranges

    def _fetch_one_ar(self, ar_num: int,
                      t_min: datetime,
                      t_max: datetime) -> Tuple[int, Optional[pd.DataFrame]]:
        """Fetch all SWPC flares for a single AR over its full time range."""
        try:
            t_start = Time(t_min)
            t_end   = Time(t_max) + TimeDelta(self.window_hours * 3600, format='sec')
            df_fl   = query_swpc_flares(
                self.hek_client, ar_num, t_start, t_end, verbose=False
            )
            return ar_num, df_fl
        except Exception as e:
            self.log(f"  [WARNING] Failed to fetch AR {ar_num}: {e}")
            return ar_num, None

    def prefetch_flares_by_ar(self, df: pd.DataFrame) -> Dict[int, Optional[pd.DataFrame]]:
        """
        Query HEK once per unique AR number using parallel threads.
        Results are a dict mapping ar_num -> DataFrame (or None).

        This reduces total HEK calls from N_rows to N_unique_ars, and
        parallelism further cuts wall-clock time by ~n_workers.
        """
        ar_time_ranges = self._build_ar_time_ranges(df)
        n = len(ar_time_ranges)
        self.log(f"Pre-fetching {n} unique ARs "
                 f"({self.n_workers} parallel workers)...")

        cache: Dict[int, Optional[pd.DataFrame]] = {}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._fetch_one_ar, ar, t0, t1): ar
                for ar, (t0, t1) in ar_time_ranges.items()
            }
            for i, future in enumerate(as_completed(futures), 1):
                ar_num, df_flares = future.result()
                cache[ar_num] = df_flares
                if i % 100 == 0 or i == n:
                    found = sum(v is not None for v in cache.values())
                    self.log(f"  {i}/{n} ARs fetched  "
                             f"({found} with flares, {i - found} empty)")

        self.log(f"Pre-fetch complete.")
        return cache

    def load_or_build_cache(self, df: pd.DataFrame) -> Dict[int, Optional[pd.DataFrame]]:
        """
        Load the HEK cache from disk if it exists, otherwise build and save it.
        The cache file sits next to the output CSV with a .pkl extension.
        Re-running the script (e.g. for a different window) skips the fetch phase.
        """
        cache_path = os.path.splitext(self.output_file)[0] + '_hek_cache.pkl'

        if os.path.exists(cache_path):
            self.log(f"Loading HEK cache from {cache_path} ...")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.log(f"  Loaded {len(cache)} ARs from cache.")
            return cache

        cache = self.prefetch_flares_by_ar(df)

        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        self.log(f"HEK cache saved to {cache_path}")
        return cache

    # ------------------------------------------------------------------
    # Row processing
    # ------------------------------------------------------------------

    def process_row(self, row: pd.Series,
                    flare_cache: Optional[Dict] = None) -> Optional[Dict]:
        """
        Compute labels and assemble the output dict for a single input row.

        Uses flare_cache when available (preferred), otherwise falls back
        to a live HEK query.
        """
        ar_num      = int(row['region_id'])
        region_type = row['region_type']

        obs_time = self.get_observation_time(row)
        if obs_time is None:
            self.log(f"  WARNING: Could not parse timestamp for AR {ar_num}")
            return None

        dataset, filename, timestamp_str = self.get_dataset_preference(row)

        self.log(f"  Processing AR {ar_num} ({region_type}) "
                 f"at {obs_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Retrieve flares from cache or live query
        if flare_cache is not None:
            df_flares = flare_cache.get(ar_num)
        else:
            t_start   = Time(obs_time)
            t_end     = t_start + TimeDelta(self.window_hours * 3600, format='sec')
            df_flares = query_swpc_flares(
                self.hek_client, ar_num, t_start, t_end, self.verbose
            )

        labels = compute_labels(df_flares, obs_time, self.window_hours)

        # Skip rows with no C+ flares in the forecast window
        '''
        if labels['total_count'] == 0:
            self.log(f"    No C+ flares in {self.window_hours}h window — SKIPPING")
            self.stats['ars_without_flares'] += 1
            return None
        '''
    
        # Skip rows where all flare fluxes summed to zero (data anomaly)
        '''
        if labels['total_flux_integrated'] == 0.0:
            self.log(f"    Zero integrated flux — SKIPPING")
            self.stats['ars_without_flares'] += 1
            return None
        '''

        self.stats['ars_with_flares'] += 1
        self.log(
            f"    C={labels['C_count']}  M={labels['M_count']}  "
            f"X={labels['X_count']} " # Total={labels['total_count']}  
            #f"IntFlux={labels['total_flux_integrated']:.2e}  "
            #f"MaxFlux={labels[f'xrsb_max_in_{self.window_hours}h']:.2e} W/m²"
        )

        mag = self.get_magnetogram_features(row, dataset)
        
        # Skip rows where the magnetogram image is blank (corrupted file)
        if mag.get('tot_us_flux', 0.0) == 0.0:
            self.log(f"    Zero magnetogram flux (corrupted image) — SKIPPING")
            self.stats['errors'] += 1
            return None
        '''
        return {
            'filename':              filename,
            'sample_time':           timestamp_str,
            'dataset':               dataset,
            'ar_number':             ar_num,
            'region_type':           region_type,
            'tot_us_flux':           mag.get('tot_us_flux', np.nan),
            'tot_flux':              mag.get('tot_flux', np.nan),
            'datamin':               mag.get('datamin', np.nan),
            'datamax':               mag.get('datamax', np.nan),
            'C_count':               labels['C_count'],
            'M_count':               labels['M_count'],
            'X_count':               labels['X_count'],
            #'binary_C+':            labels['binary_C+'],
            #'binary_M+':            labels['binary_M+'],
            #'binary_X+':            labels['binary_X+'],
            #'total_count':           labels['total_count'],
            #'total_flux_integrated': labels['total_flux_integrated'],
            #f'xrsb_max_in_{self.window_hours}h': labels[f'xrsb_max_in_{self.window_hours}h'],
        }
        '''
        return {
            'filename':              filename,
            'sample_time':           timestamp_str,
            'dataset':               dataset,
            'ar_number':             ar_num,
            'region_type':           region_type,
            'tot_us_flux':           mag.get('tot_us_flux', np.nan),
            'tot_flux':              mag.get('tot_flux', np.nan),
            'datamin':               mag.get('datamin', np.nan),
            'datamax':               mag.get('datamax', np.nan),
            'C_count':               labels['C_count'],
            'M_count':               labels['M_count'],
            'X_count':               labels['X_count'],
            'C_peak_times':     labels['C_peak_times'],
            'C_hours_from_obs': labels['C_hours_from_obs'],
            'C_details':        labels['C_details'],
            'M_peak_times':     labels['M_peak_times'],
            'M_hours_from_obs': labels['M_hours_from_obs'],
            'M_details':        labels['M_details'],
            'X_peak_times':     labels['X_peak_times'],
            'X_hours_from_obs': labels['X_hours_from_obs'],
            'X_details':        labels['X_details'],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    '''
    def get_fieldnames(self) -> List[str]:
        return [
            'filename', 'sample_time', 'dataset', 'ar_number', 'region_type',
            'tot_us_flux', 'tot_flux', 'datamin', 'datamax',
            'C_count', 'M_count', 'X_count',
            #, 'binary_C+', 'binary_M+', 'binary_X+'
            #'total_count',
            #'total_flux_integrated',
            #f'xrsb_max_in_{self.window_hours}h',
        ]
    '''
    def get_fieldnames(self) -> List[str]:
            return [
                'filename', 'sample_time', 'dataset', 'ar_number', 'region_type',
                'tot_us_flux', 'tot_flux', 'datamin', 'datamax',
                'C_count', 'M_count', 'X_count',
                'C_peak_times', 'C_hours_from_obs', 'C_details',
                'M_peak_times', 'M_hours_from_obs', 'M_details',
                'X_peak_times', 'X_hours_from_obs', 'X_details',
            ]
    
    def process_all(self):
        self.log("\n" + "=" * 80)
        self.log("AR FLARE LABELING — HEK/SWPC")
        self.log("=" * 80)
        self.log(f"Input file:       {self.input_file}")
        self.log(f"Output file:      {self.output_file}")
        self.log(f"Forecast window:  {self.window_hours}h")
        self.log(f"HEK workers:      {self.n_workers}")
        self.log(f"Data source:      HEK filtered to FRM.Name == 'SWPC'")
        self.log(f"Labels:           C_count, M_count, X_count, binary_C+, binary_M+, binary_X+") #total_count,
        #self.log(f"                  total_flux_integrated, xrsb_max_in_{self.window_hours}h")
        self.log(f"Filters:          C+ flares present | flux > 0 | valid magnetogram")
        self.log("=" * 80 + "\n")

        df = self.read_input_data()
        if df.empty:
            self.log("No data found in input file!")
            return

        # Build / load the per-AR HEK cache before entering the row loop
        flare_cache = self.load_or_build_cache(df)

        fieldnames = self.get_fieldnames()

        # Write CSV header
        with open(self.output_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        # Process all rows, flushing after each write for safe resumption
        self.log("Processing rows...\n")
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for idx, row in df.iterrows():
                try:
                    out = self.process_row(row, flare_cache)
                    if out is not None:
                        writer.writerow(out)
                        f.flush()
                        self.stats['processed_ars'] += 1
                except Exception as e:
                    self.log(f"  ERROR processing row {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['errors'] += 1
                self.log("")

        self.print_summary()

    def print_summary(self):
        self.log("\n" + "=" * 80)
        self.log("PROCESSING SUMMARY")
        self.log("=" * 80)
        self.log(f"Total input rows:            {self.stats['total_rows']}")
        self.log(f"ARs with C+ flares (kept):   {self.stats['ars_with_flares']}")
        self.log(f"ARs without flares (skip):   {self.stats['ars_without_flares']}")
        self.log(f"Successfully written:         {self.stats['processed_ars']}")
        self.log(f"Errors:                       {self.stats['errors']}")
        self.log("=" * 80)
        self.log(f"\nOutput saved to: {self.output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Label AR data with SWPC flare counts and flux via HEK (optimized).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python ARCAFF_label_dataset_v3.py 24 input.csv output_24h.csv
  python ARCAFF_label_dataset_v3.py 48 input.csv output_48h.csv
  python ARCAFF_label_dataset_v3.py 72 input.csv output_72h.csv

Output columns:
  filename, sample_time, dataset, ar_number, region_type
  tot_us_flux, tot_flux, datamin, datamax
  C_count                number of C-class flares in window  (C+ only)
  M_count                number of M-class flares in window
  X_count                number of X-class flares in window
  total_count            C + M + X total
  total_flux_integrated  sum of peak fluxes [W/m^2] for all C+ flares
  xrsb_max_in_Xh         maximum peak flux [W/m^2] in window

Note: Only entries WITH at least one C+ flare in the window are written.
        """
    )
    parser.add_argument('window', type=int, choices=[24, 48, 72],
                        help='Forecast window in hours')
    parser.add_argument('input_file', type=str,
                        help='Input CSV file with AR data')
    parser.add_argument('output_file', type=str,
                        help='Output CSV file with labels')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help='Number of parallel HEK query workers (default: 8)')
    return parser.parse_args()


def main():
    args = parse_args()
    ARFlareLabeler(
        input_file=args.input_file,
        output_file=args.output_file,
        window_hours=args.window,
        verbose=not args.quiet,
        n_workers=args.workers,
    ).process_all()


if __name__ == '__main__':
    main()