"""
Organize raw solar magnetogram FITS files into a standardized directory structure.

The script reads FITS file paths from the region classification catalog,
extracts the instrument (MDI or HMI) and acquisition year from each filename,
and copies the files into instrument/year-specific directories. It also
provides dataset statistics, dry-run validation, and final integrity checks.

Output structure:
data/
├── MDI/
│   ├── 1996/
│   ├── 1997/
│   └── ...
└── HMI/
    ├── 2010/
    ├── 2011/
    └── ...
"""

import os
import shutil
import glob
from collections import defaultdict
import pandas as pd

# Configuration
CSV_PATH = os.environ.get("CSV_PATH", "data/region_classification_final.csv")
DEST_BASE = os.environ.get("DEST_BASE", "data")

def parse_filename(filename):
    """Extract instrument and year from filename"""
    basename = os.path.basename(filename)
    parts = basename.replace('.fits', '').split('_')
    
    # Extract date (YYYYMMDD format)
    date_str = parts[0] if len(parts) > 0 else None
    year = date_str[:4] if date_str and len(date_str) >= 4 else None
    
    # Extract instrument (MDI or HMI)
    if len(parts) > 0:
        instrument_full = parts[-1]  
        if instrument_full.startswith('MDI'):
            instrument = 'MDI'
        elif instrument_full.startswith('HMI'):
            instrument = 'HMI'
        else:
            instrument = None
    else:
        instrument = None
    
    return instrument, year

def remove_mag_from_filename(filename):
    """Remove '_mag' from filename"""
    return filename.replace('_mag', '')

def create_directory_structure(base_path, start_year=1996, end_year=2022):
    """Create directory structure for both instruments and all years"""
    instruments = ['MDI', 'HMI']
    
    for instrument in instruments:
        for year in range(start_year, end_year + 1):
            dir_path = os.path.join(base_path, instrument, str(year))
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def organize_files(csv_path, dest_base, dry_run=True):
    """Organize FITS magnetogram files into instrument/year structure"""
    
    # Load CSV and get file paths
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    fits_files = df['path_image_cutout'].tolist()
    
    print(f"Found {len(fits_files)} FITS files from CSV")
    
    # Group files by instrument and year
    file_groups = defaultdict(lambda: defaultdict(list))
    skipped = 0
    not_found = 0
    
    for filepath in fits_files:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            not_found += 1
            continue
            
        instrument, year = parse_filename(filepath)
        
        if instrument is None or year is None:
            print(f"Warning: Could not parse: {filepath}")
            skipped += 1
            continue
        
        file_groups[instrument][year].append(filepath)
    
    print(f"\nStatistics:")
    print(f"  Total files in CSV: {len(fits_files)}")
    print(f"  Files not found: {not_found}")
    print(f"  Files skipped (parse error): {skipped}")
    print(f"  Files to process: {len(fits_files) - not_found - skipped}")
    
    # Show distribution
    print("\nFile distribution:")
    for instrument in sorted(file_groups.keys()):
        print(f"\n{instrument}:")
        for year in sorted(file_groups[instrument].keys()):
            count = len(file_groups[instrument][year])
            print(f"  {year}: {count} files")
    
    if dry_run:
        print("\n=== DRY RUN - No files will be copied ===")
        return
    
    # Copy files
    print("\nCopying files...")
    total_copied = 0
    total_errors = 0
    
    for instrument in file_groups:
        for year in file_groups[instrument]:
            dest_dir = os.path.join(dest_base, instrument, year)
            files_to_copy = file_groups[instrument][year]
            
            for source_file in files_to_copy:
                original_basename = os.path.basename(source_file)
                new_basename = remove_mag_from_filename(original_basename)
                dest_file = os.path.join(dest_dir, new_basename)
                
                try:
                    shutil.copy2(source_file, dest_file)
                    total_copied += 1
                    if total_copied % 100 == 0:
                        print(f"  Copied {total_copied} files...")
                except Exception as e:
                    print(f"Error copying {original_basename}: {e}")
                    total_errors += 1
    
    print(f"\nCompleted!")
    print(f"  Successfully copied: {total_copied} files")
    print(f"  Errors: {total_errors} files")

def verify_structure(dest_base):
    """Verify the created directory structure and file counts"""
    print("\n=== VERIFYING STRUCTURE ===\n")
    
    for instrument in ['MDI', 'HMI']:
        inst_path = os.path.join(dest_base, instrument)
        if not os.path.exists(inst_path):
            print(f"Warning: {instrument} directory not found")
            continue
        
        print(f"{instrument}:")
        years = sorted([d for d in os.listdir(inst_path) 
                       if os.path.isdir(os.path.join(inst_path, d))])
        
        total_files = 0
        for year in years:
            year_path = os.path.join(inst_path, year)
            fits_files = [f for f in os.listdir(year_path) if f.endswith('.fits')]
            file_count = len(fits_files)
            total_files += file_count
            if file_count > 0:
                print(f"  {year}: {file_count} files")
        
        print(f"  Total: {total_files} files\n")

def main():
    """Main execution function"""
    print("FITS Magnetogram Organization Script")
    print("=" * 50)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file does not exist: {CSV_PATH}")
        return
    
    print(f"\nConfiguration:")
    print(f"  CSV file: {CSV_PATH}")
    print(f"  Destination base: {DEST_BASE}")
    
    # Create directory structure
    print(f"\nCreating directory structure...")
    create_directory_structure(DEST_BASE)
    
    # Run dry run
    print(f"\nRunning dry run...")
    organize_files(CSV_PATH, DEST_BASE, dry_run=True)
    
    # Ask for confirmation
    response = input("\nProceed with copying? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        organize_files(CSV_PATH, DEST_BASE, dry_run=False)
        verify_structure(DEST_BASE)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()