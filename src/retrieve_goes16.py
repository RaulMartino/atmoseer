import pandas as pd
import sys, getopt
from datetime import datetime
from util import is_posintstring
from globals import *
import s3fs
import numpy as np
import xarray as xr
import pandas as pd
import os

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

# Latitude e longitude dos municipios que o forte de copacabana pega
# Latitude: entre -22.717° e -23.083°
# Longitude: entre -43.733° e -42.933°
# estacoes = [{"nome": "copacabana",
#             "n_lat": -22.717,
#             "s_lat": -23.083,
#             'w_lon': -43.733,
#             'e_lon': -42.933}]

# Latitude e Longitude do RJ
def filter_coordinates(ds:xr.Dataset):
  """
    Filter lightning event data in an xarray Dataset based on latitude and longitude boundaries.

    Args:
        ds (xarray.Dataset): Dataset containing lightning event data with variables `event_energy`, `event_lat`, and `event_lon`.
    Returns:
        xarray.Dataset: A new dataset with the same variables as `ds`, but with lightning events outside of the specified latitude and longitude boundaries removed.
  """
  return ds['event_energy'].where(
      (ds['event_lat'] >= -22.9035) & (ds['event_lat'] <= -22.7469) &
      (ds['event_lon'] >= -43.7958) & (ds['event_lon'] <= -43.0962),
      drop=True)

# Download all files in parallel, and rename them the same name (without the directory structure)
def download_file(files):
    """
    Downloads a GOES-16 netCDF file from an S3 bucket, filters it for events that fall within a specified set of coordinates, 
    and saves the filtered file to disk.
    Args:
        file (str): A string representing the name of the file to be downloaded from an S3 bucket.
    """
    files_process = []
    count = 1
    for file in files:
        print(f"Reading file number {count}, remaining {len(files) - count} files")
        filename = file.split('/')[-1]
        fs.get(file, filename)
        ds = xr.open_dataset(filename)
        ds = filter_coordinates(ds)
        if ds.number_of_events.nbytes != 0:
            files_process.append(ds)
            # Break for debugger
            break
        os.remove(filename)
        count += 1

    if len(files_process) > 0:
        # concatenate datasets along the time dimension
        merged_ds = xr.concat(files_process, dim='time')

        # convert the merged dataset to a dataframe
        merged_df = merged_ds.to_dataframe()

        # save merged dataframe to a parquet file
        merged_df.to_parquet("atmoseer/data/goes16/merged_file.parquet")
    else:
        print("No data found within the specified coordinates and Date.")

def import_data(station_code, initial_year, final_year):
    """
    Downloads and saves GOES-16 data files from Amazon S3 for a given station code and time period.

    Args:
        station_code (str): The station code to download data for.
        initial_year (int): The initial year of the time period to download data for.
        final_year (int): The final year of the time period to download data for.

    Returns:
        None

    This function first reads a CSV file with relevant dates to download data for, then constructs a list of
    file paths for the requested station code and time period using these dates. The files are then downloaded
    using a thread pool executor for parallel processing.

    Note: This function assumes that the relevant data files are stored in the Amazon S3 bucket 'noaa-goes16'.
    """
    # Get files of GOES-16 data (multiband format) on multiple dates
    # format: <Product>/<Year>/<Day of Year>/<Hour>/<Filename>
    hours = [f'{h:02d}' for h in range(25)]  # Download all 24 hours of data

    start_date = pd.to_datetime(f'{initial_year}-01-01')
    end_date = pd.to_datetime(f'{final_year}-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    files = []
    for date in dates:
        year = str(date.year)
        day_of_year = f'{date.dayofyear:03d}'
        print(f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}')
        if day_of_year == '060':
            # Break for debugger
            break
        for hour in hours:
            target = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}/{hour}'
            files.extend(fs.ls(target))

    download_file(files)

def main(argv):
    station_code = ""

    start_goes_16 = 2017
    start_year = 2017
    end_year = datetime.now().year

    help_message = "{0} -s <station_id> -b <begin> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hs:b:e:t:", ["help", "station=", "begin=", "end="])
    except:
        print(help_message)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)
            sys.exit(2)
        # elif opt in ("-s", "--station"):
        #     station_code = arg
        #     if not ((station_code == "all") or (station_code in INMET_STATION_CODES_RJ)):
        #         print(help_message)
        #         sys.exit(2)
        elif opt in ("-b", "--begin"):
            if not is_posintstring(arg):
                sys.exit("Argument start_year must be an integer. Exit.")
            start_year = int(arg)
        elif opt in ("-e", "--end"):
            if not is_posintstring(arg):
                sys.exit("Argument end_year must be an integer. Exit.")
            end_year = int(arg)

    # assert (station_code is not None) and (station_code != '')
    assert (start_year <= end_year) and (start_year >= start_goes_16)

    station_code = 'copacabana'
    start_year = 2018
    end_year = 2018

    import_data(station_code, start_year, end_year)


if __name__ == "__main__":
    main(sys.argv)