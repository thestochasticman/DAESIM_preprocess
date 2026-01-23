#!/usr/bin/env python
# The shebang above allows the script to be added to the python path

# Catalog of OzWald daily variables is here: https://thredds.nci.org.au/thredds/catalog/ub8/au/OzWALD/daily/meteo/catalog.html

# +
# Standard Libraries
import os
import argparse
import json
from pathlib import Path

# Dependencies
import requests
import xarray as xr
import numpy as np

# +
ozwald_daily_abbreviations = {
    "Pg" : "Gross precipitation",  # 4km grid
    "Tmax" : "Maximum temperature",  # 500m grid
    "Tmin" : "Minimum temperature",  # 500m grid
    "Uavg" : "Average 24h windspeed",  # 5km grid
    "Ueff" : "Effective daytime windspeed",  # 5km grid
    "VPeff" : "Volume of effective rainfall",  # 5km grid
    "kTavg" : "Coefficient to calculate mean screen level temperature",  # 5km grid
    "kTeff" : "Coefficient to calculate effective screen level temperature"  # 5km grid
}


def ozwald_daily_singleyear_thredds(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021", stub="TEST", tmp_dir="scratch_dir", verbose=True):
    
    north = latitude + buffer 
    south = latitude - buffer 
    west = longitude - buffer
    east = longitude + buffer
    
    time_start = f"{year}-01-01"
    time_end = f"{year}-12-31"
    
    base_url = "https://thredds.nci.org.au"
    prefix = ".daily" if var == "Pg" else ""
    url = f'{base_url}/thredds/ncss/grid/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc?var={var}&north={north}&west={west}&east={east}&south={south}&time_start={time_start}&time_end={time_end}' 

    # Check the file exists before downloading it
    head_response = requests.head(url)
    if head_response.status_code == 200:
        response = requests.get(url)
        filename = os.path.join(tmp_dir, f"{stub}_{var}_{year}.nc")
        with open(filename, 'wb') as f:
            f.write(response.content)
        if verbose:
            print("Downloaded", filename)
        ds = xr.open_dataset(filename)
    else:
        return None

    return ds


def ozwald_daily_singleyear_gdata(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, year="2021"):
    
    prefix = ".daily" if var == "Pg" else ""
    filename = os.path.join(f"/g/data/ub8/au/OzWALD/daily/meteo/{var}/OzWALD{prefix}.{var}.{year}.nc")

    # OzWald doesn't have 2025 data in this folder yet as of 22/05/2025
    if not os.path.exists(filename):
        return None
        
    ds = xr.open_dataset(filename)
    
    bbox = [longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer]
    ds_region = ds.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))
    
    if buffer < 0.03:
        # Find a single point but keep the lat and lon dimensions for consistency
        ds_region = ds.sel(latitude=[latitude], longitude=[longitude], method='nearest')
    
    return ds_region


def ozwald_daily_multiyear(var="VPeff", latitude=-34.3890427, longitude=148.469499, buffer=0.1, years=["2020", "2021"], stub="TEST", tmpdir=".", thredds=True, verbose=True):
    dss = []
    for year in years:
        if thredds:
            ds_year = ozwald_daily_singleyear_thredds(var, latitude, longitude, buffer, year, stub, tmpdir, verbose=verbose)
        else:
            ds_year = ozwald_daily_singleyear_gdata(var, latitude, longitude, buffer, year)
        if ds_year:
            dss.append(ds_year)
    ds_concat = xr.concat(dss, dim='time')
    return ds_concat


def save_ozwald_daily_json(ds, outdir, stub, start_year, end_year, buffer, reducer='median', verbose=True):
    """
    Save ozwald daily data as JSON for frontend consumption

    Parameters
    ----------
        ds: xarray dataset with ozwald daily variables
        outdir: Directory to save the JSON file
        stub: Filename prefix
        start_year, end_year: Year range for metadata
        buffer: Spatial buffer used for metadata
        reducer: 'median', 'mean', 'min', 'max' - how to aggregate spatial data
        verbose: Print output messages

    Returns
    -------
        json_path: Path to the saved JSON file
    """
    # Aggregate spatially using the specified reducer
    if reducer == 'median':
        ds_point = ds.median(dim=['latitude', 'longitude'])
    elif reducer == 'mean':
        ds_point = ds.mean(dim=['latitude', 'longitude'])
    elif reducer == 'min':
        ds_point = ds.min(dim=['latitude', 'longitude'])
    elif reducer == 'max':
        ds_point = ds.max(dim=['latitude', 'longitude'])
    else:
        ds_point = ds.median(dim=['latitude', 'longitude'])  # default to median

    # Convert to records format
    data = []
    for i, time_val in enumerate(ds_point.time.values):
        row = {"time": str(time_val)[:10]}  # "YYYY-MM-DD"
        for var in ds_point.data_vars:
            val = float(ds_point[var].isel(time=i).values)
            row[var] = None if np.isnan(val) else round(val, 2)
        data.append(row)

    # Create payload with metadata
    payload = {
        "meta": {
            "start_year": start_year,
            "end_year": end_year,
            "buffer": buffer,
            "reducer": reducer,
            "variables": list(ds.data_vars.keys())
        },
        "data": data
    }

    # Save to JSON
    json_path = Path(outdir) / f"{stub}_ozwald_daily.json"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    if verbose:
        print(f"Saved JSON with {len(data)} records and {len(payload['meta']['variables'])} variables: {json_path}")

    return json_path


def ozwald_daily(variables=["VPeff", "Uavg"], lat=-34.3890427, lon=148.469499, buffer=0.1, start_year="2020", end_year="2021", outdir=".", stub="TEST", tmpdir=".", thredds=True, save_netcdf=True, save_json=True, plot=True, reducer='median', verbose=True):
    """Download daily variables from OzWald at varying resolutions for the region/time of interest

    Parameters
    ----------
        variables: See ozwald_daily_abbreviations at the top of this file for a complete list & resolutions
        lat, lon: Coordinates in WGS 84 (EPSG:4326)
        buffer: Distance in degrees in a single direction. e.g. 0.01 degrees is ~1km so would give a ~2kmx2km area.
        start_year, end_year: Inclusive, so setting both to 2020 would give data for the full year.
        outdir: The directory that the final .NetCDF gets saved. The filename includes the first variable in the csv.
        stub: The name to be prepended to each file download.
        tmpdir: The directory that the temporary NetCDFs get saved when downloading from Thredds. This does not get used if Thredds=False.
        thredds: A boolean flag to choose between using the public facing API (slower but works locally), or running directly on NCI (requires access to the ub8 project)
        save_netcdf: Save the data as NetCDF file
        save_json: Save the data as JSON for frontend consumption
        reducer: How to spatially aggregate data for JSON export ('median', 'mean', 'min', 'max')

    Returns
    -------
        ds_concat: an xarray containing the requested variables in the region of interest for the time period specified
        A NetCDF file of this xarray gets downloaded to outdir/(stub)_ozwald_daily_(first_variable).nc'
        A JSON file gets saved to outdir/(stub)_ozwald_daily.json if save_json=True
    """
    if verbose:
        print(f"Starting ozwald_daily")

    dss = []
    years = [str(year) for year in list(range(int(start_year), int(end_year) + 1))]
    for variable in variables:
        ds_variable = ozwald_daily_multiyear(variable, lat, lon, buffer, years, stub, tmpdir, thredds, verbose=verbose)
        dss.append(ds_variable)
    ds_concat = xr.merge(dss)

    if save_netcdf:
        filename = os.path.join(outdir, f'{stub}_ozwald_daily_{variables[0]}.nc')
        ds_concat.to_netcdf(filename, engine='h5netcdf')
        if verbose:
            print("Saved:", filename)

    if save_json:
        save_ozwald_daily_json(ds_concat, outdir, stub, start_year, end_year, buffer, reducer, verbose)

    if plot:
        import matplotlib.pyplot as plt
        variables = list(ds_concat.data_vars)
        figsize = (10, 2 * len(variables))
        ds_point = ds_concat.median(dim=['latitude', 'longitude'])
        fig, axes = plt.subplots(nrows=len(variables), figsize=figsize, sharex=True)
        if len(variables) == 1: 
            axes = [axes]
        for ax, var in zip(axes, variables):
            ds_point[var].plot(ax=ax, add_legend=False)
            ax.set_xlabel("")
        filename = os.path.join(outdir, f'{stub}_ozwald_daily_{variables[0]}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        if verbose:
            print("Saved:", filename)

    return ds_concat


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description="""Download daily variables from OzWald at varying resolutions (depending on the variable) for the region/time of interest""")
    
    parser.add_argument('--variable', default="VPeff", help=f"Default is 'VPeff', and options are: {list(ozwald_daily_abbreviations.keys())}")
    parser.add_argument('--lat', default='-34.389', help='Latitude in EPSG:4326 (default: -34.389)')
    parser.add_argument('--lon', default='148.469', help='Longitude in EPSG:4326 (default: 148.469)')
    parser.add_argument('--buffer', default='0.1', help='Buffer in each direction in degrees (default is 0.1, or about 20kmx20km)')
    parser.add_argument('--start_year', default='2020', help='Inclusive, and the minimum start year is 2000. Setting the start and end year to the same value will get all data for that year.')
    parser.add_argument('--end_year', default='2021', help='Specifying a larger end_year than available will automatically give data up to the most recent date (currently 2024)')
    parser.add_argument('--outdir', default='.', help='Directory for the output NetCDF file (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--tmpdir', default='.', help='The directory that the temporary NetCDFs get saved when downloading from Thredds. This does not get used if Thredds=False. (default is the current directory)')
    parser.add_argument('--nci', default=False, action="store_true", help='A boolean flag to choose between using the public facing Thredds API (slower but works locally), or running directly on NCI (requires access to the ub8 project). (default is to use Thredds)')
    parser.add_argument('--plot', default=False, action="store_true", help='Boolean flag to generate a time series plot of the downloaded variables (default: False)')
    return parser.parse_args()
# -

# +
if __name__ == '__main__':
    
    args = parse_arguments()
    
    variable = args.variable
    lat = float(args.lat)
    lon = float(args.lon)
    buffer = float(args.buffer)
    start_year = args.start_year
    end_year = args.end_year
    outdir = args.outdir
    stub = args.stub
    tmpdir = args.tmpdir
    thredds = not args.nci
    plot = args.plot
    
    ozwald_daily([variable], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds=thredds, plot=plot)
    