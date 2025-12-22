#!/usr/bin/env python
# The shebang above allows the script to be added to the python path

# +
#  Pysheds documentation is here: https://mattbartos.com/pysheds/
import os
import argparse

# Dependencies
import numpy as np
from pysheds.grid import Grid
import rasterio
from rasterio.enums import Resampling
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

topographic_variables = ['accumulation', 'aspect', 'slope', 'twi']

# +
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
def pysheds_accumulation(terrain_tif):
    """Read in the grid and dem and calculate the water flow direction and accumulation"""

    # Load both the dem (basically a numpy array), and the grid (all the metadata like the extent)
    grid = Grid.from_raster(terrain_tif, nodata=np.float64(np.nan))  # Crucial to specify these nodata parameters with the correct dtype or else everything breaks in the latest version of numpy
    dem = grid.read_raster(terrain_tif, nodata=np.float64(np.nan))

    # Hydrologically enforce the DEM so water can flow downhill to the edge and not get stuck
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Calculate the aspect (fdir) and accumulation of water (acc)
    fdir = grid.flowdir(inflated_dem, nodata_out=np.int64(0))
    acc = grid.accumulation(fdir, nodata_out=np.int64(0))
        
    return grid, dem, fdir, acc


def calculate_slope(terrain_tif):
    """Calculate the slope of a DEM"""
    with rasterio.open(terrain_tif) as src:
        dem = src.read(1)  
        transform = src.transform 
    gradient_y, gradient_x = np.gradient(dem, transform[4], transform[0])
    slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)) * (180 / np.pi)
    return slope


def calculate_accumulation(terrain_tif):
    """Calculate the upstream area of each pixel using pysheds"""
    _, _, _, acc = pysheds_accumulation(terrain_tif)
    return acc


def calculate_TWI(acc, slope):
    """Calculate the topographic wetness index based on upstream area and local slope
        TWI = ln( accumulation / tan(slope) )
    """
    ratio_acc_slope = acc / np.tan(np.radians(slope))
    ratio_acc_slope[ratio_acc_slope <= 0] = 1     # Tried to avoid the division by 0 runtime warning, but don't think it worked
    twi = np.log(ratio_acc_slope)
    return twi


def add_numpy_band(ds, variable, array, affine, resampling_method):
    """Add a new band to the xarray from a numpy array and affine using the given resampling method"""
    da = xr.DataArray(
        array, 
        dims=["y", "x"], 
        attrs={
            "transform": affine,
            "crs": "EPSG:3857"
        }
    )
    da.rio.write_crs("EPSG:3857", inplace=True)
    reprojected = da.rio.reproject_match(ds, resampling=resampling_method)
    ds[variable] = reprojected
    return ds

def plot_topography(ds, outdir='.', stub='Test', verbose=True):
    """Create 4 side by side plots of elevation, accumulation, aspect, slope"""
    # Reproject to World Geodetic System
    ds = ds.rio.reproject("EPSG:4326")
    left, bottom, right, top = ds.rio.bounds()
    extent = (left, right, bottom, top)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes
    
    # ===== Elevation Plot =====
    dem = ds['terrain']
    im = ax1.imshow(dem, cmap='terrain', interpolation='bilinear', extent=extent)
    ax1.set_title("Elevation")
    plt.colorbar(im, ax=ax1, label='height above sea level (m)')
    
    # Contours
    interval = 10
    contour_levels = np.arange(np.floor(np.nanmin(dem)), np.ceil(np.nanmax(dem)), interval)
    contours = ax1.contour(dem, levels=contour_levels, colors='black',
                           linewidths=0.5, alpha=0.5, extent=extent, origin='upper')
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    # Scale bar
    scalebar = AnchoredSizeBar(ax1.transData, 0.01, '1km', loc='lower left', pad=0.1,
                               color='black', frameon=False, size_vertical=0.0001,
                               fontproperties=FontProperties(size=12))
    ax1.add_artist(scalebar)
    
    # North arrow
    ax1.annotate('N',
                 xy=(0.95, 0.1), xycoords='axes fraction',   # Arrow tip (higher position)
                 xytext=(0.95, 0.04),                         # Arrow base (lower position)
                 arrowprops=dict(facecolor='black', width=5, headwidth=12),
                 ha='center', va='center',
                 fontsize=12, fontweight='bold', color='black')
    
    # ===== Accumulation Plot =====
    acc = ds['accumulation']
    im = ax2.imshow(acc, cmap='cubehelix', norm=colors.LogNorm(1, np.nanmax(acc)),
                    interpolation='bilinear', extent=extent)
    ax2.set_title("Accumulation")
    plt.colorbar(im, ax=ax2, label='upstream cells')
    
    # ===== Aspect Plot =====
    arcgis_dirs = np.array([1, 2, 4, 8, 16, 32, 64, 128]) 
    sequential_dirs = np.array([1, 2, 3, 4, 5, 6, 7, 8]) 
    fdir = ds['aspect']
    fdir_equal_spacing = np.zeros_like(fdir)  
    for arcgis_dir, sequential_dir in zip(arcgis_dirs, sequential_dirs):
        fdir_equal_spacing[fdir == arcgis_dir] = sequential_dir 
    
    im = ax3.imshow(fdir_equal_spacing, cmap="twilight_shifted", origin="upper", extent=extent)
    ax3.set_title("Aspect")
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_ticks(sequential_dirs)
    cbar.set_ticklabels(["E", "SE", "S", "SW", "W", 'NW', "N", "NE"])
    
    # ===== Slope Plot =====
    slope = ds['slope']
    im = ax4.imshow(slope, cmap="YlGn", origin="upper", extent=extent)
    ax4.set_title("Slope")
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("degrees")
    
    # Add lat/lon labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    
    # Save combined figure
    plt.tight_layout()
    filepath = os.path.join(outdir, stub + "_topography.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    if verbose:
        print(f"Saved: {filepath}")

def topography(outdir=".", stub="TEST", smooth=True, sigma=5, ds=None, savetifs=True, verbose=True, plot=True):
    """Derive topographic variables from the elevation. 
    This function assumes there already exists a file named (outdir)/(stub)_terrain.tif"
    
    Parameters
    ----------
        outdir: The directory to save the topographic variables.
        stub: The name to be prepended to each file download.
        smooth: Boolean to determine whether to apply a gaussian filter to the elevation before deriving topographic variables. 
                This is necessary when using terrain tiles to remove artifacts from the elevation being stored as ints.
        sigma: smoothing parameter to use for the gaussian filter. Not used if smooth=False. 
        ds: The output of terrain_tiles so that you don't have to re-load the tif again.
        save_tifs: Boolean to determine whether to write the data to files
        verbose: Boolean for extra print statements about progress
        plot: Save a png file of the topographic variables (not geolocated, but can be opened in Preview)
    
    Returns
    ---------
        ds: An xarray containing the aspect, slope, accumulation and TWI.

    """
    if verbose:
        print(f"Starting topography.py")

    if not ds:
        if verbose:
            print("Loading the pre-downloaded terrain tif")
        terrain_tif = os.path.join(outdir, f"{stub}_terrain.tif")
        if not os.path.exists(terrain_tif):
            raise Exception(f"{terrain_tif} does not exist. Please run terrain_tiles.py first.")
        da = rxr.open_rasterio(terrain_tif).isel(band=0).drop_vars('band')
        ds = da.to_dataset(name='terrain')
    
    ds.rio.write_crs("EPSG:3857", inplace=True)

    if smooth:
        if verbose:
            print("Smoothing the terrain using a gaussian filter")
        terrain_tif = os.path.join(outdir, f"{stub}_terrain_smoothed.tif")
        sigma = int(sigma)
        dem = ds['terrain'].values
        dem_smooth = gaussian_filter(dem.astype(float), sigma=sigma)
        ds['dem_smooth'] = (["y", "x"], dem_smooth)
        ds["dem_smooth"].rio.to_raster(terrain_tif)
    
    if verbose:
        print("Calculating accumulation")
    grid, dem, fdir, acc = pysheds_accumulation(terrain_tif)
    aspect = fdir.astype('uint8')

    if verbose:
        print("Calculating slope and TWI")
    slope = calculate_slope(terrain_tif)
    twi = calculate_TWI(acc, slope)

    ds['accumulation'] = (["y", "x"], acc)
    ds['aspect'] = (["y", "x"], aspect)
    ds['slope'] = (["y", "x"], slope)
    ds['twi'] = (["y", "x"], twi)

    if savetifs:
        if verbose:
            print("Saving the tif files")
        for topographic_variable in topographic_variables:
            filepath = os.path.join(outdir, f"{stub}_{topographic_variable}.tif")
            ds[topographic_variable].rio.to_raster(filepath)
            if verbose:
                print("Saved:", filepath)

    if plot:
        plot_topography(ds, outdir, stub, verbose=verbose)

    ds = ds.drop_vars('spatial_ref')

    return ds


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description=f"""Derive topographic variables from the elevation ({topographic_variables}). 
                                     Note: this function assumes there already exists a file named (outdir)/(stub)_terrain.tif""")
    
    parser.add_argument('--outdir', default='.', help='The directory to save the topographic variables. (default is the current directory)')
    parser.add_argument('--stub', default='TEST', help='The name to be prepended to each file download. (default: TEST)')
    parser.add_argument('--smooth', default=False, action="store_true", help='boolean to determine whether to apply a gaussian filter to the elevation before deriving topographic variables. (default: False)')
    parser.add_argument('--sigma', default='5', help='Smoothing parameter to use for the gaussian filter. Not used if smooth=False (default: 5)')
    parser.add_argument('--plot', default=False, action="store_true", help="Save a png of the topographic variables that isn't geolocated but can be opened in Preview (default: False)")

    return parser.parse_args()
# -

if __name__ == '__main__':
    args = parse_arguments()
    
    outdir = args.outdir
    stub = args.stub
    smooth = args.smooth
    sigma = args.sigma
    plot = args.plot

    topography(outdir, stub, smooth, sigma, plot=plot)
