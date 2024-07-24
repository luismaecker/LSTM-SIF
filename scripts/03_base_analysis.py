# scripts/base_analysis.py

import xarray as xr
from scipy.signal import detrend
import matplotlib.pyplot as plt
import os

from utils import create_paths



def calc_change(summer_mean_cube, sif_variable="sif_gosif", years=[2018, 2019]):
    """
    Calculate the change in summer mean SIF for the specified years compared to the baseline up to 2017.
    """
    changes = {}
    summer_mean_to_2017 = summer_mean_cube.sel(year=slice(None, 2017)).mean(dim='year')

    for year in years:
        summer_mean = summer_mean_cube.sel(year=year)
        change = summer_mean - summer_mean_to_2017
        changes[year] = change

    return changes

# TODO: add plot
def plot_save_diff(changes, save_path):
    return None


# TODO: add timeseries plot

def plot_timeseries(cube,  save_path, variable="sif_gosif", show = False):
    """
    Plot and save the timeseries of the SIF data.
    
    Parameters
    ----------
    cube : xarray.Dataset
        The input cube containing the SIF data.
    save_path : str 
        The path where the plot should be saved.
    variable : str, optional
        The variable to plot.
    show : bool, optional
        Whether to show the plot.
    
    """

    
    plt.figure(figsize=(10, 6))
    
    cube.plot(marker='o', color='blue', linestyle='dashed')

    plt.title(f'Time Series of SIF', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Sun-Induced Chlorophyll Fluorescence at 757 nm \n [W m^-2 sr^-1 um^-1]', fontsize=12)
    plt.grid(True, which='major', axis='both')
    #plt.ylim(.1, .6) 
            
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return plt



def base_analysis(cube, years=[2018, 2019]):
    """
    Perform the base analysis by calculating the summer mean for each year and the change compared to the baseline up to 2017.

    Parameters
    ----------
    cube : xarray.Dataset
        The input cube containing the SIF data.
    years : list, optional
    """


    # Calculate summer mean for each year
    summer_data = cube.sif_gosif.sel(time=cube['time.season'] == 'JJA')
    summer_mean_cube = summer_data.groupby('time.year').mean(dim='time')

    # Calculate change in summer mean SIF for each year compared to baseline up to 2017
    changes = calc_change(summer_mean_cube, years=years)
    
    # TODO: Add Plotting and saving results to results folder

    # Plot and save the timeseries

    return changes


if __name__ == "__main__":
    
    data_path = "data"

    # Get path to the cube subset
    _, _, _, cube_crop_path = create_paths(data_path)

    # Load the cropped cube subset
    cube_subset_crop = xr.open_dataset(cube_crop_path)

    changes = base_analysis(cube_subset_crop, years=[2018, 2019], detrend_data=False)