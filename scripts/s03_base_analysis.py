# scripts/base_analysis.py

import xarray as xr
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import create_paths
import matplotlib.gridspec as gridspec


def base_analysis(cube, years=[2018, 2019]):
    """
    Perform the base analysis by calculating the summer mean for each year and the change compared to the baseline up to 2017.

    Parameters
    ----------
    cube : xarray.Dataset
        The input cube containing the SIF data.
    years : list, optional
    """


    # Calculate summer mean for each year for the sif variable
    summer_sif_data = cube.sif_gosif.sel(time=cube['time.season'] == 'JJA')
    summer_sif_mean_cube = summer_sif_data.groupby('time.year').mean(dim='time')

    # Calculate change in summer mean SIF for each year compared to baseline up to 2017
    changes = {}
    summer_mean_to_2017 = summer_sif_mean_cube.sel(year=slice(None, 2017)).mean(dim='year')

    for year in years:
        summer_mean = summer_sif_mean_cube.sel(year=year)
        change = summer_mean - summer_mean_to_2017
        changes[year] = change
    

    return summer_sif_mean_cube, summer_mean_to_2017, changes 

# Creates a figure with 2x2 subplots to visualize reference period data, 2018 data, and the difference between the two.
def change_plot(ref_period,data_2018, changes, save_path = None):

    # Create the figure and 2x2 subplots
    fig, axd = plt.subplot_mosaic([['upleft', 'right'],
                                ['lowleft', 'right']], layout='constrained', figsize=(10, 7))

    # Plot each time slice on a different subplot
    img1 = ref_period.plot(ax=axd["upleft"], cmap="viridis", vmin=0, vmax=0.5, add_colorbar=False)
    axd["upleft"].set_title("Mean 2002 - 2017", fontsize=13, fontweight='bold', pad=15)
    axd["upleft"].set_xlabel("Longitude", fontsize=12)
    axd["upleft"].set_ylabel("Latitude", fontsize=12)

    img2 = data_2018.plot(ax=axd["lowleft"], cmap="viridis", vmin=0, vmax=0.5, add_colorbar=False)
    axd["lowleft"].set_title("Mean 2018", fontsize=13, fontweight='bold', pad=15)
    axd["lowleft"].set_xlabel("Longitude", fontsize=12)
    axd["lowleft"].set_ylabel("Latitude", fontsize=12)

    img3 = changes[2018].plot(ax=axd["right"], cmap="RdBu", vmin=-0.15, vmax=0.15, add_colorbar=False)
    axd["right"].set_title("Difference SIF 2018 to mean of 2002 - 2017", fontsize=13, fontweight='bold', pad=15)
    axd["right"].set_xlabel("Longitude", fontsize=12)
    axd["right"].set_ylabel("Latitude", fontsize=12)

    # Add colorbars for each row
    divider1 = make_axes_locatable(axd["upleft"])
    cax1 = divider1.append_axes("right", size="5%", pad=0.5)
    fig.colorbar(img1, cax=cax1, orientation="vertical").ax.tick_params(labelsize=12)

    divider2 = make_axes_locatable(axd["lowleft"])
    cax2 = divider2.append_axes("below", size="5%", pad=0.5)
    fig.colorbar(img2, cax=cax2, orientation="vertical").ax.tick_params(labelsize=12)

    divider3 = make_axes_locatable(axd["right"])
    cax3 = divider3.append_axes("right", size="5%", pad=0.5)
    fig.colorbar(img3, cax=cax3, orientation="vertical").ax.tick_params(labelsize=12)


    if save_path:
        # save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt

def change_plot(ref_period, data_2018, changes, save_path=None):
    # Create the figure with GridSpec for custom layout
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 0.05, 0.15, 1,0.05 ], wspace=0.3, hspace=0.3)

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 3])
    cax = fig.add_subplot(gs[:, 1])
    cax_2 = fig.add_subplot(gs[:, 4])

    # Plot each dataset on a different subplot
    img1 = ref_period.plot(ax=ax1, cmap="viridis", vmin=0, vmax=0.5, add_colorbar=False)
    ax1.set_title("Mean 2002 - 2017", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Longitude", fontsize=12)
    ax1.set_ylabel("Latitude", fontsize=12)

    img2 = data_2018.plot(ax=ax2, cmap="viridis", vmin=0, vmax=0.5, add_colorbar=False)
    ax2.set_title("Mean 2018", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("Longitude", fontsize=12)
    ax2.set_ylabel("Latitude", fontsize=12)

    img3 = changes[2018].plot(ax=ax3, cmap="RdBu", vmin=-0.15, vmax=0.15, add_colorbar=False)
    ax3.set_title("Difference SIF 2018 to mean of 2002 - 2017", fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel("Longitude", fontsize=12)
    ax3.set_ylabel("Latitude", fontsize=12)

    # Add a shared colorbar for the left plots
    cbar = fig.colorbar(img1, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Sun-Induced Chlorophyll Fluorescence \n [W m^-2 sr^-1 um^-1]", fontsize=10)

    # Add a shared colorbar for the left plots
    cbar2 = fig.colorbar(img3, cax=cax_2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label("Sun-Induced Chlorophyll - Change", fontsize=10)
    
    ax1.set_xlabel("")

    if save_path:
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()




# Plotting a timeseries of the mean german SIF data
def plot_timeseries(cube_subset_mask, save_path, time_range=[None, None], show = False):
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

    # only use sif variable
    cube_subset_mask_sif = cube_subset_mask.sif_gosif

    # Calculate the mean of the SIF data over germany
    time_series = cube_subset_mask_sif.mean(dim=['lat', 'lon'])

    # select range
    time_series = time_series.sel(time=slice(time_range[0], time_range[1]))

    plt.figure(figsize=(10, 6))
    
    time_series.plot(marker='o', color='blue', linestyle='dashed')

    plt.title(f'Time Series of SIF', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Sun-Induced Chlorophyll Fluorescence at 757 nm \n [W m^-2 sr^-1 um^-1]', fontsize=12)
    plt.grid(True, which='major', axis='both')
    #plt.ylim(.1, .6) 
            
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()

def main():
    
    data_path = "data"

    # Get path to the cube subset
    _, _, _, cube_crop_path, cube_crop_mask_path = create_paths(data_path)

    # Load masked cube subset (croped with forest mask and germany border)
    cube_subset_mask = xr.open_dataset(cube_crop_mask_path)

    # Create the results directory
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    # Save plot of timeseries:
    plot_timeseries(cube_subset_mask, save_path = os.path.join("results", "figures", "timeseries_full.png"))
    plot_timeseries(cube_subset_mask, time_range= ["2015-01-01", "2022-12-31"], save_path = os.path.join("results", "figures", "timeseries_recent.png"))

    # Load croped cube subset (not masked yet)
    cube_subset_crop = xr.open_dataset(cube_crop_path)

    # Calculate the summer mean for each year and the change compared to the baseline up to 2017
    summer_mean_cube, summer_mean_to_2017, changes = base_analysis(cube_subset_crop, years=[2018, 2019])

    # Select only year 2018
    summer_mean_2018 = summer_mean_cube.sel(year=2018)

    # Create and save plot showing differences
    save_path = os.path.join("results", "figures", "base_analysis.png")
    change_plot(summer_mean_to_2017,summer_mean_2018, changes, save_path)


if __name__ == "__main__":
    main()