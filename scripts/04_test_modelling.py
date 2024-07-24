import os
import rioxarray as rio
import xarray as xr
import logging
import pandas as pd

from utils import create_paths
from config import variables, param_grid_local, param_grid_global
from modelling_functions import full_modelling, data_preprocess, save_results

############  Data Setup ############

data_path = "data"

# Setup file paths
_, _, _, cube_crop_path = create_paths(data_path=data_path)

# Load the croped cube (croped with forest mask and germany border)
cube_subset_crop = xr.open_dataset(cube_crop_path)

# transform the cube to a dataframe
all_data_df = cube_subset_crop.to_dataframe().dropna()

# Basic preprocessing - Scaling to mean 0 and std 1 
all_data_scaled, scalar_x, scalar_y = data_preprocess(all_data_df, variables)

# based on the dataframe create a list of lat lon pairs, defining all timeseries (pixels)
lat_lon_pairs = all_data_scaled[["lat", "lon"]].drop_duplicates()


############  Modelling ############

# Create lookback array
look_backs = [15,30,45]


# print grid searc parameter grid for the local model (can be found in config.py)
logging.info(print(param_grid_local))

# Run the local models for a subset
# and saving the results - filename is created based on model type and lookback
for look_back in look_backs:
    
    # not auto regressive
    output_data_local_auto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_local, scalar_y,
                    auto_regressive=False, global_model=False,
                    subset=True, n_subset=6)

    save_results(output_data_local_auto, look_back, auto_regressive=False, global_model=False)

    # auto regressive
    output_data_local_noauto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_local, scalar_y,
                    auto_regressive=True, global_model=False,
                    subset=True, n_subset=6)
    
    save_results(output_data_local_noauto, look_back, auto_regressive=True, global_model=False)


# Run the global model on the full dataset
# and saving the results - filename is created based on model type and lookback
for look_back in look_backs:
    
    # not auto regressive
    output_data_global_auto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_local, scalar_y,
                    auto_regressive=False, global_model=False)
    
    save_results(output_data_global_auto, look_back, auto_regressive=False, global_model=False)

    # auto regressive
    output_data_global_noauto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_local, scalar_y,
                    auto_regressive=False, global_model=False)
    
    save_results(output_data_global_noauto, look_back, auto_regressive=True, global_model=False)
