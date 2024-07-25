import os
import rioxarray as rio
import xarray as xr
import logging
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils import create_paths, start_logging
from config import variables, param_grid_local, param_grid_global
from modelling_functions import full_modelling, data_preprocess, save_results


def main():

    data_path = "data"

    ############  Setup ############

    # Setup file paths
    _, _, _, _, cube_crop_mask_path = create_paths(data_path=data_path)

    # Load the croped cube (croped with forest mask and germany border)
    cube_subset_crop_mask = xr.open_dataset(cube_crop_mask_path)

    # transform the cube to a dataframe
    all_data_df = cube_subset_crop_mask.to_dataframe().dropna()

    # Basic preprocessing - Scaling to mean 0 and std 1 
    all_data_scaled, scalar_x, scalar_y = data_preprocess(all_data_df, variables)

    # based on the dataframe create a list of lat lon pairs, defining all timeseries (pixels)
    lat_lon_pairs = all_data_scaled[["lat", "lon"]].drop_duplicates()


    ############  Modelling ############

    # Set lookback to 30 as it was the best performing in the test modelling
    look_back = 30


    # print grid searc parameter grid for the local model (can be found in config.py)
    logging.info(print(param_grid_local))

    cv = TimeSeriesSplit(n_splits=2)
    
    os.makedirs(os.path.join("results","logs"), exist_ok=True)


    # Run the gridsearch gridsearch and training again with the local model without auto regression as it was the best perfoming model in 04_test_modelling.py
    # Also relu is not used as activation function as tanh was performing better in every case
            
    # Create a log file: 
    start_logging(os.path.join("results", "logs", f"final_local_models_{look_back}.log"))

    # not auto regressive
    output_data_local_auto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_local, scalar_y,
                    auto_regressive=False, global_model=False, cv=cv)

    save_results(output_data_local_auto, look_back,
                 auto_regressive=False, global_model=False,
                 out_path=os.path.join("results", "modelling", "final", "results_full_local_auto_l30.json"))

    
if __name__ == "__main__":
    main()