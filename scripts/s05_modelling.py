import os
import rioxarray as rio
import xarray as xr
import logging
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils import start_logging, create_paths
from config import variables, param_grid_final
from modelling_functions import full_modelling, save_results

from modelling_functions import setup_model_data

def main():

    data_path = "data"
   
    # Setup file paths
    _, _, _, _, cube_crop_mask_path = create_paths(data_path=data_path)

    # Load the croped cube (croped with forest mask and germany border)
    cube_subset_crop_mask = xr.open_dataset(cube_crop_mask_path)
   
    # do preprocessing: scaling and creating a dataframe, as well as getting the lat lon pairs defining all pixels
    all_data_scaled, lat_lon_pairs, scalar_y = setup_model_data(cube_subset_crop_mask, variables)


    ############  Modelling ############

    # Set lookback to 30 as it was the best performing in the test modelling
    look_back = 30


    # print grid searc parameter grid for the local model (can be found in config.py)
    logging.info(print(param_grid_final))

    cv = TimeSeriesSplit(n_splits=2)
    
    os.makedirs(os.path.join("results","logs"), exist_ok=True)


    # Run the gridsearch gridsearch and training again with the local model without auto regression as it was the best perfoming model in 04_test_modelling.py
    # Also relu is not used as activation function as tanh was performing better in every case
            
    # Create a log file: 
    start_logging(os.path.join("results", "logs", f"final_local_models_{look_back}.log"))

    # not auto regressive
    output_data_local_auto, test_index = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_final, scalar_y,
                    auto_regressive=False, global_model=False, cv=cv)

    save_results(output_data_local_auto, look_back,
                 auto_regressive=False, global_model=False,
                 out_path=os.path.join("results", "modelling", "final", "results_full_local_auto_l30.json"))

    np.save(test_index, os.path.join("results", "modelling", "final", "test_index.npy"))
    
if __name__ == "__main__":
    main()