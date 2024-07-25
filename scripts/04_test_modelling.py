import os
import rioxarray as rio
import xarray as xr
import logging
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils import create_paths, start_logging
from config import variables
from modelling_functions import  setup_model_data, save_results, full_modelling



def main():

    data_path = "data"
   
    # Setup file paths
    _, _, _, _, cube_crop_mask_path = create_paths(data_path=data_path)

    # Load the croped cube (croped with forest mask and germany border)
    cube_subset_crop_mask = xr.open_dataset(cube_crop_mask_path)
   
    # do preprocessing: scaling and creating a dataframe, as well as getting the lat lon pairs defining all pixels
    all_data_scaled, lat_lon_pairs, scalar_y = setup_model_data(cube_subset_crop_mask, variables)



    ############  Modelling ############

    # Create lookback array
    look_backs = [15,30,45]

    # Define the parameter grid for the local model
    param_grid_local = {
    'units_lstm': [64, 128],
    'activation': ['relu', 'tanh'], 
    'epochs': [100],    
    'learning_rate': [0.0001],
    'dropout_rate': [0.2,0.4],
    'batch_size': [25],
    'num_lstm_layers': [1, 2, 3]
    }



    cv = TimeSeriesSplit(n_splits=3)
    # Run the local models for a subset
    # and saving the results - filename is created based on model type and lookback
    for look_back in look_backs:
        
        # os.makedirs(os.path.join("results","logs"), exist_ok=True)
        
        # Create a log file: 
        #start_logging(os.path.join("results", "logs", f"local_models_{look_back}.log"))

        # not auto regressive
        output_data_local_auto = full_modelling(all_data_scaled, look_back, 
                        lat_lon_pairs, param_grid_local, scalar_y,
                        auto_regressive=False, global_model=False,
                        subset=True, n_subset=2, cv=cv)

        save_results(output_data_local_auto, look_back, auto_regressive=False, global_model=False)

        # auto regressive
        output_data_local_noauto = full_modelling(all_data_scaled, look_back, 
                        lat_lon_pairs, param_grid_local, scalar_y,
                        auto_regressive=True, global_model=False,
                        subset=True, n_subset=2, cv=cv)
        
        save_results(output_data_local_noauto, look_back, auto_regressive=True, global_model=False)


    # Run the global model on the full dataset
    # and saving the results - filename is created based on model type and lookback

    # grid for global model
    # The parameters were reduced based on a first run for the results of the local model

    param_grid_global = {
        'units_lstm': [64, 128],
        'activation': ['tanh'], 
        'epochs': [100],    
        'learning_rate': [0.0001],
        'dropout_rate': [0.2],
        'batch_size': [25],
        'num_lstm_layers': [1, 2]
    }

    # The lookback is set to 30 for the global model, as it was the best performing for the local models
    look_back = 30

    # not auto regressive
    output_data_global_auto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_global, scalar_y,
                    auto_regressive=False, global_model=False, cv=cv)

    save_results(output_data_global_auto, look_back, auto_regressive=False, global_model=False)

    # auto regressive
    output_data_global_noauto = full_modelling(all_data_scaled, look_back, 
                    lat_lon_pairs, param_grid_global, scalar_y,
                    auto_regressive=False, global_model=False, cv=cv)

    save_results(output_data_global_noauto, look_back, auto_regressive=True, global_model=False)


if __name__ == "__main__":
    main()