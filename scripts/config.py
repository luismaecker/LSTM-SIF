# scripts/config.py


variables = [
    "sif_gosif",
    "evaporation_era5",
    "precipitation_era5",
    "radiation_era5",
    "air_temperature_2m",
    "max_air_temperature_2m",
    "min_air_temperature_2m",
]


# Define the time and spatial subset
min_time = '2002-01-01'
max_time = '2021-12-31'
lon_min, lon_max = 5.866, 15.042
lat_min, lat_max = 47.270, 55.058


############ Hyperparameter Grid setup for GridSearchCV  ############  

# for local model
param_grid_local = {
    'units_lstm': [64, 128],
    'activation': ['relu', 'tanh'], 
    'epochs': [100],    
    'learning_rate': [0.0001],
    'dropout_rate': [0.2,0.4],
    'batch_size': [25],
    'num_lstm_layers': [1, 2, 3]
}

# for global model
param_grid_global = {
    'units_lstm': [64, 128],
    'activation': ['tanh'], 
    'epochs': [100],    
    'learning_rate': [0.0001],
    'dropout_rate': [0.2],
    'batch_size': [25],
    'num_lstm_layers': [1, 2]
}
