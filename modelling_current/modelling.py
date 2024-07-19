# %% [markdown]
# # Packages

# %%
import os
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
import logging

import json
import numpy as np
import keras

# %%
from scikeras.wrappers import KerasRegressor


# %%
import os
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr

import json
import numpy as np


# %%
import lexcube

# %% [markdown]
# # Functions

# %% [markdown]
# ### Preprocess

# %%
#TODO: Delete
# with forest classes
def data_preprocess(df, variables, forest_vars):
    """
    Preprocesses the DataFrame by resetting index, sorting, removing NaNs, converting types, and normalizing.

    Parameters:
    df (DataFrame): Input DataFrame.
    variables (list of str): Columns to normalize and convert to float32.
    forest_vars (list of str): Columns to keep unscaled.

    Returns:
    DataFrame: Processed and normalized DataFrame.
    """
    df = df.reset_index(inplace=False)
    df = df.sort_values("time")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[variables] = df[variables].astype("float32")

    # Scale only the specified variables
    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_minmax.fit_transform(df[variables])
    scaled_df = pd.DataFrame(scaled_data, columns=variables)

    # Combine scaled variables with unscaled forest variables and other columns
    all_data_scaled = scaled_df.copy()
    all_data_scaled[forest_vars] = df[forest_vars].values
    all_data_scaled["time"] = df["time"].values
    all_data_scaled["lat"] = df["lat"].values
    all_data_scaled["lon"] = df["lon"].values

    return all_data_scaled, scaler_minmax

# %%
def data_preprocess(df, variables):
    """
    Preprocesses the DataFrame by resetting index, sorting, removing NaNs, converting types, and normalizing.

    Parameters:
    df (DataFrame): Input DataFrame.
    variables (list of str): Columns to normalize and convert to float32.
    forest_vars (list of str): Columns to keep unscaled.

    Returns:
    DataFrame: Processed and normalized DataFrame.
    """
    df.reset_index(inplace=True)
    df.sort_values("time", inplace = True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[variables] = df[variables].astype("float32")

    # Scale the data using to a mean of 0 and standard div of 1
    # do this seperately for the target variable to be able to apply inverse_transform on the target variable only data
    scalar_x = StandardScaler()
    scalar_y = StandardScaler()
    scalar_y.fit(pd.DataFrame(df, columns=['sif_gosif']))

    scaled_data = scalar_x.fit_transform(df[variables])
    

    scaled_df = pd.DataFrame(scaled_data, columns=variables)

    # Combine scaled variables with unscaled forest variables and other columns
    scaled_df["time"] = df["time"].values
    scaled_df["lat"] = df["lat"].values
    scaled_df["lon"] = df["lon"].values

    return scaled_df, scalar_x, scalar_y

# %%
def convert_to_matrix(data_arr, look_back, target_col):
    """
    Convert the dataset into input features and target variable with specified look-back period.

    Parameters:
    data_arr (np.array): Input dataset with features and target in the last column.
    look_back (int): Number of past observations each input sample should consist of.
    target_col (string): Name of target variabel column.
    exclude_cols (list): List of Strings containing the column names to be excluded.

    Returns:
    np.array, np.array: Arrays for input features (X) and target variable (Y).
    """
    data_arr_x = data_arr.drop(columns=target_col)
    data_arr_y = data_arr[target_col]

    X, Y = [], []

    for i in range(1, len(data_arr_x) - look_back):

        d = i + look_back

        x_seq = np.array(data_arr_x[i:d])

        y_shifted = np.array(data_arr_y[(i - 1) : (d - 1)]).reshape((-1, 1))

        assert x_seq.shape[0] == y_shifted.shape[0]

        x_sequence = np.hstack([x_seq, y_shifted])

        X.append(x_sequence)
        Y.append(data_arr_y.iloc[d - 1])

    return np.array(X), np.array(Y)

# %%
def split_data(df_scaled, lat, lon, look_back, target_col="sif_gosif"):
    """
    Splits the scaled DataFrame into training, validation, and test sets for a specified location and look-back period.
    The timeframes for splitting are partly overlapping as to model timestep t, the timesteps from t to t-lookback are neede

    Parameters:
    df_scaled (DataFrame): Preprocessed and scaled DataFrame.
    lat (float): Latitude to filter data.
    lon (float): Longitude to filter data.
    look_back (int): Number of past observations each input sample should consist of.

    Returns:
    tuple: Arrays of features and target variables for training, validation, and test datasets.
    """
    df_scaled = df_scaled.loc[(df_scaled["lat"] == lat) & (df_scaled["lon"] == lon)]

    train_data = df_scaled[df_scaled["time"].dt.year <= 2015]
    val_data = df_scaled[
        (df_scaled["time"].dt.year == 2016) | (df_scaled["time"].dt.year == 2017)
    ]
    test_data = df_scaled[(df_scaled["time"].dt.year >= 2018)]

    train = train_data.drop(columns=["time", "lat", "lon"])
    val = val_data.drop(columns=["time", "lat", "lon"])
    test = test_data.drop(columns=["time", "lat", "lon"])

    # Extend the validation and test sets by the look-back period to include necessary preceding time steps
    if not train_data.empty:
        val = pd.concat([train.iloc[-(look_back):], val])
    if not val_data.empty:
        test = pd.concat([val.iloc[-(look_back):], test])

    trainX, trainY = convert_to_matrix(train, look_back, target_col)
    valX, valY = convert_to_matrix(val, look_back, target_col)
    testX, testY = convert_to_matrix(test, look_back, target_col)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    valX = np.reshape(valX, (valX.shape[0], valX.shape[1], valX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    test_index = sorted(list(set(test_data.time)))

    return trainX, trainY, valX, valY, testX, testY, test_index

# %% [markdown]
# ### Model and Predict

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam, RMSprop

def create_lstm_model(look_back, features, units_lstm=50, units_dense=50, learning_rate=0.001, dropout_rate=0.2, num_lstm_layers=1, activation='relu', optimizer='adam'):
    """
    Create an LSTM model with the specified hyperparameters.
    
    Parameters:
    look_back (int): The number of previous time steps to use as input.
    features (int): The number of features in the input data.
    units_lstm (int): Number of units in the LSTM layer(s).
    units_dense (int): Number of units in the Dense layer.
    activation (str): Activation function to use.
    optimizer (str): Optimizer to use ('adam' or 'rmsprop').
    learning_rate (float): Learning rate for the optimizer.
    dropout_rate (float): Dropout rate to use after LSTM layers.
    num_lstm_layers (int): Number of LSTM layers (1 or 2).

    Returns:
    model (Sequential): The compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(look_back, features)))

    if num_lstm_layers == 1:
        model.add(LSTM(units_lstm, activation='tanh'))
        model.add(Dropout(dropout_rate))

    elif num_lstm_layers == 2:
        model.add(LSTM(units_lstm, activation='tanh', return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units_lstm, activation='tanh'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(units_dense, activation=activation))
    model.add(Dense(1, activation='linear'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


# %%
# Function to create a KerasRegressor for GridSearchCV
def create_keras_regressor(look_back, features, units_lstm=50, units_dense=50, learning_rate=0.001, dropout_rate=0.2, num_lstm_layers=1,  activation='relu', optimizer='adam'):
    return KerasRegressor(
        model=create_lstm_model,
        look_back=look_back,
        features = features,
        units_lstm=50, 
        units_dense=50, 
        learning_rate=0.001, 
        dropout_rate=0.2, 
        num_lstm_layers=1,  
        activation='relu', 
        optimizer='adam',
        verbose = 0
    )


# %%
# Iterative prediction and substitution
def predict_replace(model, X_test):
    """
    Generates predictions and updates the test set input for iterative forecasting.

    Parameters:
    model (keras.Model): Trained LSTM model.
    X_test (array): Test data to predict.

    Returns:
    np.array: Array of forecasted values.
    """
    forecasts = []
    for i in range(len(X_test)):
        forecast = model.predict(X_test[i].reshape(1, look_back, -1), verbose=0)
        forecasts.append(forecast[0][0])
        if i < len(X_test) - 1:
            X_test[i + 1, :-1, -1] = X_test[i + 1, 1:, -1]
            X_test[i + 1, -1, -1] = forecast[0][0]
    forecasts_array = np.array(forecasts)
    return forecasts_array

# %% [markdown]
# ### Evaluating - Plotting

# %%
def plot_training(history):
    """
    Plots the training and validation loss and metrics from the training history.

    Parameters:
    history (History): History object from Keras training session.

    Returns:
    None
    """
    plt.figure(figsize=(14, 7))
    plt.plot(history.history["mse"], label="Train MSE")
    plt.plot(history.history["val_mse"], label="Validation MSE")
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Model Loss and Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

# %%
# Evaluation function for model performance
def evaluate_model(true_values, predicted_values, data_type="Validation"):
    # Remove NaN values
    mask = ~np.isnan(predicted_values)

    true_values = true_values[mask]
    predicted_values = predicted_values[mask]

    if len(true_values) > 0 and len(predicted_values) > 0:
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        mae = mean_absolute_error(true_values, predicted_values)
        print(f"{data_type} Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"{data_type} Mean Absolute Error (MAE): {mae:.2f}")
    else:
        print(f"{data_type} evaluation skipped due to insufficient data.")

    return rmse, mae

# %%
# Function to plot predicted vs. actual values
def plot_predicted_vs_actual(testY, forecasts, test_index, look_back):
    plt.figure(figsize=(14, 7))
    plt.plot(sorted(test_index[look_back + 1 :]), testY, label="Actual")
    plt.plot(sorted(test_index[look_back + 1 :]), forecasts, label="Predicted")
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
import matplotlib.pyplot as plt
# Function to plot predicted vs. actual values with MSE in subplots
def plot_multiple_results(results, evaluation, unique_pairs, look_back):
    num_plots = len(results)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            testY, forecasts = results[i]
            mae, rmse = evaluation[i]["mae"], evaluation[i]["rmse"]
            lat, lon = unique_pairs.iloc[i]
            time_index = sorted(test_index)

            ax.plot(time_index[:-1], testY, label="Actual")
            ax.plot(time_index[:-1], forecasts, label="Predicted")
            ax.set_title(f"Lat: {lat}, Lon: {lon}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            # Add MSE to the corner
            ax.text(
                0.95,
                0.05,
                f"RMSE: {rmse:.2f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
            )

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Writing results

# %%
# Convert the output data to a serializable format
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar types to Python scalars
    elif isinstance(obj, dict):
        # Recursively convert each item in the dictionary
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert each item in the list
        return [convert_to_serializable(v) for v in obj]
    return obj  # Return the object if it's already serializable

# %% [markdown]
# # 1. Data

# %% [markdown]
# ### Read Germany border

# %%
# Read
germany_shp_path = os.path.join("..", "data", "germany_border.shp")
germany_gpd = gpd.read_file(germany_shp_path)

# %% [markdown]
# ### Load and subset cube

# %%
cube_ger_path = os.path.join("..","data", "processed", "cube_preprocessed.nc")
cube_ger = xr.open_dataset(cube_ger_path, chunks={"time": 92, "lat": -1, "lon": -1})
cube_ger

# %% [markdown]
# ### Mask Cube only keeping cells with 50% forest cover in 2002

# %% [markdown]
# Create mask

# %%
forest_2000_mask = (cube_ger.all_classes.isel(time=0) > 50).astype(int)
forest_2000_mask.plot()

# %% [markdown]
# apply mask

# %%
cube_ger_f = cube_ger.where(forest_2000_mask)
cube_ger_f.sif_gosif.isel(time=0).plot()

# %% [markdown]
# crop with germany border

# %%
cube_ger_f.rio.write_crs(4326, inplace=True)
cube_ger_f_crop = cube_ger_f.rio.clip(
    germany_gpd.geometry.values, germany_gpd.crs, drop=False, all_touched=False
)

fig, ax = plt.subplots(figsize=(10, 8))
cube_ger_f_crop.sif_gosif.isel(time=0).plot(ax=ax)
germany_gpd.plot(ax=ax, edgecolor="red", facecolor="none")  # Adjust colors as needed

# %% [markdown]
# ### Preprocessing Data

# %%
variables = [
    "sif_gosif",
    "evaporation_era5",
    "precipitation_era5",
    "radiation_era5",
    "air_temperature_2m",
    "max_air_temperature_2m",
    "min_air_temperature_2m",
]

# %%
cube_ger_f_crop

# %%
# list of forest variable column names and spatial_ref, to remove them from the dataframe
forest_vars = list(set(cube_ger_f_crop.data_vars) - set(variables) - {"spatial_ref"})

# Create dataframe from cube
all_data_df = cube_ger_f_crop.to_dataframe().dropna().drop(columns=forest_vars).drop(columns="spatial_ref")
all_data_df


# %% [markdown]
# ### Preprocess dataframe 

# %% [markdown]
# scale data and drop forest variables

# %%
all_data_scaled, scalar_x, scalar_y = data_preprocess(all_data_df, variables)
all_data_scaled

# %%
sif = all_data_scaled["sif_gosif"]

# %%
all_data_scaled

# %%
all_data_scaled.iloc[:-1]

# %% [markdown]
# # 3. Model multiple timeseries with with local model using hyperparameter Tuning

# %%
# get unique paris 
unique_pairs = all_data_scaled[["lat", "lon"]].drop_duplicates()
unique_pairs

# %% [markdown]
# Setting up logging

# %%
from datetime import date

today = date.today()

# dd/mm/YY
current_date = today.strftime("%d-%m-%Y")

# %%
logging.basicConfig(filename= f"modelling_{current_date}.log",level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)

# %% [markdown]
# Running the model

# %%
print(2*"\n")
print(100*"-")
print(2*"\n")

print("Starting Modelling and GridsearchCV")


# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'units_lstm': [64, 128],
    'units_dense': [64, 128],
    'activation': ['relu'], #, 'tanh'
    'optimizer': ['adam'], #, 'rmsprop'
    'epochs': [100],    
    'learning_rate': [0.0001],
    'dropout_rate': [0.2,0.4],
    'batch_size': [25],
    'num_lstm_layers': [1, 2]
}

look_back = 40
batch_size = 32
epochs = 50
learning_rate = 0.0001
results = []
evaluation = []
histories = []
best_params = []

number = len(unique_pairs)
output_data = {}

n_splits = 3

cv = TimeSeriesSplit(n_splits=n_splits)

for i in range(6):

    
    lat, lon = unique_pairs.iloc[i]

    logging.info(f"Starting Grid Search for \n lat: {lat}\n lon: {lon}")

    trainX, trainY, valX, valY, testX, testY, test_index = split_data(
        all_data_scaled, lat, lon, look_back=look_back
    )

    # Create a KerasRegressor
    features = trainX.shape[2]
    model = create_keras_regressor(look_back, features)

    # Define GridSearchCV
    lstm_grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
    )

    # Perform grid search
    lstm_grid_search.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        shuffle=False,
    )

    # Get the best model from the grid search
    best_params = lstm_grid_search.best_params_

    logging.info(f"Gridsearch done, The best parameters are: {best_params}")


    logging.info(f"Running and evaluating model")

    lstm_model = create_lstm_model(
        look_back=look_back,
        features=features,
        units_lstm=best_params['units_lstm'],
        units_dense=best_params['units_dense'],
        activation=best_params['activation'],
        optimizer=best_params['optimizer'],
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate'],
        num_lstm_layers=best_params['num_lstm_layers']
    )

    history = lstm_model.fit(
        trainX,
        trainY,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        validation_data=(valX, valY)
    )


    forecasts = predict_replace(lstm_model, testX)

    testY_rescaled = scalar_y.inverse_transform(pd.DataFrame(testY))
    forecasts_rescaled = scalar_y.inverse_transform(pd.DataFrame(forecasts))


    rmse = root_mean_squared_error(testY_rescaled, forecasts_rescaled)
    mae = mean_absolute_error(testY_rescaled, forecasts_rescaled)

    results.append([testY_rescaled.tolist(), forecasts_rescaled.tolist()])
    evaluation.append({"lat": lat, "lon": lon, "mae": mae, "rmse": rmse})

    histories.append(history.history)


    # Add results to the output dictionary
    output_data[(lat, lon)] = {
        "best_params": best_params,
        "evaluation": {"mae": mae, "rmse": rmse},
        "results": {"true_values": testY_rescaled.tolist(), "predicted_values": forecasts_rescaled.tolist()},
        "history": history.history
    }

    
    # Convert the entire data dictionary to a serializable format
    output_data_serializable = {str(k): convert_to_serializable(v) for k, v in output_data[(lat, lon)].items()}

    output_json_file = os.path.join("result_jsons", f"{str(lat).replace('.', '_')}_{str(lon).replace('.', '_')}_model_results.json")

    # Writing the converted data to 'model_results.json'
    with open(output_json_file, "w") as file:
        json.dump(output_data_serializable, file, indent=4)

    logging.info(f"Results and evaluation written to: {output_json_file}")

    logging.info(f"Completed {i + 1}/{len(unique_pairs)}")
    print(f"Completed {i + 1}/{len(unique_pairs)}")

    logging.info(100*"-")
    logging.info(100*"-")


print(2*"\n")
print(100*"-")
print(2*"\n")





# Convert the entire data dictionary to a serializable format
output_data_serializable = {str(k): convert_to_serializable(v) for k, v in output_data.items()}

# Writing the converted data to 'model_results.json'
with open("model_results_full.json", "w") as file:
    json.dump(output_data_serializable, file, indent=4)

print("Results and evaluation have been written to 'model_results.json'")

