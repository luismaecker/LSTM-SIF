# %% [markdown]
# # Packages

# %%
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import seaborn as sns
import tensorflow as tf
import xarray as xr
import xcube
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam
from scalecast.Forecaster import Forecaster
from scalecast.SeriesTransformer import SeriesTransformer
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xcube.core.store import new_data_store

# %%
import lexcube

# %% [markdown]
# # Functions

# %%
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

# %%
def improved_model_lstm(look_back, features, learning_rate=0.001):
    """
    Constructs an LSTM model with dropout for regularization.

    Parameters:
    look_back (int): Number of time steps each input sequence contains.
    features (int): Number of input features per time step.
    learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
    keras.Model: A compiled LSTM model with specified architecture and loss function.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(look_back, features), activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=optimizer, metrics=["mse", "mae"]
    )
    return model


def model_lstm(look_back, features):
    """
    Defines a simpler LSTM model for regression tasks.

    Parameters:
    look_back (int): Number of time steps in the input data.
    features (int): Number of features in the input data.

    Returns:
    keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(look_back, features), activation="relu"))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse", "mae"])
    return model

# %%
def run_model(
    trainX,
    trainY,
    valX,
    valY,
    look_back,
    epochs,
    learning_rate=0.001,
    batch_size=32,
    verbose=1,
):
    """
    Trains an LSTM model using provided training and validation data.

    Parameters:
    trainX (array): Input features for training.
    trainY (array): Target output for training.
    valX (array): Input features for validation.
    valY (array): Target output for validation.
    look_back (int): Number of time steps to consider for each input sequence.
    epochs (int): Number of epochs for training the model.
    learning_rate (float): Learning rate for the optimizer.
    batch_size (int): Number of samples per batch.
    verbose (int): Verbosity mode.

    Returns:
    tuple: Returns the trained model and the history object containing training details.
    """
    num_features = trainX.shape[2]
    model = improved_model_lstm(look_back, num_features, learning_rate)
    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valX, valY),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        verbose=verbose,
        shuffle=False,
    )
    return model, history

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

# %% [markdown]
# # 1. Data

# %% [markdown]
# ### Read Germany border

# %%
# Read
germany_shp_path = "data/germany_border.shp"
germany_gpd = gpd.read_file(germany_shp_path)

# %% [markdown]
# ### Load and subset cube

# %%
cube_ger_path = os.path.join("data", "processed", "cube_preprocessed.nc")
cube_ger = xr.open_dataset(cube_ger_path, chunks={"time": 92, "lat": -1, "lon": -1})
cube_ger

# %% [markdown]
# # Lex

# %%
# Create a plot
fig, ax = plt.subplots()

# Plot the border with specified colors
germany_gpd.plot(ax=ax, edgecolor="red", facecolor="none")

# Remove axis
ax.axis("off")
plt.savefig(
    "output/germany_border.png", bbox_inches="tight", pad_inches=0, transparent=True
)

# %%
cube_ger

# %%
np.nanmax(cube_ger.evaporation_era5)

# %%
cube_ger.rio.write_crs(4326, inplace=True)
cube_ger = cube_ger.rio.clip(
    germany_gpd.geometry.values, germany_gpd.crs, drop=False, all_touched=False
)

# %%
w = lexcube.Cube3DWidget(
    cube_ger.radiation_era5, cmap="viridis", vmin=0, vmax=1006265
)  #
w

# %%
wa = lexcube.Cube3DWidget(
    cube_ger.radiation_era5, cmap="inferno", vmin=0, vmax=1006265.8
)  #
wa

# %%
c = lexcube.Cube3DWidget(cube_ger.precipitation_era5, cmap="YlGnBu", vmin=0, vmax=20)  #
c

# %%
d = lexcube.Cube3DWidget(
    cube_ger.air_temperature_2m, cmap="cividis", vmin=0, vmax=25
)  #
d

# %%
w = lexcube.Cube3DWidget(
    cube_ger.air_temperature_2m, cmap="bamako_r", vmin=0, vmax=0.7
)  #
w

# %%
os.getcwd()

# %%
w.savefig(fname="sif_cube.png", include_ui=False, dpi_scale=2.0)

# %% [markdown]
# ### Mask Cube only keeping cells with 50% forest cover in 2002

# %%
forest_2000_mask = (cube_ger.all_classes.isel(time=0) > 50).astype(int)
forest_2000_mask.plot()

# %%
cube_ger_f = cube_ger.where(forest_2000_mask)
cube_ger_f.sif_gosif.isel(time=0).plot()

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
all_data_df = cube_ger_f_crop.to_dataframe()
all_data_df

# %%
all_data_df = cube_ger_f_crop.to_dataframe().dropna()
print(len(all_data_df))
num_unique_lat_lon_pairs = (
    all_data_df.index.to_frame(index=False)[["lat", "lon"]].drop_duplicates().shape[0]
)
print(num_unique_lat_lon_pairs)

# %% [markdown]
# ### Preprocess dataframe 

# %%
forest_vars = list(set(all_data_df.columns) - set(variables) - {"spatial_ref"})
len(forest_vars)
forest_vars

# %%
all_data_scaled, scaler_minmax = data_preprocess(all_data_df, variables, forest_vars)
all_data_scaled
all_data_scaled = all_data_scaled.drop(columns=forest_vars)

# %%
cols = all_data_scaled.columns.tolist()
new_order = cols[1:-3] + [cols[0]] + cols[-3:]
all_data_scaled = all_data_scaled[new_order]
all_data_scaled

# %% [markdown]
# # 2. Model multiple timeseries with local model using fixed parameters

# %% [markdown]
# ## 2.1 Train and test split

# %%
unique_pairs = all_data_scaled[["lat", "lon"]].drop_duplicates()
unique_pairs

# %%
look_back = 40
epochs = 50

results_untuned = []
evaluation_untuned = []
length = len(unique_pairs)

for i in range(2):
    lat, lon = unique_pairs.iloc[i]
    trainX, trainY, valX, valY, testX, testY, test_index = split_data(
        all_data_scaled, lat, lon, look_back=look_back
    )
    print(trainX.shape, valX.shape, testX.shape, testY.shape, len(test_index))

    model_untuned, history_untuned = run_model(
        trainX, trainY, valX, valY, look_back=look_back, epochs=epochs, verbose=0
    )
    forecasts_untuned = predict_replace(model_untuned, testX)
    # evaluate_model(trainY, train_predict[:, 0], 'Train')
    mse, mae = evaluate_model(testY, forecasts_untuned[:], "Test")
    results_untuned.append([testY, forecasts_untuned])
    evaluation_untuned.append([mae, mse])

# %% [markdown]
# # Results

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
            mae, mse = evaluation[i]
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
                f"MSE: {mse:.2f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
            )

    plt.tight_layout()
    plt.show()


# Usage example:
# results is a list of tuples (testY, forecasts)
# evaluation is a list of tuples (mae, mse)
# unique_pairs is a DataFrame with columns 'lat' and 'lon'
plot_multiple_results(results_untuned, evaluation_untuned, unique_pairs, look_back)

# %% [markdown]
# # 3. Model multiple timeseries with with local model using hyperparameter Tuning

# %%
# Train the model
def run_model(
    trainX,
    trainY,
    valX,
    valY,
    look_back,
    epochs,
    learning_rate=0.001,
    batch_size=32,
    verbose=1,
):

    num_features = trainX.shape[2]
    model = improved_model_lstm(look_back, num_features, learning_rate)
    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valX, valY),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        verbose=verbose,
        shuffle=False,
    )
    return model, history

# %%
# Function to create a KerasRegressor for GridSearchCV
def create_keras_regressor(look_back, features):
    return KerasRegressor(
        model=create_model,
        look_back=look_back,
        features=features,
        dropout_rate=0.1,
        hidden_dim=64,
        hidden_dense=64,
        verbose=0,
    )


params = {
    #'learning_rate': [1e-3, 1e-4, 1e-5],
    "hidden_dim": [32, 64, 128],
    "dropout_rate": [0.1, 0.3],
    "hidden_dense": [32, 64, 128]
    #'epochs': [50, 100, 150],
    #'batch_size': [16, 32, 64]
}


# %%
def create_model(
    look_back,
    features,
    learning_rate=0.001,
    hidden_dim=128,
    dropout_rate=0.1,
    hidden_dense=64
):
    """
    Creates and compiles an LSTM model.

    Parameters:
    look_back (int): Number of previous time steps to consider as input.
    features (int): Number of features in the input data.
    learning_rate (float): Learning rate for the optimizer.
    hidden_dim (int): Number of neurons in the LSTM layer.
    dropout_rate (float): Dropout rate for regularization.
    hidden_dense (int): Number of neurons in the dense layer following the LSTM layer.

    Returns:
    model (keras.Model): Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(look_back, features), activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(hidden_dense, activation="relu"))
    model.add(Dense(1))  # Output layer; assumes a single target variable.

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="mean_squared_error", optimizer=optimizer, metrics=["mse", "mae"]
    )

    return model

# %%
look_back = 40
batch_size = 32
epochs = 100
learning_rate = 0.0001
n_splits = 3
results = []
evaluation = []
histories = []
best_params = []

number = len(unique_pairs)

output_data = {}

for i in range(number):
    lat, lon = unique_pairs.iloc[i]
    trainX, trainY, valX, valY, testX, testY, test_index = split_data(
        all_data_scaled, lat, lon, look_back=look_back
    )

    # Create a KerasRegressor
    features = trainX.shape[2]
    model = create_keras_regressor(look_back, features)

    # Define GridSearchCV
    lstm_gs = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )

    # Perform grid search
    lstm_gs.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        shuffle=False,
    )

    # Get the best model from the grid search
    best_params = lstm_gs.best_params_

    model = create_model(
        look_back,
        features,
        learning_rate=learning_rate,
        hidden_dim=best_params["hidden_dim"],
        hidden_dense=best_params["hidden_dense"],
        dropout_rate=best_params["dropout_rate"],
    )

    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valX, valY),
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        verbose=1,
        shuffle=False,
    )

    forecasts = predict_replace(model, testX)

    rmse = root_mean_squared_error(testY, forecasts)
    mae = mean_absolute_error(testY, forecasts)

    results.append([testY.tolist(), forecasts.tolist()])
    evaluation.append({"lat": lat, "lon": lon, "mae": mae, "rmse": rmse})

    histories.append(history.history)
     
    # Add results to the output dictionary
    output_data[(lat, lon)] = {
        "best_params": best_params,
        "evaluation": {"mae": mae, "rmse": rmse},
        "results": {"true_values": testY.tolist(), "predicted_values": forecasts.tolist()},
        "history": history.history,
    }
    
    print(f"Completed {i + 1}/{len(unique_pairs)}")


import json
import numpy as np

def convert_to_serializable(obj):
    """Convert objects to a type that can be serialized with JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

# Convert the data to a serializable format
output_data_serializable = convert_to_serializable(output_data)

# Writing data to 'model_results.txt'
with open("output/model_results_0711.txt", "w") as file:
    file.write(json.dumps(output_data_serializable, indent=4))

print("Results and evaluation have been written to 'model_results.txt'")


# %%
mean_squared_error(testY, forecasts)

# %%
best_params

# %%
mae

# %%
evaluation

# %%
evaluation

# %%
plot_multiple_results(results, evaluation, unique_pairs, look_back)

# %%
plot_training(history)


