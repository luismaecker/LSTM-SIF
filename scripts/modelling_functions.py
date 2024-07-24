import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

import json
import numpy as np






########################################################
# Data Preprocessing
########################################################

# Function to preprocess: scale and restructure the full dataset
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


# Function to convert the DataFrame to 3D array for LSTM model training
def convert_to_matrix(data_arr, look_back, target_col =  "sif_gosif", autoregressive = True):
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

    # if auto-regressive model, we include the target variable in our predictors
    # we need to shift the target variable by one timestep to use it as a feature
    if autoregressive:

        # start range at 1 as we use the shifted target variable as a feature - one timestep before the other features begin  
        # we go from i to the next look_back timesteps, so we need to stop look_back timesteps before the end of the array
        for i in range(1, len(data_arr_x) - look_back):
            
            # when modelling timestep t, d is t+1
            d = i + look_back

            x_seq = np.array(data_arr_x[i:d])

            y_shifted = np.array(data_arr_y[(i - 1) : (d - 1)]).reshape((-1, 1))

            assert x_seq.shape[0] == y_shifted.shape[0]

            x_sequence = np.hstack([x_seq, y_shifted])

            X.append(x_sequence)
            Y.append(data_arr_y.iloc[d - 1])

    else:
        for i in range(1, len(data_arr_x) - look_back):
            d = i + look_back
            x_seq = np.array(data_arr_x[i:d])
            X.append(x_seq)
            Y.append(data_arr_y.iloc[d - 1])

    

    return np.array(X), np.array(Y)


# Function to split the data into training, validation, and test sets
def split_data(df_scaled, look_back,  lat_lon_pairs, lat = None, lon = None,global_model = False, target_col="sif_gosif", autoregressive = True):
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

    # if the model is global, we store the indices of the lat_lon_pairs in a dictionary 
    if global_model:

        df_scaled = df_scaled.sort_values(by=["time", "lat", "lon"])

        pixel_indices = {}


        for idx, (lat, lon) in lat_lon_pairs.iterrows():
            pixel_data = df_scaled.loc[(df_scaled["lat"] == lat) & (df_scaled["lon"] == lon)]
            pixel_indices[(lat, lon)] = pixel_data.index

    # if the model is local, we filter the data for the specific lat and lon
    else:
        df_scaled = df_scaled.loc[(df_scaled["lat"] == lat) & (df_scaled["lon"] == lon)]


    # create an index based on the lookback period, so we can dynamically the data, using the lookback period
    # we do this as we want our test data to start at 2018, but we need lookback timesteps before to model the first timestep in 2018
    first_index_2017 = df_scaled[df_scaled["time"].dt.year == 2017].index[0]
    val_end_index =  first_index_2017 + look_back

    # split the data into training, validation and test data
    train_data = df_scaled[df_scaled["time"].dt.year <= 2014]
    
    val_data = df_scaled[
        (df_scaled["time"].dt.year == 2015) | 
        (df_scaled["time"].dt.year == 2016) | 
        ((df_scaled["time"].dt.year == 2017) & (df_scaled.index < val_end_index))
        ]

    test_data = df_scaled[
        (df_scaled.index >= val_end_index) |
        (df_scaled["time"].dt.year >= 2018)
        ]

    # drop features not wanted for modelling
    train = train_data.drop(columns=["time", "lat", "lon"])
    val = val_data.drop(columns=["time", "lat", "lon"])
    test = test_data.drop(columns=["time", "lat", "lon"])


    # Create modelling samples, either with or without autoregressive component
    trainX, trainY = convert_to_matrix(train, look_back, target_col, autoregressive=autoregressive)
    valX, valY = convert_to_matrix(val, look_back, target_col, autoregressive=autoregressive)
    testX, testY = convert_to_matrix(test, look_back, target_col, autoregressive=autoregressive)

    # Reshape the data for LSTM model
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    valX = np.reshape(valX, (valX.shape[0], valX.shape[1], valX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    # Get a time index for the test data (used for plotting)
    test_index = sorted(list(set(test_data.time)))

    # Return the data
    if global_model:
        return trainX, trainY, valX, valY, testX, testY, test_index, pixel_indices
    else:
        return trainX, trainY, valX, valY, testX, testY, test_index




########################################################
# Modelling
########################################################

############  Create LSTM model structure ############
def create_lstm_model(look_back, features, units_lstm=50, learning_rate=0.001, dropout_rate=0.2, num_lstm_layers=1, activation='relu'):
    """
    Create an LSTM model with the specified hyperparameters.
    
    Parameters:
    look_back (int): The number of previous time steps to use as input.
    features (int): The number of features in the input data.
    units_lstm (int): Number of units in the LSTM layer(s).
    activation (str): Activation function to use.
    learning_rate (float): Learning rate for the optimizer.
    dropout_rate (float): Dropout rate to use after LSTM layers.
    num_lstm_layers (int): Number of LSTM layers (1 or 2).

    Returns:
    model (Sequential): The compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(look_back, features)))

    if num_lstm_layers == 1:
        model.add(LSTM(units_lstm, activation=activation, dropout=dropout_rate, recurrent_dropout = dropout_rate))

    elif num_lstm_layers == 2:
        
        model.add(LSTM(units_lstm, activation=activation, return_sequences=True, dropout=dropout_rate, recurrent_dropout = dropout_rate))
       
        model.add(LSTM(units_lstm, activation=activation))

    elif num_lstm_layers == 3:
    
        model.add(LSTM(units_lstm, activation=activation, return_sequences=True, dropout=dropout_rate, recurrent_dropout = dropout_rate))
        
        model.add(LSTM(units_lstm, activation=activation, return_sequences=True, dropout=dropout_rate, recurrent_dropout = dropout_rate))
        
        model.add(LSTM(units_lstm, activation=activation, dropout=dropout_rate, recurrent_dropout = dropout_rate))



    model.add(Dense(1, activation='linear'))

    opt = Adam(learning_rate=learning_rate)
  
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model


# Function to create a KerasRegressor for GridSearchCV
def create_keras_regressor(look_back, features, units_lstm=50, learning_rate=0.001, dropout_rate=0.2, num_lstm_layers=1,  activation='relu', optimizer='adam'):
    return KerasRegressor(
        model=create_lstm_model,
        look_back=look_back,
        features = features,
        units_lstm=units_lstm, 
        learning_rate=learning_rate, 
        dropout_rate=dropout_rate, 
        num_lstm_layers=num_lstm_layers,  
        activation=activation, 
        optimizer=optimizer,
        verbose = 0
    )


############ Function to perform grid search cv ############

def perform_grid_search(trainX, trainY, look_back, param_grid, epochs, batch_size, cv):
    """
    Perform grid search to find the best hyperparameters for the LSTM model.

    Parameters:
    - trainX: Training features.
    - trainY: Training labels.
    - look_back: Number of previous time steps to consider for prediction.
    - param_grid: Grid of hyperparameters for the grid search.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - cv: Cross-validation splitting strategy.

    Returns:
    - best_params: Best hyperparameters found by the grid search.
    """
    # Get the number of features
    features = trainX.shape[2]

    # Create a KerasRegressor
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

    # Return the best parameters from the grid search
    return lstm_grid_search.best_params_




# Iterative prediction and substitution (in the case of autoregressive model, otherwise just predict)
def predict_replace(model, X_test, autoregressive = True):
    """
    Generates predictions and updates the test set input for iterative forecasting.

    Parameters:
    model (keras.Model): Trained LSTM model.
    X_test (array): Test data to predict.

    Returns:
    np.array: Array of forecasted values.
    """
    forecasts = []
    
    # sequentially replace shifted sif data (in X_test) by forecasts 
    # after modelling replace according value in X_test with prediction and give all values shifted by 1 timestep to the next sequence.
    
    if autoregressive:
        for i in range(len(X_test)):
            forecast = model.predict(X_test[i].reshape(1, look_back, -1), verbose=0)
            forecasts.append(forecast[0][0])
            if i < len(X_test) - 1:
                X_test[i + 1, :-1, -1] = X_test[i + 1, 1:, -1]
                X_test[i + 1, -1, -1] = forecast[0][0]
    
    else:
        for i in range(len(X_test)):
            forecast = model.predict(X_test[i].reshape(1, look_back, -1), verbose=0)
            forecasts.append(forecast[0][0])

    forecasts_array = np.array(forecasts)


    return forecasts_array


############  Evaluate model ############
def evaluate_model(trainX, trainY, valX, valY, testX, testY, look_back, features, best_params, scalar_y, auto_regressive):
    """
    Train and evaluate the LSTM model with the best hyperparameters.

    Parameters:
    - trainX: Training features.
    - trainY: Training labels.
    - valX: Validation features.
    - valY: Validation labels.
    - testX: Testing features.
    - testY: Testing labels.
    - look_back: Number of previous time steps to consider for prediction.
    - features: Number of features in the input data.
    - best_params: Best hyperparameters found by the grid search.
    - scalar_y: Scaler for the output variable.
    - auto_regressive: Boolean indicating if the model is autoregressive.

    Returns:
    - model_results: Dictionary containing the evaluation results and model history.
    """

    # Create LSTM model with the best hyperparameters
    lstm_model = create_lstm_model(
        look_back=look_back,
        features=features,
        units_lstm=best_params['units_lstm'],
        activation=best_params['activation'],
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate'],
        num_lstm_layers=best_params['num_lstm_layers']
    )

    # Fit the model
    history = lstm_model.fit(
        trainX,
        trainY,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        validation_data=(valX, valY)
    )

    # Predict - with replacement of shifted target_variable in predictor set in case of an autoregressive model
    forecasts = predict_replace(lstm_model, testX, autoregressive=auto_regressive)

    # Rescale the data before evaluation
    testY_rescaled = scalar_y.inverse_transform(pd.DataFrame(testY))
    forecasts_rescaled = scalar_y.inverse_transform(pd.DataFrame(forecasts))

    # Evaluate model performance
    rmse = root_mean_squared_error(testY_rescaled, forecasts_rescaled)
    mae = mean_absolute_error(testY_rescaled, forecasts_rescaled)

    # Return the evaluation results and model history
    return {
        "best_params": best_params,
        "look_back": look_back,
        "evaluation": {"mae": mae, "rmse": rmse},
        "results": {"true_values": testY_rescaled.tolist(), "predicted_values": forecasts_rescaled.tolist()},
        "history": history.history
    }



############  Writing results ############

# Convert the model results to a serializable format so its writeable as a json
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

# Save the model evaluation results to a JSON file
def save_results(output_data, look_back, global_model=False, auto_regressive=False):
    """
    Save the model evaluation results to a JSON file.

    Parameters:
    - output_data: Dictionary containing the model results.
    - look_back: Number of previous time steps to consider for prediction.
    - lat: Latitude (for local models).
    - lon: Longitude (for local models).
    - global_model: Boolean indicating if the model is a global model.
    """

    # Convert the entire data dictionary to a serializable format
    output_data_serializable = {str(k): convert_to_serializable(v) for k, v in output_data.items()}

    # Construct the output file path
    folder_name_json = os.path.join("results", f"result_l{look_back}")
    os.makedirs("results", exist_ok=True)
    os.makedirs(folder_name_json, exist_ok=True)

        
    

    # Determine the file name based on whether the model is global or local and autoregressive or not
    auto_string = "auto" if auto_regressive else "noauto"
    glob_string = "global" if global_model else "local"
    file_name_json = f"results_{glob_string}_{auto_string}.json"
    
    output_json_file = os.path.join(folder_name_json, file_name_json)

    # Write the results to the JSON file
    with open(output_json_file, "w") as file:
        json.dump(output_data_serializable, file, indent=4)
    
    logging.info(f"Results and evaluation written to: {output_json_file}")


############  Full modelling function ############
# Function to train and evaluate global or local LSTM models with or without autoregressive component 
def full_modelling(df_scaled, look_back, lat_lon_pairs, param_grid, scalar_y,
              epochs=100, 
              batch_size=32, 
              cv=TimeSeriesSplit(n_splits=3),
              auto_regressive=False,
              global_model=False,
              lat=None, lon=None, 
              subset = False, n_subset = None):
    """
    Function to train and evaluate global or local LSTM models with or without autoregressive component.

    Parameters:
    - df_scaled: Scaled input dataframe.
    - look_back: Number of previous time steps to consider for prediction.
    - lat_lon_pairs: List of (lat, lon) tuples for local models.
    - param_grid: Grid of hyperparameters for the grid search.
    - scalar_y: Scaler for the output variable.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - cv: Cross-validation splitting strategy.
    - auto_regressive: Boolean indicating if the model is autoregressive.
    - global_model: Boolean indicating if the model is a global model.
    - lat: Latitude (for local models).
    - lon: Longitude (for local models).
    """

    output_data = {}

    if global_model:
        # Training a global model
        logging.info("Starting Global Model Training")

        # Split data for the global model
        trainX, trainY, valX, valY, testX, testY, test_index, pixel_indices = split_data(
            df_scaled, look_back=look_back, lat_lon_pairs=lat_lon_pairs,
            auto_regressive=auto_regressive, global_model=global_model
        )

        # Perform grid search to find the best hyperparameters
        best_params = perform_grid_search(trainX, trainY, look_back, param_grid, epochs, batch_size, cv)

        # Train and evaluate the model
        model_results = evaluate_model(trainX, trainY, valX, valY, testX, testY,
                                        look_back, trainX.shape[2], 
                                        best_params, scalar_y, 
                                        auto_regressive)

        # Store the results
        output_data = model_results

    else:
        
        # if subset, draw 10 random lat_lon_pairs with the same seed
        if subset:
            lat_lon_pairs = lat_lon_pairs.sample(n=n_subset, random_state=42)

        # Training local models for each pixel
        for i, (lat, lon) in enumerate(lat_lon_pairs):

            logging.info(f"Starting Model Training for \n lat: {lat}\n lon: {lon}")

            # Split data for the specific latitude and longitude
            trainX, trainY, valX, valY, testX, testY, test_index = split_data(
                df_scaled, look_back=look_back, lat_lon_pairs = lat_lon_pairs, lat=lat, lon=lon,  
                auto_regressive=auto_regressive, global_model=global_model
            )

            # Perform grid search to find the best hyperparameters
            best_params = perform_grid_search(trainX, trainY, look_back, param_grid, epochs, batch_size, cv)

            # Train and evaluate the model
            model_results = evaluate_model(trainX, trainY, valX, valY, testX, testY,
                                           look_back, trainX.shape[2], 
                                           best_params, scalar_y, auto_regressive)

            # Store the results for the specific latitude and longitude
            output_data[(lat, lon)] = model_results

            logging.info(f"Completed {i + 1}/{len(lat_lon_pairs)}")
            print(f"Completed {i + 1}/{len(lat_lon_pairs)}")

            logging.info(100*"-")

    logging.info(100*"-")
    logging.info(100*"-")

    return output_data


    # Save the results to a JSON file, in a folder named after the lookback period


############  Plotting ############

# Function to plot predicted vs. actual values with MSE in subplots
def plot_multiple_results(results, evaluation, lat_lon_pairs, test_index):
    num_plots = len(results)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            testY, forecasts = [results["true_values"]][i], [results["predicted_values"]][i]
            mae, rmse = [evaluation["mae"]][i], [evaluation["rmse"]][i]
            lat, lon = lat_lon_pairs.iloc[i]
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

