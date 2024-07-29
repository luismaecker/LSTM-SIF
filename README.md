# Forecasting SIF for German Forest after the Heatwave in 2018 using Deep Learning

## Table of Contents
- [Forecasting SIF for German Forest after the Heatwave in 2018 using Deep Learning](#forecasting-sif-for-german-forest-after-the-heatwave-in-2018-using-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Structure](#structure)
  - [Usage](#usage)
  - [License](#license)
  - [Contact](#contact)
  - [Authors \& Acknowledgements](#authors--acknowledgements)

## Introduction
This README is for the project group "LSTM SIF," which is part of the course "Spatio-Temporal Data Analysis" taught by Dr. Guido Kraemer at Leipzig University. The code developed in this project also serves as the foundation for the final paper titled "Forecasting Solar Induced Fluorescence for German Forests after the Heatwave in 2018 using Deep Learning," which is part of the course "Scientific Writing and Publishing" led by Prof. Miguel Mahecha at the Institute of Earth System Data Science and Remote Sensing at Leipzig University during the summer semester of 2024.

**Abstract**:   
German Forest have faced a strong dieback in the last years. Climate extremes like the Heatwave in 2018
are regarded as one of the major drivers for this process. Remote Sensing data can be used to effectively
estimate vegetation variables over large areas. While most commonly spectral indices like the NDVI are
applied, plant traits like the sun-induced fluorescence (SIF) provide a more direct link to the physiological processes in vegetation. SIF is linked to photosynthetic activity and it reacts near-instantaneously to changes in the environment e.g. light or temperature. So far there is a lack of reliable forecasts of vegetation variables for German forests under varying climatic conditions. Particularly for extreme weather events forecasts were so far not successful. In this study, we use a Long Short-Term Memory (LSTM) neural network to forecast SIF for German forests, with ERA5 climate data as predictors. Data of the years 2018 and 2019 is used to evaluate the models performance during and after the heatwave in Germany. We show that the model does very well in capturing the general temporal dynamics of SIF in response to climatic variations. However at timesteps with particularly steep increases or decreases the model did not capture the strong response of the vegetation. Here more research is needed. Still the results show the potential of SIF for ecosystem analysis as well as LSTM for modelling vegetation dynamics based on climatic drivers. Reliable SIF forecasts under differing climatic conditions are a valuable tool to asses potential future impact of climate change on German forests. This can lead to more informed decision-making and better mitigation of these challenges.

## Getting Started
The project was made using Python Version 3.11.8.

### Prerequisites

<!-- To get a copy of the project and make it run on Your local machine, use following command: 
```bash
  git clone https://git.sc.uni-leipzig.de/ss2024-12-geo-m-ds02/sif-prediction
```

Make sure to install all packages needed - found in requirements.txt

Use our conda env, for full reproduction: 
```bash
  conda env create --file environment.yml
```
 -->
Instructions on how to setup your environment to run the code found in this repository.

```bash
# Clone the repository
git clone https://git.sc.uni-leipzig.de/ss2024-12-geo-m-ds02/sif-prediction.git

# Navigate to the project directory
cd sif-prediction

# Create a virtual environment e.g. through conda with Python 3.11
conda create -n sif_env python=3.11 -y 

# Activate the conda environment
conda activate sif_env

# Install necessary packages
pip install -r requirements.txt

```
## Structure

```bash
tree 
```
```bash
├── _book                                  = Rendered files for quarto book   
├── data                                   = Data used in the project  
│   ├── cubes                              = Data cubes used in the analysis
│   ├── germany_shape                      = Shapefile of Germany
│   ├── landcover                          = Corine landcover data  
├── environment.yml                        = Yml file to create conda environment
├── _extensions                            = Quarto extensions 
├── index.qmd                              = Quarto file to create the quarto book
├── LICENSE                                = License file
├── main_workflow.ipynb                    = Main workflow notebook
├── quarto_files                           = qmd files to create quarto book
├── _quarto.yml                            = Quarto file to create the quarto book
├── README.md                              = Readme file
├── results                                = Results of the analysis
│   ├── figures                            = Figures created during the analysis
│   ├── logs                               = Logs created during final modelling 
│   ├── modelling                          = Results of the modelling
└── scripts                                = Scripts to run the analysis
```




## Usage
Within the Repository You can find the **scripts** folder, which guides You through the processing step-by-step. The scripts are numbered in the order they should be run. To run the analysis you should create an 


- s01_load_aux_data.py: Integrates the auxiliary data loading process by initializing GEE, downloading the German border and CORINE data, and creating a SIF sample TIFF, managing file paths and downloads
- s02_cube_preprocessing.py:  Computes the percentage of forest cover within a specified window of land cover data by identifying pixels that match predefined forest classes.Resamples CORINE land cover data to match the resolution and dimensions of a sample SIF raster, calculating forest cover percentages for each resampled cell and returning a flipped array of these percentages. Than it clips a given data cube to the borders of Germany, calculates forest cover percentages over the grid, adds this data to the cube, creates a binary forest cover layer, and optionally writes the processed data to disk.
- s03_base_analysis.py: Plots and saves the time series of SIF data. Calculates the summer mean for each year in the dataset and the changes in SIF compared to the baseline period up to 2017. Returns the summer mean cube, the baseline mean to 2017, and the changes for specified years.
- s04_test_modelling.py: This script will create 4  basic types of models, which are differing whether they are trained global or local models and whether they have an auto regressive component or not. To find the best hyperparameters 3 look back periods are tested and a grid search cross-validation is done. 
- s05_modelling.py: This script runs the model again using the local non auto regressive model, as this was the best performing setup.
The Grid search is done on a reduced Hyperparameter grid, based on the prelimnary analysis. 

## License
This project falls under an MIT License. More Information in the LICENSE file

## Contact
You can reach the authors via Email: 
- Luis Maecker - maecker@studserv.uni-leipzig.de, 
- Imke Ott - imke.ott@studserv.uni-leipzig.de, 3735724
- Moritz Mischi - moritz.mischi@studserv.uni-leipzig.de, 3778634

## Authors & Acknowledgements
The three authors contributed qually to this project. 

This project was made possible by the sponsorship of [ESA NoR](https://eo4society.esa.int/network-of-resources/nor-sponsorship/), which provided server-capacity to process our data. 
