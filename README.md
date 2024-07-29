# Forecasting SIF for German Forest after the Heatwave in 2018 using Deep Learning

## Table of Contents
- [Forecasting SIF for German Forest after the Heatwave in 2018 using Deep Learning](#forecasting-sif-for-german-forest-after-the-heatwave-in-2018-using-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Book](#project-book)
  - [Structure](#structure)
    - [Scripts:](#scripts)
    - [Notebooks:](#notebooks)
  - [Reproducability](#reproducability)
    - [Prerequisites](#prerequisites)
    - [Usage](#usage)
  - [License](#license)
  - [Contact](#contact)
  - [Authors \& Acknowledgements](#authors--acknowledgements)

## Introduction
This repository is for the project group "LSTM SIF," which is part of the course "Spatio-Temporal Data Analysis" taught by Dr. Guido Kraemer at Leipzig University. The code developed in this project also serves as the foundation for the final paper titled "Forecasting Solar Induced Fluorescence for German Forests after the Heatwave in 2018 using Deep Learning," which is part of the course "Scientific Writing and Publishing" led by Prof. Miguel Mahecha at the Institute of Earth System Data Science and Remote Sensing at Leipzig University during the summer semester of 2024.



**Abstract**:   
German Forest have faced a strong dieback in the last years. Climate extremes like the Heatwave in 2018
are regarded as one of the major drivers for this process. Remote Sensing data can be used to effectively
estimate vegetation variables over large areas. While most commonly spectral indices like the NDVI are
applied, plant traits like the sun-induced fluorescence (SIF) provide a more direct link to the physiological processes in vegetation. SIF is linked to photosynthetic activity and it reacts near-instantaneously to changes in the environment e.g. light or temperature. So far there is a lack of reliable forecasts of vegetation variables for German forests under varying climatic conditions. Particularly for extreme weather events forecasts were so far not successful. In this study, we use a Long Short-Term Memory (LSTM) neural network to forecast SIF for German forests, with ERA5 climate data as predictors. Data of the years 2018 and 2019 is used to evaluate the models performance during and after the heatwave in Germany. We show that the model does very well in capturing the general temporal dynamics of SIF in response to climatic variations. However at timesteps with particularly steep increases or decreases the model did not capture the strong response of the vegetation. Here more research is needed. Still the results show the potential of SIF for ecosystem analysis as well as LSTM for modelling vegetation dynamics based on climatic drivers. Reliable SIF forecasts under differing climatic conditions are a valuable tool to asses potential future impact of climate change on German forests. This can lead to more informed decision-making and better mitigation of these challenges.


## Project Book

If you are not interested in reporduction of the analysis you should simply download the sif_prediction_book.zip. This file contains the rendered book of the analysis, walking you through the full analysis.

If you want to reproduce the analysis go to (Reproducability)[#reproducability].


## Structure

├── _book                                  : Rendered files for quarto book   
├── data                                   : data is not included in this repository
│   ├── cubes                              : Data cubes used in the analysis
│   ├── germany_shape                      : Shapefile of Germany
│   └── landcover                          : Corine landcover data  
├── environment.yml                        : Yml file to create conda environment
├── _extensions                            : Quarto extensions 
├── index.qmd                              : Quarto file to create the quarto book
├── LICENSE                                : License file
├── main_workflow.ipynb                    : Main workflow notebook
├── quarto_files                           : QMD files to create quarto book
├── _quarto.yml                            : Quarto file to create the quarto book
├── README.md                              : Readme file
├── results                                : Results of the analysis
│   ├── figures                            : Figures created during the analysis
│   ├── logs                               : Logs created during final modelling 
│   └── modelling                          : Results of the modelling
└── scripts                                : Scripts to run the analysis

### Scripts: 

- s01_load_aux_data.py: Download all auxillary data
- s02_cube_preprocessing.py:  Preprocessing the Earth System Data Cube 
- s03_base_analysis.py: Analyzing Sif Change in 2018 compared to the preceeding years
- s04_test_modelling.py: Testing different LSTM modelling structures to model SIF based on climatic drivers
- s05_modelling.py: Actual Modelling of all timeseries based on s04_test_modelling.py

### Notebooks:
- main_workflow.ipynb: Main notebook to walk through the analysis, as well as plotting and compiling the results

## Reproducability

THis analysis was done using Python 3.11. 

### Prerequisites

To get a copy of the project and make it run on Your local machine, use following command: 

```bash
  git clone https://git.sc.uni-leipzig.de/ss2024-12-geo-m-ds02/sif-prediction
```

Create a python environment using conda and the environment.yml file provided in the repository. 

```bash
  conda env create --file environment.yml
```

### Usage

The main part of the analysis is done by running the scripts in the **scripts** folder. If you want to reproduce the whole analysis you can run:

```bash
  conda activate sif_env

  python3 scripts/s01_load_aux_data.py
  python3 scripts/s02_cube_preprocessing.py
  python3 scripts/s03_base_analysis.py
  python3 scripts/s04_test_modelling.py
  python3 scripts/s05_modelling.py

```

After that you can run main_workflow.ipynb to visualize and compile the results. The notebook "main_workflow.ipynb" also walks you through the general workflow of the analysis. 

If you do not want to reproduce the results, but only want to walk through main_workflow.ipynb to understand the workflow, you can simply run scripts 01 and 02. 

```bash
  conda activate sif_env

  python scripts/s01_load_aux_data.py
  python scripts/s02_cube_preprocessing.py

```


After that you find all data and results in the respective folders.

As the data is not contained in this repository, you might only want to create the data, to be able to use the main_workflow.ipynb. To do so simply run scripts 01 and 02. 

```bash
  conda activate sif_env

  python scripts/s01_load_aux_data.py
  python scripts/s02_cube_preprocessing.py

```



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
