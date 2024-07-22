from scripts.utils import create_cube_subset
from scripts.load_aux_data import load_aux_data
from scripts.preprocess import preprocess
from scripts.base_analysis import base_analysis









def main():
    
    data_path = "data"
    
    os.makedirs(data_path, exist_ok=True)

    # Create a subset of the Earth System Data Cube, containing only relevant variables and the desired spatial and temporal extent
    cube_subset = create_cube_subset()

    # Download auxiliary data (Germany border, Corine landcover data, sample tif)
    germany_gpd, corine_file_path, sample_path = load_aux_data(data_path, cube_subset, download = True)

    # Crop the cube to the extent of Germany and mask it with the Corine landcover data (50% forest cover)
    cube_subset_crop = preprocess(cube_subset, germany_gpd, corine_file_path, sample_path, data_path, all_touched = True, write = True)

    # Calculate the temporal changes in the variables 
    changes = base_analysis(cube_subset_crop, years=[2018, 2019], detrend_data=False)



if __name__ == "__main__":
    main()