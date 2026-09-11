Project Directory Structure:
project-root/
│
├── PPT                 
├── Video                
├── code/                 
│   ├── main.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── mlmodel_training.py
│   ├── ga_feature_selection.py
│   ├── requirements.txt
│   └── data/             # Dedicated folder for datasets
│        └── <your_dataset_zip_file.zip>
│
└── README.txt            # Project description and instructions

1. PREREQUISITES AND LIBRARIES
Python Version: 3.12 or more

Recommended: Use 64-bit Python for best ML compatibility

Command to install all required libraries:

pip install -r requirements.txt    <---this command should be executed in the working directory of code folder
Requirements include: torch, torchvision, numpy, pandas, scikit-learn, matplotlib, pillow, tqdm

2. DATASETS
   DATASET-1:Gallbladder Diseases Dataset
   Link to Download:https://data.mendeley.com/datasets/r6h24d2d3y/2
   DATASET-2:LC25000
   Link to Download:https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data
   DATASET-3:Plants Type Dataset
   Link to Download:https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets
3. DATA STRUCTURE
This code handles three types of image dataset organizations:

A. Predefined split folders (three or two):
text
dataset/
├── train/
├── validation/   (optional)
└── test/
The pipeline uses these splits as provided.

B. Single folder with class subfolders:
text
dataset/
├── Class1/
├── Class2/
└── Class3/
The code will:

Read all images by class.

Automatically create new folders for train (70%), validation (10%), test (20%) splits.

C. Two folders (train + test):
text
dataset/
├── train/
└── test/
These are used directly. If you need a validation split, run preprocessing to separate a portion from train.

4. How to run:
   ->After installing required libraries as mentioned at the top
    pip install -r requirements.txt(same command as above)
   ->after downloading place dataset zip in data folder and unzip it as mentioned below 
   ->we have to unzip it through this commands:
   For any dataset:
   For windows:
   python3 -m zipfile -e "data/downloaded Dataset.zip" "data/"
   For mac:
   unzip "data/downloaded Dataset.zip" -d "data/"
   Dataset-1:
   For windows and mac:
   python3 -m zipfile -e "data/Gallblader Diseases Dataset.zip" "data/"  <-for outer zip
   
   Get-ChildItem -Path "data\Gallblader Diseases Dataset" -Filter *.zip | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath ($_.FullName -replace '.zip','') -Force
}               <-for inner zip present in main folder
   both the above commands has to be run because main dataset zip contains 9 zip files in it 

   For mac:
   mkdir -p "data"
   unzip "Gallblader Diseases Dataset.zip" -d "data/"
   Verify extraction:
   ls -la "data"
   ls -la "data/Gallblader Diseases Dataset"
   find "data/Gallblader Diseases Dataset" -type f -name "*.zip" -print
   while IFS= read -r z; do d="${z%.zip}"; mkdir -p "$d"; unzip -o "$z" -d "$d"; rm -f "$z"; done < <(find "data/Gallblader Diseases Dataset" -type f -name "*.zip")
   Dataset-2:
   For windows:
   python3 -m zipfile -e "data/archive.zip" "data/"
   For mac:
   unzip "data/archive.zip" -d "data/"
   Dataset-3:
   For windows:
   python3 -m zipfile -e "data/archive.zip" "data/"
   For mac:
   unzip "data/archive.zip" -d "data/"
   ->after unzipping run the following commands to execute the code
   Dataset-1:
   For windows:
   python main.py "data/Gallblader Diseases Dataset"
   For mac:
   python3 main.py "data/Gallblader Diseases Dataset"
   Dataset-2:
   For windows:
   python main.py "data/lung_colon_image_set/Train and Validation Set" "data/lung_colon_image_set/Test Set"
   For mac:
   python3 main.py "data/lung_colon_image_set/Train and Validation Set" "data/lung_colon_image_set/Test Set"
   Dataset-3:
   For windows:
   python main.py data/split_ttv_dataset_type_of_plants/Train_Set_Folder data/split_ttv_dataset_type_of_plants/Validation_Set_Folder data/split_ttv_dataset_type_of_plants/Test_Set_Folder
   For mac:
   python3 main.py data/split_ttv_dataset_type_of_plants/Train_Set_Folder data/split_ttv_dataset_type_of_plants/Validation_Set_Folder data/split_ttv_dataset_type_of_plants/Test_Set_Folder

  ->after running these commands, you will be prompted to enter a prefix for the output result files. This prefix ensures that the files created do not collide with result files from other datasets. The prefix can be any name you choose.
  Prefixes we used:
  Dataset-1:Gallblader
  Dataset-2:Lung
  Dataset-3:Plants
  ->Once you enter your desired prefix, the main project pipeline will execute using that prefix for all its output result files.
   
