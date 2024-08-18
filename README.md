# Chest X-Ray Detection Project

## Dataset

This project uses the [Tuberculosis Chest X-Ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) available on Kaggle. 

### Dataset Structure

The dataset consists of two main folders:
- `Normal/` - Contains images of normal chest X-rays.
- `Tuberculosis/` - Contains images of tuberculosis chest X-rays.

### Setup Instructions

1. **Download the Dataset:**
   - Go to the [dataset page on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
   - Download the dataset and extract the files.

2. **Prepare the Data:**
   - Create a directory named `Data` in the current directory if it does not already exist.
   - Move the extracted folders `Normal` and `Tuberculosis` into `Data/TB_Chest_Radiography_Database/`.
   - The final directory structure should look like this:
     ```
     Data/
     └── TB_Chest_Radiography_Database/
         ├── Normal/
         └── Tuberculosis/
     ```

3. **Save Metadata and Readme:**
   - Ensure that the `Tuberculosis.metadata`, `Normal.metadata`, and `readme.md` are in the `TB_Chest_Radiography_Database/` directory. These files are not required by the code but should be kept for reference.

## Running the Code

1. **Training the Model:**
   - Run the training script to train the model and save it to the specified directory.
     ```bash
     python train.py
     ```
   - The trained model will be saved in `models/final_model`.

2. **Testing the Model:**
   - Run the testing script to load the trained model and evaluate its performance on the test set.
     ```bash
     python test.py
     ```
   - This will load the model from `models/final_model` and output the results.

3. **Evaluating the Model:**
   - Run the evaluation script to obtain comprehensive evaluation metrics.
     ```bash
     python evaluation.py
     ```
   - This will provide detailed results including accuracy, loss, and additional metrics.
