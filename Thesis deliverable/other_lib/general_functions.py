"""
This function takes the following arugments:
 - a String value of a file location of a csv file you want to encode
 - a String value of the model type that is being calculated (e.g. lstm, cnn, transformer, logistic)
 
 the function returns the train/test splitted data, whether it is a binary class problem or not, the complete X and y data for cross
 validation purposes and finally the model_type as it is used in other functions aswell
"""

import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from orange_lib import DeepInsight_train_norm


def prepare_dataset_for_model(file_location, model_type):
    
    df = pd.read_csv(file_location)
    model_name = file_location.split("\\")[-1:][0].split('.')[0] #get filename (without.csv)
    binary = False if 'los' in model_name.lower() else True #check if a binary prediction (for paracetamol/cancel datasets) or a regression prediction (for length of stay) is being made
    
    #define label (aka outcome) and prediction data
    y = df['Label'] if 'Label' in df else df['outcome']
    X = df.loc[:, df.columns != 'Label'] if 'Label' in df else df.loc[:, df.columns != 'outcome']
    
    #remove TraceID (aka case_id) from the training and testing data
    if 'TraceID' in X.columns or 'case_id' in X.columns:
        X = X.drop('TraceID', 1) if 'TraceID' in X.columns else X.drop('case_id', 1)
    
    #train/val/test set split, must be done before scaling and upsampling to prevent data leakage between train/test data
    if binary:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

        
    #fill NaN value with mean of training data for both train and test data. Cant do mean per group since many groups have no data at all
    x_train.fillna(x_train.mean(), inplace=True)
    x_val.fillna(x_train.mean(), inplace=True)
    x_test.fillna(x_train.mean(), inplace=True)
    
    #scaling for non-additional features, only on train/test data to prevent data leakage, complete X returned without scaling
    additional_features = ['MedicationCode_B01AA04', 'MedicationCode_B01AA07', 'MedicationCode_B01AE07', 'MedicationCode_B01AF01', 
                           'MedicationCode_B01AF02', 'MedicationCode_B01AF03', 'MedicationCode_N02AJ13', 'MedicationCode_N02BE01',
                           'PlannedDuration', 'Duration', 'MedicationType', 'NOAC', 'MedicationStatus', 'temperature', 
                           'bloodPressure', 'Test_Hemoglobine', 'Test_eGFR', 'Test_INR', 'Test_Trombocyten']

    scaler = StandardScaler()    
    
    if 'tokenized' in model_name and 'transformer' not in model_type: #means all columns need to be encoded, regardless of additional or not
        x_train = pd.DataFrame(scaler.fit_transform(x_train))
        x_val = pd.DataFrame(scaler.fit_transform(x_val))
        x_test = pd.DataFrame(scaler.fit_transform(x_test))
    elif 'additional' in model_name.lower() and 'ae_agg' not in model_name.lower(): #means only the additionally added columns need to be scaled
        x_train[additional_features] = scaler.fit_transform(x_train[additional_features])
        x_val[additional_features] = scaler.fit_transform(x_val[additional_features])
        x_test[additional_features] = scaler.fit_transform(x_test[additional_features])
        
    #oversampling of training data for cancellation data, skip test data (data leakage) and validation (validation needs to be representative of test data)
    if 'can' in model_name:
        oversampler = RandomOverSampler(sampling_strategy='minority')
        x_train, y_train = oversampler.fit_resample(x_train, y_train)
        
    #For lstm models, the input needs to be 3d instead of 2d. Therefore, add another dimension to the data so the data passes correctly
    if model_type == 'lstm' or model_type=='transformer' and 'additional' not in model_name.lower():
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        x_test= np.expand_dims(x_test, -1) 
    
    return x_train, x_val, x_test, y_train, y_val, y_test, binary, X, y, model_type



"""
This function takes a model folder name (like 'di_mauro_cnn') and returns a list of all the csv file locations for that model that have  yet to be encoded
"""
def find_all_csv_locations(model_folder):
    # make a list of all csv files
    data_folder = 'C:\\Users\\20190337\\Downloads\Tracebook_v2 (Projectfolder)\\encoded_logs\\' #define folder where all encoded logs are stored
    file_extension = '*.csv' #define file extension
    all_csv_files = [] #store csv files in this list
    for path, subdir, files in os.walk(data_folder):
        for file in glob(os.path.join(path, file_extension)):
            all_csv_files.append(file)

    #calculate files that are skipped for now to save time
    skip_files = [] 
    filter_list = ['max', '16', '32', '64', 'random', 'test', 'train', 'outcome', 'aggregated']
    for file in all_csv_files:
        for filter_val in filter_list:
            if filter_val in file:
                skip_files.append(file)
    file_locations = [file for file in all_csv_files if file not in skip_files] #keep only the csv files that are not to be skipped

    #check if model results have already been calculated before by comparing .h5 file names to csv file names, if so remove them
    results_folder = 'C:\\Users\\20190337\\Downloads\Tracebook_v2 (Projectfolder)\\model_results\\' + model_folder + '\\'
    results_extension = '*.h5'
    h5_locations = [] #store h5 files in this list
    for path, subdir, files in os.walk(results_folder):
        for file in glob(os.path.join(path, results_extension)):
            if file.endswith('_model-weights.h5'):
                h5_locations.append(file.replace('_model-weights', '')) #remove the '_model-weights' part from the transformer model weights
            else:
                h5_locations.append(file)
    h5_locations_names = [file.split("\\")[-1:][0].split('.')[0] for file in h5_locations]

    #finally, get all the csv locations for datasets that are not to be skipped and have not been tested yet
    file_locations = [file for file in file_locations if file.split("\\")[-1:][0].split('.')[0] not in h5_locations_names]
    
    print('{} csv files left'.format(len(file_locations)))
    
    return file_locations


"""
These two functions below are used for the orange model to convert a trace to pixel values before it passes through the model, one also makes validation split and the other one simply returns a train/test split used for cross validation

"""

def image_encoder_val(x_train, x_val, x_test, y_train, y_val, y_test):
        
    # define parameters for when the data is turned into images
    param = {"Max_A_Size": 10, "Max_B_Size":10, #image size
             "Dynamic_Size": False, 
             'Method': 'tSNE', 
             #"ValidRatio": 0.1, 
             "seed": 42,
             "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
             "mutual_info": True, # Mean or Mutual info when more than one features are located to same pixel
             "hyper_opt_evals": 50, "epoch": 40, "No_0_MI": False,  # True -> Removing 0 mutual info features
             'cut':None}

    #make data dict 
    data = {}
    data = {"Xtrain": x_train.astype(float), "class": 2}
    data["Ytrain"] = y_train
    data["Xval"] = x_val.astype(float)
    data["Yval"] = y_val
    data["Xtest"] = x_test.astype(float)
    data["Ytest"] = y_test

    #transform the log into image data
    x_train_images, x_val_images, x_test_images, y_train, y_val, y_test = DeepInsight_train_norm.train_norm_val(param, data)
    
    return x_train_images, x_val_images, x_test_images


def image_encoder_cv(x_train, x_test, y_train, y_test):
        
    # define parameters for when the data is turned into images
    param = {"Max_A_Size": 10, "Max_B_Size":10, #image size
             "Dynamic_Size": False, 
             'Method': 'tSNE', 
             #"ValidRatio": 0.1, 
             "seed": 42,
             "Mode": "CNN2",  # Mode : CNN_Nature, CNN2
             "mutual_info": True, # Mean or Mutual info when more than one features are located to same pixel
             "hyper_opt_evals": 50, "epoch": 40, "No_0_MI": False,  # True -> Removing 0 mutual info features
             'cut':None}

    #make data dict 
    data = {}
    data = {"Xtrain": x_train.astype(float), "class": 2}
    data["Ytrain"] = y_train
    data["Xtest"] = x_test.astype(float)
    data["Ytest"] = y_test

    #transform the log into image data
    x_train_images, x_test_images, y_train, y_test = DeepInsight_train_norm.train_norm_cv(param, data)
    
    return x_train_images, x_test_images