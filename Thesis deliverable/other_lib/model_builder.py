import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import Input, Concatenate, Conv1D, Conv2D, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, MaxPooling1D, Flatten, Dense, Embedding, Dropout, MaxPooling2D, Lambda, BatchNormalization, Activation
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers
from tensorflow.keras import layers, backend, Model, regularizers
from transformer_lib import constants
from transformer_lib.data import loader
from transformer_lib.models import transformer
from transformer_lib.models.transformer import TokenAndPositionEmbedding
from transformer_lib.models.transformer import TransformerBlock
from other_lib import globalvar
from other_lib.auk_score import AUK

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_transformer_from_h5(x_train, x_test, y_train, y_test, binary, model_name, param_dict):
    
    #include dict for embedding in tokenized datasets
    x_dict= {"[PAD]": 0, "[UNK]": 1, "arrive_cathlab": 2, "start_operation": 3, "end_operation": 4, "leave_cathlab": 5, "prepare": 6, 
             "start_introduction": 7, "end_introduction": 8, "cancellation": 9, "scheduled": 10, "waitfor_schedule": 11, "admission": 12, 
             "discharge": 13, "recovery": 14, "restart_noac": 15, "start_ac": 16, "stop_ac": 17, "paracetamol": 18, "measurebps": 19, 
             "measuretemps": 20, "test_hemoglobine": 21, "test_egfr": 22, "test_inr": 23, "test_trombocyten": 24}
    additional_features = ['MedicationCode_B01AA04', 'MedicationCode_B01AA07', 'MedicationCode_B01AE07', 'MedicationCode_B01AF01', 
                           'MedicationCode_B01AF02', 'MedicationCode_B01AF03', 'MedicationCode_N02AJ13', 'MedicationCode_N02BE01',
                           'PlannedDuration', 'Duration', 'MedicationType', 'NOAC', 'MedicationStatus', 'temperature', 
                           'bloodPressure', 'Test_Hemoglobine', 'Test_eGFR', 'Test_INR', 'Test_Trombocyten']
    vocab_size = len(x_dict) #number of different values, in this case 24 tokens
    max_case_length = x_train.shape[1] #Used for token embedding, make this all columns except traceID
    num_heads = param_dict['num_heads'] # number of attention heads in the transformer block
    ff_dim = 64 #feedforward dimension, number of nodes for the dense layer
    embed_dim = 36 #embedding dimension size
    
    x_train_list = [x_train]
    x_test_list = [x_test]
    
    if 'tokenized' in model_name and 'additional' in model_name:
        x_token_train = np.expand_dims(x_train[x_train.columns.difference(additional_features)], -1)
        x_additional_train = x_train[additional_features]
        x_token_test = np.expand_dims(x_test[x_test.columns.difference(additional_features)], -1)
        x_additional_test = x_test[additional_features]
        
        x_train_list = [x_token_train, x_additional_train]
        x_test_list = [x_token_test, x_additional_test]
        
    #for tokenized encodings an additional pass through the TokenAndPositionEmbedding layer is required
    if 'tokenized' in model_name:
        if 'additional' in model_name: #if dataset = tokenized + additional, the additional data needs to be split and concatenated later
            
            tokenized_input = layers.Input(shape=(x_token_train.shape[1],)) #input layer for tokenized data
            additional_input = layers.Input(shape=(x_additional_train.shape[1],)) #input layer for additional data
            inputs = [tokenized_input, additional_input]
            
            x = TokenAndPositionEmbedding(x_token_train.shape[1], vocab_size, embed_dim)(tokenized_input)
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Concatenate(axis=1)([x, additional_input])
        else:
            inputs = layers.Input(shape=(max_case_length,))
            x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            x = layers.GlobalAveragePooling1D()(x)

    else:
        embed_dim = max_case_length
        inputs = layers.Input(shape=(max_case_length, 1)) #add a ',1' to account for additional dimension
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(inputs)
        x = layers.GlobalAveragePooling1D()(x)
    
    #x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(param_dict['Dropout'])(x)
    x = layers.Dense(param_dict['Dense'], activation="relu")(x)
    x = layers.Dropout(param_dict['Dropout_1'])(x)

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=param_dict['learning_rate'], clipnorm=1.)   
    
    #determine output shape based on prediction task, either for binary/length of stay prediction
    if binary:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', globalvar.f1, globalvar.precision, globalvar.recall, globalvar.auc])
        return model, x_train_list, x_test_list
    else:
        outputs = layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse', 'mape'])
        return model, x_train_list, x_test_list
    
    
def rebuild_di_mauro_model(x_train, x_test, y_train, y_test, binary, param_dict):
    
    #create input layer
    inputs = [] #create empty list to store the input layers in, will be updated and used when compiling the model
    input_layer = Input(shape=(x_train.shape[1], 1)) #input layer, width of data = timesteps with each timestep having 1 feature (1 value per column)
    inputs = [input_layer]
    
    #add first inception module
    filters = []
    for i in range(param_dict['range']): #number of different conv modules in the inception layer
        filters.append(Conv1D(filters=32, strides=1, kernel_size=1+i, activation='relu', padding='same')(input_layer)) #add the conv layers of different sizes
    filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(input_layer)) #add the max pool layer
    concat_layer = Concatenate(axis=2)(filters) #concatenate the output of the different conv modules and max pool layer to get output of inception module
    
    for m in range(param_dict['range_1']): #number of inception modules you want to stack on top of the first one (for a total of either 1, 2 or 3)
        filters = []
        for i in range(param_dict['range_2']): #number of different conv modules in the inception layer
            filters.append(Conv1D(filters=32, strides=1, kernel_size=1+i, activation='relu', padding='same')(concat_layer)) #add the conv layers of different sizes
        filters.append(MaxPooling1D(pool_size=3, strides=1, padding='same')(concat_layer)) #add the max pool layer
        concat_layer = Concatenate(axis=2)(filters) #concatenate the output of the different conv modules and max pool layer to get output of inception module
    
    reg_max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=(x_train.shape[1]), strides=1, padding='valid')(concat_layer)
    squeeze_layer = tf.keras.layers.Lambda(lambda s: backend.squeeze(s, 1))(reg_max_pool_1d)
    
    choiceval = param_dict['choiceval']
    if choiceval == 'adam':
        optim = tf.keras.optimizers.Adam(learning_rate=param_dict['learning_rate'], clipnorm=1.)
    elif choiceval == 'rmsprop':
        optim = rmsprop = tf.keras.optimizers.RMSprop(learning_rate=param_dict['learning_rate_1'], clipnorm=1.)
    else:
        optim = tf.keras.optimizers.SGD(learning_rate=param_dict['learning_rate_2'], clipnorm=1.)
    
    #determine output shape based on prediction task, either for binary/length of stay prediction
    if binary:
        output_layer = Dense(1, activation='sigmoid')(squeeze_layer)
        model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer=optim, loss='binary_crossentropy',
                      metrics=['accuracy', globalvar.f1, globalvar.precision, globalvar.recall, globalvar.auc])
        return model
    else:
        output_layer = Dense(1, activation='linear')(squeeze_layer)
        model = Model(inputs=inputs, outputs=output_layer)
        model.compile(optimizer=optim, loss='mae', metrics=['mae', 'mse', 'mape'])
        return model
    
def rebuild_pasqua_model(x_train, x_test, y_train, y_test, binary, param_dict):
    
    #build sequential model
    model = Sequential()
    input_shape = (x_train.shape[1], x_test.shape[1], 1)
    model.add(Conv2D(param_dict['Conv2D'], (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(param_dict['l2'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(param_dict['Conv2D_1'], (4, 4), padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(param_dict['l2_1'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #replacement layers for globalmaxpooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Lambda(lambda s: backend.squeeze(s, 1)))
    model.add(Lambda(lambda s: backend.squeeze(s, 1)))
    #model.add(GlobalMaxPooling2D())
    
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=param_dict['learning_rate'], clipnorm=1.)   

    #add output layer based on the prediction type        
    if binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', globalvar.f1, globalvar.precision, globalvar.recall, globalvar.auc])
    else:
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse', 'mape'])
        
    return model


def prepare_dataset_for_model_shapeley(file_location, model_type, shuffle_split=True):
    
    df = pd.read_csv(file_location)
    model_name = file_location.split("\\")[-1:][0].split('.')[0] #get filename (without.csv)
    binary = False if 'los' in model_name.lower() else True #check if a binary prediction (for paracetamol/cancel datasets) or a regression prediction (for length of stay) is being made
    
    #define label (aka outcome) and prediction data
    y = df['Label'] if 'Label' in df else df['outcome']
    X = df.loc[:, df.columns != 'Label'] if 'Label' in df else df.loc[:, df.columns != 'outcome']
    
    #remove TraceID (aka case_id) from the training and testing data
    if 'TraceID' in X.columns or 'case_id' in X.columns:
        X = X.drop('TraceID', 1) if 'TraceID' in X.columns else X.drop('case_id', 1)
    
    #train/test set split, must be done before scaling to prevent data leakage between train/test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=shuffle_split)
        
    #fill NaN value with mean of training data for both train and test data. Cant do mean per group since many groups have no data at all
    x_train.fillna(x_train.mean(), inplace=True)
    x_test.fillna(x_train.mean(), inplace=True)
    
    #scaling for non-additional features, only on train/test data to prevent data leakage, complete X returned without scaling
    additional_features = ['MedicationCode_B01AA04', 'MedicationCode_B01AA07', 'MedicationCode_B01AE07', 'MedicationCode_B01AF01', 
                           'MedicationCode_B01AF02', 'MedicationCode_B01AF03', 'MedicationCode_N02AJ13', 'MedicationCode_N02BE01',
                           'PlannedDuration', 'Duration', 'MedicationType', 'NOAC', 'MedicationStatus', 'temperature', 
                           'bloodPressure', 'Test_Hemoglobine', 'Test_eGFR', 'Test_INR', 'Test_Trombocyten']

    scaler = StandardScaler()    
    
    if 'tokenized' in model_name and 'transformer' not in model_type: #means all columns need to be encoded, regardless of additional or not
        x_train = pd.DataFrame(scaler.fit_transform(x_train))
        x_test = pd.DataFrame(scaler.fit_transform(x_test))
    elif 'additional' in model_name.lower() and 'ae_agg' not in model_name.lower(): #means only the additionally added columns need to be scaled
        x_train[additional_features] = scaler.fit_transform(x_train[additional_features])
        x_test[additional_features] = scaler.fit_transform(x_test[additional_features])
        
    #For lstm models, the input needs to be 3d instead of 2d. Therefore, add another dimension to the data
    if model_type == 'lstm' or model_type=='transformer' and 'additional' not in model_name.lower():
        x_train = np.expand_dims(x_train, -1)
        x_test= np.expand_dims(x_test, -1) 
    
    return x_train, x_test, y_train, y_test, binary, X, y, model_type