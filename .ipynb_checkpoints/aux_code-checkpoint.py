import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np

#--------- FUNCTION FOR CREATING FOLDER

def create_folder(folder_path):
    """
    This function checks if the folder defined by "folder_path" already exists. If not, it will create the folder
    
    Inputs:
        folder_path: string
                     Path of the folder to be created
    """
    
    if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)

#--------- DEFINE MODEL ARCHITECTURE

def define_mlp():
    """
    This function defines a multilayer perceptron with:
    - 9 inputs
    - 3 hidden layers with 100 neurons each. 
      The activation function of each neuron is ReLU. Dropout is added after each layer
    - 2 outputs
    """
    model = models.Sequential([
        layers.Flatten(input_shape=(9,)),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(2)
    ])
    return model

#------- NORMALIZED DATASET

def normalize_dataset(x_nn, y_nn, y_nn_norm_factor=4000):
    """
    This function normalizes the data and the corresponding labels to be used for training the MLP model
    
    Inputs:
    
        x_nn: numpy array of size N_EXAMPLES x 9
              Matrix containing the CFL data that are given as input to the MLP
        
        y_nn: numpy array of size N_EXAMPLES x 2
              Matrix containing the X and Y coordinates of the flare locations
              
    Optional:
    
        y_nn_norm_factor: float
                          Factor to be used for scaling the X and Y coordinates of the flare locations
              
    Returns:
        x_nn_norm: numpy array of size N_EXAMPLES x 9
                   Matrix containing the normalized CFL data that are given as input to the MLP. 
                   Each row of the input 'x_nn' matrix are is divided by the corresponding maximum value
                   
        y_nn_norm: numpy array of size N_EXAMPLES x 2
                   Matrix containing the X and Y coordinates of the flare location. 
                   Both the X and the Y coordinates are divided by 'y_nn_norm_factor'
    """
    
    max_x_nn  = np.repeat(np.expand_dims(x_nn.max(axis=1), axis=1), 9, axis=1)
    x_nn_norm = x_nn / max_x_nn # Divide each row by the corresponding maximum value

    y_nn_norm = y_nn / y_nn_norm_factor
    
    return x_nn_norm, y_nn_norm

def normalize_quantized_dataset(x_nn, SCALE=11):
    """
    This function normalizes the data to be used for predictions with the quantized MLP model.
    
    Inputs:
    
        x_nn: numpy integer array of size (N_EXAMPLES, 9). 
              Matrix containing the CFL data that are given as input to the MLP (which have been cast as integers)
        
    Optional:
        SCALE: int
               Factor to be used for scaling the input values, which are multiplied by 2**SCALE. Default, 11
              
    Returns:
        x_nn_norm: numpy integer array of size (N_EXAMPLES,9).
                   Matrix containing the normalized CFL data that are given as input to the MLP. 
                   Each row of the input 'x_nn' matrix is multiplied by 2**SCALE and then divided by the original maximum value
        
    """
    
    max_x_nn  = np.repeat(np.expand_dims(x_nn.max(axis=1), axis=1), 9, axis=1)
    
    max_nn_input    = np.max(x_nn)
    x_nn_norm = (x_nn * (2 ** SCALE)) // max_x_nn
    
    return x_nn_norm

#------------- DEFINE QUANTIZED NETWORK

def nn_model(inputs, 
             weightsI1, bias1,
             weights12, bias2,
             weights23, bias3,
             weights3O, biasO,
             SCALE = 11,
             int_type=np.int32):
    """
    This function provides predictions of the quantized MLP for a single input data. 
    The input data must be an integer array which has been appropriately normalized with the function 'normalize_quantized_dataset' using the same scale factor.
    
    Inputs:
    
        x_nn: numpy integer array of size (9,). 
              Matrix containing the CFL data that are given as input to the MLP. 
              This array must be normalized by means with the function 'normalize_quantized_dataset'.
              
        weightsI1: numpy integer array of size (100,9).
                   Quantized weigth values of the MLP input layer
                   
        bias1: numpy integer array of size (100,)
               Quantized values of the MLP input layer bias
               
        weights12: numpy integer array of size (100,100).
                   Quantized weigth values of the MLP first hidden layer
                   
        bias2: numpy integer array of size (100,)
               Quantized values of the MLP first hidden layer bias
               
        weights23: numpy integer array of size (100,100).
                   Quantized weigth values of the MLP second hidden layer
                   
        bias3: numpy integer array of size (100,)
               Quantized values of the MLP second hidden layer bias
               
        weights3O: numpy integer array of size (100,2).
                   Quantized weigth values of the MLP output layer
                   
        biasO: numpy integer array of size (100,)
               Quantized values of the MLP output layer bias
    
    Optional:
    
        SCALE: int
               Factor to be used for scaling the weight and bias values, which are multiplied by 2**SCALE. 
               Default, 11 
               
        int_type: type
                  Integer type to be used for performing the integer multiplications. Default, np.int32
    
    Output:
        outputs: numpy integer array of size (2,).
                 Quantized MLP flare location prediction, which is scaled by 2**SCALE.
    
    """
    # Layer 1
    hidden1 = np.zeros(100, dtype=int_type)
    for h1 in range(100):
        
        sum_ = 0
        for i in range(9):
            
            sum_ += inputs[i] * weightsI1[i][h1]
        
        sum_ = ((sum_ + 2**(SCALE-1)) >> SCALE) + bias1[h1]
        hidden1[h1] = max(0, sum_)  # ReLU activation

    # Layer 2
    hidden2 = np.zeros(100, dtype=int_type)
    for h2 in range(100):
        
        sum_ = 0
        for h1 in range(100):
            
            sum_ += hidden1[h1] * weights12[h1][h2]
        
        sum_ = ((sum_ + 2**(SCALE-1)) >> SCALE) + bias2[h2]
        hidden2[h2] = max(0, sum_)  # ReLU activation

    # Layer 3
    hidden3 = np.zeros(100, dtype=int_type)
    for h3 in range(100):
       
        sum_ = 0
        for h2 in range(100):
            
            sum_ += hidden2[h2] * weights23[h2][h3]
        
        sum_ = ((sum_ + 2**(SCALE-1)) >> SCALE) + bias3[h3]
        hidden3[h3] = max(0, sum_)  # ReLU activation

    # Output layer
    outputs = np.zeros(2, dtype=int_type)
    for o in range(2):
        
        sum_ = 0
        for h3 in range(100):
            
            sum_ += hidden3[h3] * weights3O[h3][o]
        
        sum_ = ((sum_ + 2**(SCALE-1)) >> SCALE) + biasO[o]
        outputs[o] = sum_

    return outputs