import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,LSTM,BatchNormalization,MaxPooling2D,Reshape
from keras.utils import to_categorical
import matplotlib.pyplot as plt

'''
CNN
'''
def CNN():
   # Building the CNN model using sequential class
    cnn_model = Sequential()

    # Conv. block 1
    cnn_model.add(Conv2D(filters=25, kernel_size=(5,5), padding='same', activation='elu', input_shape=(400,1,22)))
    cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.6))

    # Conv. block 2
    cnn_model.add(Conv2D(filters=50, kernel_size=(5,5), padding='same', activation='elu'))
    cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.6))

    # Conv. block 3
    cnn_model.add(Conv2D(filters=100, kernel_size=(5,5), padding='same', activation='elu'))
    cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.6))

    # Conv. block 4
    cnn_model.add(Conv2D(filters=200, kernel_size=(5,5), padding='same', activation='elu'))
    cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.6))

    cnn_model.add(Flatten()) # Adding a flattening operation to the output of CNN block
    cnn_model.add(Dense((40))) # FC layer with 100 units
    cnn_model.add(Reshape((40,1))) # Reshape my output of FC layer so that it's compatible
    cnn_model.add(CNN(10, dropout=0.4, recurrent_dropout=0.1, input_shape=(40,1), return_sequences=False))

    # Output layer with Softmax activation 
    cnn_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
    return cnn_model

'''
CNN + LSTM
'''
def hybrid():
   # Building the CNN model using sequential class
    hybrid_cnn_lstm_model = Sequential()

    # Conv. block 1
    hybrid_cnn_lstm_model.add(Conv2D(filters=25, kernel_size=(5,5), padding='same', activation='elu', input_shape=(400,1,22)))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.6))

    # Conv. block 2
    hybrid_cnn_lstm_model.add(Conv2D(filters=50, kernel_size=(5,5), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.6))

    # Conv. block 3
    hybrid_cnn_lstm_model.add(Conv2D(filters=100, kernel_size=(5,5), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.6))

    # Conv. block 4
    hybrid_cnn_lstm_model.add(Conv2D(filters=200, kernel_size=(5,5), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.6))

    # FC+LSTM layers
    hybrid_cnn_lstm_model.add(Flatten()) # Adding a flattening operation to the output of CNN block
    hybrid_cnn_lstm_model.add(Dense((40))) # FC layer with 100 units
    hybrid_cnn_lstm_model.add(Reshape((40,1))) # Reshape my output of FC layer so that it's compatible
    hybrid_cnn_lstm_model.add(LSTM(10, dropout=0.4, recurrent_dropout=0.1, input_shape=(40,1), return_sequences=False))


    # Output layer with Softmax activation 
    hybrid_cnn_lstm_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
    return hybrid_cnn_lstm_model
        

def plotting(model_results):
    # Plotting accuracy trajectory
   plt.plot(model_results.history['accuracy'])
   plt.plot(model_results.history['val_accuracy'])
   plt.title('Hybrid CNN-LSTM model accuracy trajectory')
   plt.ylabel('accuracy')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.show()

   # Plotting loss trajectory
   plt.plot(model_results.history['loss'],'o')
   plt.plot(model_results.history['val_loss'],'o')
   plt.title('Hybrid CNN-LSTM model loss trajectory')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'val'], loc='upper left')
   plt.show()

    