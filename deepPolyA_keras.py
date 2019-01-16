import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Bidirectional, MaxPooling1D, GlobalAveragePooling1D, Flatten, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


def get_data(path):
    data = pd.read_csv(path);
    alphabet = np.array(['A', 'G', 'T', 'C'])
    X = [];
    for line in data['data']:
        line = list(line.strip('\n'));
        seq = np.array(line, dtype = '|U1').reshape(-1,1);
        seq_data = (seq == alphabet).astype(np.int32)
        X.append(seq_data);
    X = np.array(X);
    y = np.array(data['label'], dtype = np.int32);
 
    return X, y; #(n, 206, 4), (n,)

def main():

    train_X, train_y = get_data('./total_train.csv');
    valid_X, valid_y = get_data('./total_valid.csv');

    input_shape = (206, 4)

    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size = 10, activation = 'relu', input_shape = input_shape));
    model.add(MaxPooling1D(pool_size= 5))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv1D(filters = 64, kernel_size = 5, activation = 'relu'));
    model.add(MaxPooling1D(pool_size= 5))
    model.add(BatchNormalization())    
    model.add(Dropout(0.25))
    
    model.add(Bidirectional(LSTM(units = 32, dropout = 0.1, recurrent_dropout = 0.2)))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer= SGD(lr = 1e-4, momentum = 0.95, nesterov=True), metrics=['accuracy']);

    print(model.summary())

    history = model.fit(train_X, train_y, batch_size = 64, epochs = 100,validation_data=(valid_X, valid_y));
    
    model.save("./model1.h5")

main();
