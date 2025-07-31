import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.ndimage import gaussian_filter

DATA_PATH = 'data'
MODEL_PATH = 'model/fbg_cnn_model.h5'


def load_data(data_path):
    X = []
    Y = []
    for i in range(1, 10001):
        spectrum_file = os.path.join(data_path, f'spectrum_{i}', f'spectrum_{i}.csv')
        peaks_file = os.path.join(data_path, f'spectrum_{i}', f'spectrum_{i}_peaks.txt')

        if not os.path.exists(spectrum_file) or not os.path.exists(peaks_file):
            continue

        df = pd.read_csv(spectrum_file)
        intensity = df['Intensity'].values

        # Gaussian Filter
        intensity = gaussian_filter(intensity, sigma=10)

        # noise = np.random.normal(0, 0.1, intensity.shape)
        # intensity += noise

        with open(peaks_file, 'r') as f:
            lines = f.readlines()
            peaks = [int(line.split(' at ')[1]) for line in lines[1:] if 'Center' in line]

        X.append(intensity)
        y = np.zeros_like(intensity)
        y[peaks] = 1
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


X, Y = load_data(DATA_PATH)

X_train, X_test = X[:8000], X[8000:]
Y_train, Y_test = Y[:8000], Y[8000:]

input_shape = (2001, 1)
inputs = Input(shape=input_shape)
x = Conv1D(32, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(64, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
outputs = Dense(2001, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

X_train = X_train.reshape(-1, 2001, 1)
X_test = X_test.reshape(-1, 2001, 1)

# Early Stopping and ReduceLROnPlateau
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

#model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
model.fit(X_train, Y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[reduce_lr])

model.save(MODEL_PATH)
print("Model saved at", MODEL_PATH)
