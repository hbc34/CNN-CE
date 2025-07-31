import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

DATA_PATH = 'datatest'
MODEL_PATH = 'model/fbg_cnn_model.h5'

# Load Data
def load_data(data_path):
    X = []
    X_original = []  
    for i in range(1, 101): 
        spectrum_file = os.path.join(data_path, f'spectrum_{i}', f'spectrum_{i}.csv')

        if not os.path.exists(spectrum_file):
            continue

        df = pd.read_csv(spectrum_file)
        intensity = df['Intensity'].values
        
        #Original Data
        X_original.append(intensity)

        # Gaussian fFilter
        intensity_filtered = gaussian_filter(intensity, sigma=10)

        X.append(intensity_filtered)

    X = np.array(X)
    X_original = np.array(X_original)

    if X.size == 0:
        raise ValueError('Loaded data is empty. Please check the data path and files.')

    return X, X_original

# Post Process Peaks
def post_process_peaks(predictions, threshold=0.01, min_distance=20):
    peaks = []
    for p in predictions:
        peak_indices = np.where(p > threshold)[0]
        filtered_peaks = []
        for idx in peak_indices:
            if len(filtered_peaks) == 0 or idx - filtered_peaks[-1] > min_distance:
                filtered_peaks.append(idx)
        peaks.append(filtered_peaks)
    return peaks

# Main
def main():
    # Load model
    X, X_original = load_data(DATA_PATH)
    X = X.reshape(-1, 2001, 1)
    X_original = X_original.reshape(-1, 2001, 1)

    model = load_model(MODEL_PATH)

    inference_times = []

    for i in range(len(X)):
        spectrum = X[i:i+1] 
        original_spectrum = X_original[i:i+1]
        print(f"Input data shape for Spectrum {i + 1}: {spectrum.shape}")

        plt.figure(figsize=(10, 6))
        plt.plot(original_spectrum.flatten(), label='Intensity')
        plt.title(f"Original Spectrum {i + 1} ")
        plt.xlabel("Point Index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        start_time = time.time()

        predictions = model.predict(spectrum)

        end_time = time.time()

        inference_time = end_time - start_time
        inference_times.append(inference_time)
        print(f"Spectrum {i + 1} Inference Time: {inference_time:.4f} seconds")


        peak_positions = post_process_peaks(predictions, threshold=0.05, min_distance=1)


        for peaks in peak_positions:
            plt.figure(figsize=(10, 6))
            plt.plot(spectrum.flatten(), label='Intensity')
            plt.scatter(peaks, spectrum[0, peaks, 0], color='red', label='Predicted Peaks')
            plt.title(f"Spectrum {i + 1}")
            plt.xlabel("Point Index")
            plt.ylabel("Intensity")
            plt.legend()
            plt.show()


    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average Inference Time: {average_inference_time:.4f} seconds")


if __name__ == "__main__":
    main()
