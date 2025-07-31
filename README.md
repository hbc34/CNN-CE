# CNN-CE: 1D-CNN Based Fiber Bragg Grating (FBG) Spectrum Recognition Algorithm

## Project Overview

CNN-CE is a one-dimensional convolutional neural network (1D-CNN) based algorithm for Fiber Bragg Grating (FBG) spectrum peak detection and recognition. This project implements a complete workflow from data generation and model training to embedded deployment, specifically optimized for AX630C chip deployment.

## Algorithm Features

- **High-Precision Peak Detection**: Uses 1D-CNN network for spectrum data peak position prediction
- **Real-time Processing**: Supports real-time analysis of 2001 data points spectrum
- **Embedded Deployment**: Optimized for AX630C chip, supporting edge computing scenarios
- **Complete Workflow**: Includes data generation, training, and inference solutions


## Algorithm Principles

### Network Architecture
- **Input**: 2001 data points of spectrum intensity data
- **Network Structure**: Multi-layer 1D convolution + Batch normalization + Max pooling + Dropout
- **Output**: Peak probability prediction for 2001 positions
- **Wavelength Range**: 1525-1565 nm

### Data Preprocessing
- Gaussian filtering smoothing (σ=10)
- Data normalization
- Noise augmentation training

### Post-processing Algorithm
- Peak threshold filtering (default 0.23)
- Minimum distance constraints
- Peak centroid calculation
- Wavelength mapping conversion

## Quick Start

### 1. Data Generation

```bash
cd Spectrum_Create
python FBG_Spectrum_Create.py
```

**Features**:
- Generate simulated FBG spectrum data
- Support dual-peak spectrum simulation
- Automatically add Gaussian noise
- Output CSV format spectrum data and peak annotations

**Parameter Configuration**:
- `num_points`: Number of spectrum data points (default 2001)
- `wavelength_start/end`: Wavelength range (1525-1565 nm)
- `num_spectra`: Number of spectra to generate (default 50)
- `noise_level`: Noise level (default 1000)

### 2. Model Training

```bash
cd Model_Train
python FBG_REC_TrainV2.py
```

**Training Configuration**:
- **Dataset**: 10000 spectrum samples
- **Train/Test Ratio**: 8000/2000

### 3. AX630C Deployment

#### Environment Setup

Follow the official [ax-samples](https://github.com/AXERA-TECH/ax-samples) documentation to configure the cross-compilation environment.

#### Compilation Steps

```bash
cd AX630C_Run/ax-samples
# Configure cross-compilation toolchain
export PATH=/path/to/cross-compiler:$PATH

# Compile
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j4
```

#### Running Programs

**Complete Algorithm Version**:
```bash
./ax_spectum --model fbg_cnn_model.axmodel --data spectrum_data.csv
```

**Pure CNN Inference Version**:
```bash
./ax_spectum_cnn_only --model fbg_cnn_model.axmodel --data spectrum_data.csv
```

## Input/Output Format

### Input Data Format
CSV file with two columns:
```csv
index,Intensity
0,1234.56
1,1235.78
...
2000,1240.12
```

### Output Results
- **Peak Position**: Array index
- **Peak Wavelength**: Corresponding wavelength value (nm)
- **Peak Intensity**: Spectrum intensity value
- **Confidence**: CNN prediction probability

## Performance Metrics

- **Detection Accuracy**: >95% (on test dataset)
- **Processing Speed**: <10ms (AX630C single inference)
- **Memory Usage**: <50MB
- **Power Consumption**: <2W (AX630C runtime)

## Dependencies

### Python Environment (Training)
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- SciPy

### C++ Environment (Deployment)
- GCC 7.0+
- CMake 3.10+
- AX630C SDK
- ax-samples framework