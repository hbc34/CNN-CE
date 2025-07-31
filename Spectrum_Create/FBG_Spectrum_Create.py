import numpy as np
import time
import os
import csv

# Parameter settings
num_points = 2001  # Number of spectrum points
noise_level = 1000  # Noise level
num_spectra = 50  # Number of spectra to generate

# Wavelength range settings
wavelength_start = 1525  # Starting wavelength (nm)
wavelength_end = 1565    # Ending wavelength (nm)

# Initialize spectrum
spectrum = np.zeros(num_points, dtype=np.float32)

# Define peak parameters
peak1_width = 50  # Peak width
peak2_width = 80  # Peak width
peak1_height = 30000.0
peak2_height = 50000.0

# Calculate movement range and step size
move_range = num_points - max(peak1_width, peak2_width) * 2
step_size = move_range / (num_spectra - 1)  # Use floating-point step size, allowing decimal positions


def generate_peak(center, width, height, num_points):
    """Generate a single peak"""
    x = np.arange(num_points)
    # Use floating-point center position
    peak_mu = center  # Peak center wavelength
    peak_sigma = width * (x[1] - x[0]) / 5  # Convert width to wavelength units
    # Use the same Gaussian function formula as baseline
    peak = height * np.exp((-4*np.log(2))*((x-peak_mu)**2)/(peak_sigma**2))
    return peak


def index_to_wavelength(index):
    """Convert index position to wavelength value"""
    wavelength_range = wavelength_end - wavelength_start
    return wavelength_start + (index / (num_points - 1)) * wavelength_range


def save_spectrum(spectrum, peak_positions, spectrum_index):
    """Save spectrum data and peak positions"""
    # Create save directory
    directory = f"saved_spectra_fx/spectrum_{spectrum_index + 1}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save spectrum data as CSV file
    csv_file = os.path.join(directory, f"spectrum_{spectrum_index + 1}.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "Intensity"])
        for idx, val in enumerate(spectrum):
            writer.writerow([idx, val])
    
    # Save number of peaks and peak positions as TXT file
    txt_file = os.path.join(directory, f"spectrum_{spectrum_index + 1}_peaks.txt")
    with open(txt_file, mode='w') as file:
        file.write(f"Number of peaks: {len(peak_positions)}\n")
        for j, pos in enumerate(peak_positions):
            wavelength = index_to_wavelength(pos)
            file.write(f"Peak {j + 1}: Center at {pos:.2f}, Wavelength: {wavelength:.4f} nm\n")
    
    print(f"Saved spectrum {spectrum_index + 1}/{num_spectra}")


def main():
    """Main function to generate and save spectrum data"""
    # Ensure save directory exists
    if not os.path.exists("saved_spectra"):
        os.makedirs("saved_spectra")
    
    print(f"Starting to generate {num_spectra} spectra...")
    
    # Set initial positions (starting from the left)
    start_pos1 = peak1_width * 2
    start_pos2 = peak2_width * 2
    
    # Calculate movement step size for each peak
    # The two peaks move different distances, but must complete the entire scan simultaneously
    total_distance1 = move_range
    total_distance2 = move_range
    step1 = total_distance1 / (num_spectra - 1)  # Use floating-point step size
    step2 = total_distance2 / (num_spectra - 1)  # Use floating-point step size
    
    for i in range(num_spectra):
        # Calculate current peak positions (using floating-point positions)
        center1 = start_pos1 + i * step1
        center2 = start_pos2 + i * step2
        
        # Clear spectrum
        spectrum = np.zeros(num_points, dtype=np.float32)
        
        # Generate peaks
        peak1 = generate_peak(center1, peak1_width, peak1_height, num_points)
        peak2 = generate_peak(center2, peak2_width, peak2_height, num_points)
        
        # Synthesize spectrum and add noise
        spectrum += peak1 + peak2
        spectrum += np.random.uniform(-noise_level, noise_level, num_points).astype(np.float32)
        
        # Save spectrum and peak positions
        save_spectrum(spectrum, [center1, center2], i)
        
        # Brief delay to avoid excessive CPU usage
        time.sleep(0.01)
    
    print("All spectra generation completed!")
    print(f"Wavelength range: {wavelength_start}-{wavelength_end} nm")


if __name__ == "__main__":
    main()