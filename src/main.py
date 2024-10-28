import os
from io_utils.io_utils import ATFHandler
from filtering.filtering import apply_savgol_filter, apply_fft_filter, butter_lowpass_filter
import matplotlib.pyplot as plt

def plot_signals(original, filtered, title):
    """ Function to plot the original and filtered signals. """
    plt.figure(figsize=(10, 6))
    plt.plot(original, label="Original Signal")
    plt.plot(filtered, label="Filtered Signal", linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()

def process_atf_file(filepath):
    """ Function to load ATF file, apply filters, and display results. """
    atf_handler = ATFHandler(filepath)
    atf_handler.load_atf()

    # Print the headers to help adjust column names
    print("Headers:", atf_handler.headers)

    # Try to extract columns (error-handling for missing columns)
    try:
        time_data = atf_handler.get_column("Time")  # Adjust based on actual column names
    except ValueError as e:
        print(e)
        return

    try:
        current_data = atf_handler.get_column("Current")  # Adjust based on actual column names
    except ValueError as e:
        print(e)
        return

    # Apply filters to the current data
    savgol_filtered = apply_savgol_filter(current_data, window_length=51, polyorder=3)
    fft_filtered = apply_fft_filter(current_data, threshold=0.2)
    lowpass_filtered = butter_lowpass_filter(current_data, cutoff=0.1, fs=1000)

    # Plot and compare original and filtered data
    plot_signals(current_data, savgol_filtered, "Savitzky-Golay Filter")
    plot_signals(current_data, fft_filtered, "FFT-Based Filter")
    plot_signals(current_data, lowpass_filtered, "Butterworth Low-pass Filter")

if __name__ == "__main__":
    # Specify the path to your ATF file in the data folder
    data_folder = "data"
    atf_files = [f for f in os.listdir(data_folder) if f.endswith(".atf")]

    if atf_files:
        file_to_process = os.path.join(data_folder, atf_files[0])
        print(f"Processing file: {file_to_process}")
        process_atf_file(file_to_process)
    else:
        print("No ATF files found in the data folder.")
