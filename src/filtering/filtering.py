import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.fft import fft, ifft
from src.utils.logger import app_logger


def apply_savgol_filter(data, window_length=51, polyorder=3):
    """
    Apply Savitzky-Golay filter to the data.
    
    Args:
        data (np.array): Input signal data
        window_length (int): Length of the filter window (must be odd)
        polyorder (int): Order of the polynomial
        
    Returns:
        np.array: Filtered data
    """
    app_logger.debug(f"Applying Savitzky-Golay filter: window={window_length}, order={polyorder}")
    try:
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        filtered_data = savgol_filter(data, window_length, polyorder)
        app_logger.info("Savitzky-Golay filter applied successfully")
        return filtered_data
    
    except Exception as e:
        app_logger.error(f"Error applying Savitzky-Golay filter: {str(e)}")
        raise


def apply_fft_filter(data, threshold=0.1):
    """
    Apply FFT-based noise filtering.
    
    Args:
        data (np.array): Input signal data
        threshold (float): Threshold for frequency components (0 to 1)
        
    Returns:
        np.array: Filtered data
    """
    app_logger.debug(f"Applying FFT filter with threshold: {threshold}")
    try:
        # Compute FFT
        fft_data = fft(data)
        
        # Get magnitudes and create mask for significant frequencies
        magnitudes = np.abs(fft_data)
        max_magnitude = np.max(magnitudes)
        mask = magnitudes > (threshold * max_magnitude)
        
        # Apply mask and inverse transform
        filtered_fft = fft_data * mask
        filtered_data = np.real(ifft(filtered_fft))
        
        app_logger.info("FFT filter applied successfully")
        return filtered_data
    
    except Exception as e:
        app_logger.error(f"Error applying FFT filter: {str(e)}")
        raise


def butter_lowpass_filter(data, cutoff, fs=1000.0, order=5):
    """
    Apply Butterworth low-pass filter.
    
    Args:
        data (np.array): Input signal data
        cutoff (float): Cutoff frequency
        fs (float): Sampling frequency
        order (int): Filter order
        
    Returns:
        np.array: Filtered data
    """
    app_logger.debug(f"Applying Butterworth filter: cutoff={cutoff}, fs={fs}, order={order}")
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Create filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        filtered_data = filtfilt(b, a, data)
        
        app_logger.info("Butterworth filter applied successfully")
        return filtered_data
    
    except Exception as e:
        app_logger.error(f"Error applying Butterworth filter: {str(e)}")
        raise


def combined_filter(data, savgol_params=None, fft_threshold=None, butter_params=None):
    """
    Apply multiple filters in sequence.
    
    Args:
        data (np.array): Input signal data
        savgol_params (dict): Parameters for Savitzky-Golay filter
        fft_threshold (float): Threshold for FFT filter
        butter_params (dict): Parameters for Butterworth filter
        
    Returns:
        np.array: Filtered data
    """
    app_logger.info("Starting combined filtering process")
    filtered_data = data.copy()
    
    try:
        if savgol_params:
            app_logger.debug("Applying Savitzky-Golay filter in combination")
            filtered_data = apply_savgol_filter(
                filtered_data,
                window_length=savgol_params['window_length'],
                polyorder=savgol_params['polyorder']
            )
        
        if fft_threshold is not None:
            app_logger.debug("Applying FFT filter in combination")
            filtered_data = apply_fft_filter(
                filtered_data,
                threshold=fft_threshold
            )
        
        if butter_params:
            app_logger.debug("Applying Butterworth filter in combination")
            filtered_data = butter_lowpass_filter(
                filtered_data,
                cutoff=butter_params['cutoff'],
                fs=butter_params.get('fs', 1000.0),
                order=butter_params['order']
            )
        
        app_logger.info("Combined filtering completed successfully")
        return filtered_data
    
    except Exception as e:
        app_logger.error(f"Error in combined filtering: {str(e)}")
        raise


def calculate_filter_metrics(original_data, filtered_data):
    """
    Calculate metrics to evaluate filter performance.
    
    Args:
        original_data (np.array): Original signal data
        filtered_data (np.array): Filtered signal data
        
    Returns:
        dict: Dictionary containing filter performance metrics
    """
    try:
        # Calculate various metrics
        residuals = original_data - filtered_data
        mse = np.mean(np.square(residuals))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(np.square(filtered_data))
        noise_power = np.mean(np.square(residuals))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'snr_db': snr,
            'max_deviation': np.max(np.abs(residuals)),
            'mean_deviation': np.mean(residuals)
        }
        
        app_logger.debug("Filter metrics calculated successfully")
        return metrics
        
    except Exception as e:
        app_logger.error(f"Error calculating filter metrics: {str(e)}")
        raise


def adaptive_threshold_filter(data, window_size=50):
    """
    Apply adaptive thresholding based on local signal statistics.
    
    Args:
        data (np.array): Input signal data
        window_size (int): Size of the sliding window
        
    Returns:
        np.array: Filtered data
    """
    app_logger.debug(f"Applying adaptive threshold filter with window_size={window_size}")
    try:
        filtered_data = data.copy()
        padding = window_size // 2
        
        # Pad the data for edge handling
        padded_data = np.pad(data, (padding, padding), mode='edge')
        
        for i in range(len(data)):
            # Get local window
            window = padded_data[i:i + window_size]
            
            # Calculate local statistics
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            # Apply adaptive threshold
            if abs(data[i] - local_mean) > 2 * local_std:
                filtered_data[i] = local_mean
        
        app_logger.info("Adaptive threshold filter applied successfully")
        return filtered_data
        
    except Exception as e:
        app_logger.error(f"Error applying adaptive threshold filter: {str(e)}")
        raise

def extract_add_filter(data, window_length=51, polyorder=3, prominence_threshold=200, width_range=(1, 50)):
    """
    Implementation of the extract-filter-add method for preserving important signal features.
    1. Extracts significant peaks/events from the signal
    2. Filters the remaining baseline
    3. Adds the extracted features back to the filtered baseline
    
    Args:
        data (np.array): Input signal data
        window_length (int): Length of the filter window for baseline filtering
        polyorder (int): Order of the polynomial for baseline filtering
        prominence_threshold (float): Minimum prominence for peak detection
        width_range (tuple): Expected width range of peaks (min, max)
    
    Returns:
        np.array: Filtered data with preserved peaks/events
    """
    app_logger.debug("Starting extract-add filtering process")
    try:
        # 1. Peak Detection and Extraction
        # Find peaks with significant prominence
        peaks, properties = find_peaks(data, 
                                     prominence=prominence_threshold,
                                     width=width_range)
        
        # Find valleys (negative peaks) by inverting the signal
        valleys, valley_properties = find_peaks(-data, 
                                              prominence=prominence_threshold,
                                              width=width_range)
        
        # Create masks for peaks and their surrounding regions
        event_mask = np.zeros_like(data, dtype=bool)
        
        # Function to add peak/valley regions to mask
        def add_region_to_mask(indices, widths):
            for idx, width in zip(indices, widths):
                left_idx = max(0, int(idx - width))
                right_idx = min(len(data), int(idx + width + 1))
                event_mask[left_idx:right_idx] = True
        
        # Add peak regions to mask
        add_region_to_mask(peaks, properties['widths'])
        # Add valley regions to mask
        add_region_to_mask(valleys, valley_properties['widths'])
        
        # 2. Separate Signal Components
        # Extract events
        events = np.zeros_like(data)
        events[event_mask] = data[event_mask]
        
        # Create baseline by removing events
        baseline = data.copy()
        baseline[event_mask] = np.interp(
            np.where(event_mask)[0],
            np.where(~event_mask)[0],
            baseline[~event_mask]
        )
        
        # 3. Filter Baseline
        filtered_baseline = savgol_filter(baseline, window_length, polyorder)
        
        # 4. Reconstruct Signal
        # Add events back to filtered baseline
        filtered_data = filtered_baseline + events - baseline[event_mask].mean()
        
        # 5. Optional: Smooth Transitions
        # Find edges of event regions
        edges = np.where(np.diff(event_mask.astype(int)) != 0)[0]
        
        # Smooth transitions at edges
        for edge in edges:
            if edge > 1 and edge < len(data) - 2:  # Avoid array bounds
                # Create small overlap for smooth transition
                left_idx = max(0, edge - 2)
                right_idx = min(len(data), edge + 3)
                
                # Apply local smoothing at transition points
                transition = np.mean([
                    filtered_data[left_idx:right_idx],
                    savgol_filter(filtered_data[left_idx:right_idx], 5, 2)
                ], axis=0)
                
                filtered_data[left_idx:right_idx] = transition
        
        app_logger.info("Extract-add filter completed successfully")
        return filtered_data
        
    except Exception as e:
        app_logger.error(f"Error in extract-add filtering: {str(e)}")
        raise

def combined_filter(data, savgol_params=None, fft_threshold=None, butter_params=None, 
                   use_extract_add=False, extract_add_params=None):
    """
    Apply multiple filters in sequence.
    
    Args:
        data (np.array): Input signal data
        savgol_params (dict): Parameters for Savitzky-Golay filter
        fft_threshold (float): Threshold for FFT filter
        butter_params (dict): Parameters for Butterworth filter
        use_extract_add (bool): Whether to use extract-add filtering
        extract_add_params (dict): Parameters for extract-add filter
    
    Returns:
        np.array: Filtered data
    """
    app_logger.info("Starting combined filtering process")
    filtered_data = data.copy()
    
    try:
        if use_extract_add:
            app_logger.debug("Applying extract-add filter in combination")
            extract_add_params = extract_add_params or {}
            filtered_data = extract_add_filter(
                filtered_data,
                window_length=extract_add_params.get('window_length', 51),
                polyorder=extract_add_params.get('polyorder', 3),
                prominence_threshold=extract_add_params.get('prominence_threshold', 200),
                width_range=extract_add_params.get('width_range', (1, 50))
            )
        
        if savgol_params:
            app_logger.debug("Applying Savitzky-Golay filter in combination")
            filtered_data = apply_savgol_filter(
                filtered_data,
                window_length=savgol_params['window_length'],
                polyorder=savgol_params['polyorder']
            )
        
        if fft_threshold is not None:
            app_logger.debug("Applying FFT filter in combination")
            filtered_data = apply_fft_filter(
                filtered_data,
                threshold=fft_threshold
            )
        
        if butter_params:
            app_logger.debug("Applying Butterworth filter in combination")
            filtered_data = butter_lowpass_filter(
                filtered_data,
                cutoff=butter_params['cutoff'],
                fs=butter_params.get('fs', 1000.0),
                order=butter_params['order']
            )
        
        app_logger.info("Combined filtering completed successfully")
        return filtered_data
        
    except Exception as e:
        app_logger.error(f"Error in combined filtering: {str(e)}")
        raise


def get_filter_info(filter_name):
    """
    Get information about a specific filter.
    
    Args:
        filter_name (str): Name of the filter
        
    Returns:
        dict: Dictionary containing filter information
    """
    filter_info = {
        'savitzky_golay': {
            'description': 'Smooths data using local polynomial regression',
            'parameters': {
                'window_length': 'Length of the filter window (must be odd)',
                'polyorder': 'Order of the polynomial used for fitting'
            },
            'suitable_for': [
                'Smoothing without shifting peaks',
                'Preserving higher moments of the signal',
                'Data with uniform sampling'
            ]
        },
        'fft': {
            'description': 'Filters noise in frequency domain',
            'parameters': {
                'threshold': 'Threshold for frequency components (0 to 1)'
            },
            'suitable_for': [
                'Removing high-frequency noise',
                'Periodic signals',
                'Frequency-based analysis'
            ]
        },
        'butterworth': {
            'description': 'Low-pass filter with maximally flat response',
            'parameters': {
                'cutoff': 'Cutoff frequency',
                'fs': 'Sampling frequency',
                'order': 'Filter order'
            },
            'suitable_for': [
                'Removing high-frequency noise',
                'Smooth frequency response',
                'Real-time applications'
            ]
        }
    }
    
    return filter_info.get(filter_name, {
        'description': 'Filter information not available',
        'parameters': {},
        'suitable_for': []
    })