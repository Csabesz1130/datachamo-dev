# src/interactive_filter_app.py
import numpy as np
from tkinter import Tk, ttk, StringVar
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from pathlib import Path
from src.filtering.filtering import apply_savgol_filter, apply_fft_filter, butter_lowpass_filter
from src.io_utils.io_utils import ATFHandler
from src.utils.logger import app_logger

# src/interactive_filter_app.py
class NoiseCancellingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive Noise Cancelling")

        # Create main frame
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill='both', expand=True)

        # Create top frame for file controls
        self.file_frame = ttk.Frame(self.main_frame)
        self.file_frame.pack(side='top', fill='x', padx=5, pady=5)
        
        ttk.Button(self.file_frame, text="Load ATF File", 
                  command=self.load_atf_file).pack(side='left', padx=5)
        self.file_label = ttk.Label(self.file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=5)

        # Create frame for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side='top', fill='both', expand=True)

        # Initialize data attributes
        self.data = None
        self.filtered_data = None
        self.time_data = None

        # Initialize filter parameters
        self.savgol_window = tk.IntVar(value=51)
        self.savgol_polyorder = tk.IntVar(value=3)
        self.apply_savgol = tk.BooleanVar()

        self.fft_threshold = tk.DoubleVar(value=0.2)
        self.apply_fft = tk.BooleanVar()

        self.butter_cutoff = tk.DoubleVar(value=0.1)
        self.butter_order = tk.IntVar(value=5)
        self.apply_butter = tk.BooleanVar()

        # Setup plot
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side='bottom', fill='x', padx=5, pady=5)

        # Setup controls
        self.setup_controls()

        # Add traces for real-time updates
        self.savgol_window.trace_add("write", self.on_filter_change)
        self.savgol_polyorder.trace_add("write", self.on_filter_change)
        self.apply_savgol.trace_add("write", self.on_filter_change)
        self.fft_threshold.trace_add("write", self.on_filter_change)
        self.apply_fft.trace_add("write", self.on_filter_change)
        self.butter_cutoff.trace_add("write", self.on_filter_change)
        self.butter_order.trace_add("write", self.on_filter_change)
        self.apply_butter.trace_add("write", self.on_filter_change)

        app_logger.info("NoiseCancellingApp initialized")

    def on_filter_change(self, *args):
        """Called whenever a filter parameter changes"""
        if self.data is not None:
            self.apply_filters()

    def setup_controls(self):
        """Setup the control panel with organized filter controls"""
        # Create main control frame with better organization
        filter_frame = ttk.LabelFrame(self.control_frame, text="Filter Controls")
        filter_frame.pack(fill='x', padx=5, pady=5)

        # Savitzky-Golay controls
        savgol_frame = ttk.LabelFrame(filter_frame, text="Savitzky-Golay Filter")
        savgol_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(savgol_frame, text="Enable", variable=self.apply_savgol).pack()
        
        # Window Length control
        ttk.Label(savgol_frame, text="Window Length").pack()
        ttk.Scale(savgol_frame, from_=5, to=101, variable=self.savgol_window, 
                orient='horizontal').pack(fill='x')
        
        # Polynomial Order control
        ttk.Label(savgol_frame, text="Polynomial Order").pack()
        ttk.Scale(savgol_frame, from_=2, to=5, variable=self.savgol_polyorder,
                orient='horizontal').pack(fill='x')

        # FFT controls
        fft_frame = ttk.LabelFrame(filter_frame, text="FFT Filter")
        fft_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(fft_frame, text="Enable", variable=self.apply_fft).pack()
        
        # Threshold control
        ttk.Label(fft_frame, text="Threshold").pack()
        ttk.Scale(fft_frame, from_=0.01, to=1.0, variable=self.fft_threshold,
                orient='horizontal').pack(fill='x')

        # Butterworth controls
        butter_frame = ttk.LabelFrame(filter_frame, text="Butterworth Filter")
        butter_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(butter_frame, text="Enable", variable=self.apply_butter).pack()
        
        # Cutoff Frequency control
        ttk.Label(butter_frame, text="Cutoff Frequency").pack()
        ttk.Scale(butter_frame, from_=0.01, to=1.0, variable=self.butter_cutoff,
                orient='horizontal').pack(fill='x')
        
        # Order control
        ttk.Label(butter_frame, text="Order").pack()
        ttk.Scale(butter_frame, from_=1, to=10, variable=self.butter_order,
                orient='horizontal').pack(fill='x')

        # Add statistics display
        self.stats_var = tk.StringVar(value="No data loaded")
        stats_label = ttk.Label(self.control_frame, textvariable=self.stats_var, wraplength=300)
        stats_label.pack(fill='x', padx=5, pady=5)

        # Reset button
        ttk.Button(self.control_frame, text="Reset", command=self.reset_plot).pack(pady=5)

    def apply_filters(self):
        """Apply all selected filters in sequence"""
        if self.data is None:
            return

        app_logger.info("=== Applying Filters ===")
        # Start with original data
        self.filtered_data = self.data.copy()

        try:
            # Apply Savitzky-Golay filter first (good for smoothing)
            if self.apply_savgol.get():
                app_logger.info("Applying Savitzky-Golay filter")
                window_length = self.savgol_window.get()
                if window_length % 2 == 0:  # Make sure window length is odd
                    window_length += 1
                self.filtered_data = apply_savgol_filter(
                    self.filtered_data,
                    window_length=window_length,
                    polyorder=self.savgol_polyorder.get()
                )
                app_logger.debug(f"Savitzky-Golay applied with window={window_length}, order={self.savgol_polyorder.get()}")

            # Then apply FFT filter (good for frequency-based noise)
            if self.apply_fft.get():
                app_logger.info("Applying FFT filter")
                self.filtered_data = apply_fft_filter(
                    self.filtered_data,
                    threshold=self.fft_threshold.get()
                )
                app_logger.debug(f"FFT filter applied with threshold={self.fft_threshold.get()}")

            # Finally apply Butterworth filter (good for specific frequency cutoff)
            if self.apply_butter.get():
                app_logger.info("Applying Butterworth filter")
                self.filtered_data = butter_lowpass_filter(
                    self.filtered_data,
                    cutoff=self.butter_cutoff.get(),
                    fs=1000,  # Sampling frequency
                    order=self.butter_order.get()
                )
                app_logger.debug(f"Butterworth applied with cutoff={self.butter_cutoff.get()}, order={self.butter_order.get()}")

            # Calculate and log filter effectiveness
            if any([self.apply_savgol.get(), self.apply_fft.get(), self.apply_butter.get()]):
                original_stats = self._calculate_signal_stats(self.data)
                filtered_stats = self._calculate_signal_stats(self.filtered_data)
                self._log_filter_effectiveness(original_stats, filtered_stats)

            self.update_plot()
            app_logger.info("=== Filter Application Complete ===")

        except Exception as e:
            app_logger.error(f"Error applying filters: {str(e)}")
            self.filtered_data = self.data.copy()  # Reset on error
            raise

    def _calculate_signal_stats(self, signal):
        """Calculate signal statistics"""
        return {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'peak_to_peak': np.ptp(signal),
            'rms': np.sqrt(np.mean(np.square(signal)))
        }

    def _log_filter_effectiveness(self, original_stats, filtered_stats):
        """Log the effectiveness of the filtering"""
        app_logger.info("Filter Effectiveness:")
        app_logger.info(f"  Noise Reduction: {(original_stats['std'] - filtered_stats['std'])/original_stats['std']*100:.2f}%")
        app_logger.info(f"  Peak-to-Peak Change: {(original_stats['peak_to_peak'] - filtered_stats['peak_to_peak'])/original_stats['peak_to_peak']*100:.2f}%")
        app_logger.info(f"  RMS Change: {(original_stats['rms'] - filtered_stats['rms'])/original_stats['rms']*100:.2f}%")


    def update_plot(self):
        """Update plot with both original and filtered data"""
        if self.data is None:
            return
            
        self.ax.clear()
        
        # Plot original data with transparency
        self.ax.plot(self.time_data, self.data, 'b-', label='Original Signal', alpha=0.5)
        
        # Plot filtered data if filters are applied
        if any([self.apply_savgol.get(), self.apply_fft.get(), self.apply_butter.get()]):
            self.ax.plot(self.time_data, self.filtered_data, 'r-', label='Filtered Signal')
        
        self.ax.set_title("Signal Analysis")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Current (pA)")
        self.ax.grid(True)
        self.ax.legend()
        
        # Update statistics
        if self.filtered_data is not None:
            stats = self._calculate_signal_stats(self.filtered_data)
            self.stats_var.set(f"Signal Stats:\nMean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\nPeak-to-Peak: {stats['peak_to_peak']:.2f}")
        
        self.fig.tight_layout()
        self.canvas.draw()

    def reset_plot(self):
        """Reset the plot to the original data"""
        if self.data is None:
            return
        app_logger.info("Resetting plot to original data")
        self.filtered_data = self.data.copy()
        self.plot_original_data()

    def load_atf_file(self):
        """Load ATF file with comprehensive logging"""
        app_logger.info("=== File Operation Started ===")
        app_logger.info("Initiating file selection dialog")
        
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
        app_logger.debug(f"Looking for files in: {data_dir}")
        
        try:
            filepath = tk.filedialog.askopenfilename(
                initialdir=data_dir,
                title="Select ATF file",
                filetypes=(("ATF files", "*.atf"), ("all files", "*.*"))
            )
            
            if not filepath:
                app_logger.info("File selection cancelled by user")
                return
                
            app_logger.info(f"File selected: {filepath}")
            
            # Load ATF file
            app_logger.debug("Creating ATF handler instance")
            atf_handler = ATFHandler(filepath)
            
            app_logger.info("Loading ATF file content")
            atf_handler.load_atf()
            
            # Get data
            app_logger.debug("Extracting data columns")
            self.data = atf_handler.get_column("#1")
            self.time_data = atf_handler.get_column("Time")
            self.filtered_data = self.data.copy()
            
            app_logger.info(f"Data loaded successfully:")
            app_logger.info(f"  - Number of points: {len(self.data)}")
            app_logger.info(f"  - Time range: {self.time_data[0]:.2f}s to {self.time_data[-1]:.2f}s")
            app_logger.info(f"  - Data range: {self.data.min():.2f} to {self.data.max():.2f}")
            
            # Update UI
            filename = Path(filepath).name
            self.file_label.config(text=filename)
            app_logger.debug(f"Updated UI with filename: {filename}")
            
            self.plot_original_data()
            app_logger.info("=== File Operation Completed ===")
            
        except Exception as e:
            app_logger.error("=== File Operation Failed ===")
            app_logger.error(f"Error details: {str(e)}")
            self.file_label.config(text=f"Error loading file: {str(e)}")

    def plot_original_data(self):
        """Plot original data"""
        if self.data is None:
            return
                
        try:
            app_logger.debug("Plotting original data")
            self.ax.clear()
            self.ax.plot(self.time_data, self.data, label="Original Signal")
            self.ax.set_title("Signal with Noise")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Current (pA)")
            self.ax.legend()
            self.ax.grid(True)
            self.fig.tight_layout()
            self.canvas.draw()
            app_logger.debug("Plot updated successfully")
                
        except Exception as e:
            app_logger.error(f"Error plotting data: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    try:
        app_logger.info("==========================================")
        app_logger.info("=== Signal Analysis Application Start ===")
        app_logger.info("==========================================")
        
        root = Tk()
        app = NoiseCancellingApp(root)
        app_logger.info("Application initialized successfully")
        app_logger.info("Starting main event loop")
        root.mainloop()
        
    except Exception as e:
        app_logger.critical("=== Critical Application Error ===")
        app_logger.critical(f"Error details: {str(e)}")
        raise