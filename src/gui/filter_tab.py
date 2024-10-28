import tkinter as tk
from tkinter import ttk
from src.utils.logger import app_logger

class FilterTab:
    def __init__(self, parent, callback):
        """
        Initialize the filter tab with dynamic filter controls.
        
        Args:
            parent: Parent tkinter widget
            callback: Function to call when filters change
        """
        self.parent = parent
        self.update_callback = callback
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Filter Controls", padding="5 5 5 5")
        self.frame.pack(fill='x', padx=5, pady=5)
        
        # Initialize filter variables
        self.init_filter_vars()
        
        # Create filter controls
        self.setup_savgol_controls()
        self.setup_fft_controls()
        self.setup_butterworth_controls()
        self.setup_extract_add_controls()

    def init_filter_vars(self):
        """Initialize all filter-related variables"""
        # Savitzky-Golay filter variables
        self.use_savgol = tk.BooleanVar(value=False)
        self.savgol_window = tk.IntVar(value=51)
        self.savgol_order = tk.IntVar(value=3)
        
        # FFT filter variables
        self.use_fft = tk.BooleanVar(value=False)
        self.fft_threshold = tk.DoubleVar(value=0.2)
        
        # Butterworth filter variables
        self.use_butterworth = tk.BooleanVar(value=False)
        self.butter_cutoff = tk.DoubleVar(value=0.1)
        self.butter_order = tk.IntVar(value=5)
        
        # Extract-Add filter variables
        self.use_extract_add = tk.BooleanVar(value=False)
        self.extract_prominence = tk.DoubleVar(value=200)
        self.extract_width_min = tk.IntVar(value=1)
        self.extract_width_max = tk.IntVar(value=50)

    def setup_savgol_controls(self):
        """Setup Savitzky-Golay filter controls"""
        savgol_frame = ttk.LabelFrame(self.frame, text="Savitzky-Golay Filter")
        savgol_frame.pack(fill='x', padx=5, pady=5)
        
        # Enable checkbox
        ttk.Checkbutton(savgol_frame, text="Enable", 
                       variable=self.use_savgol,
                       command=self.on_filter_change).pack(pady=2)
        
        # Window length control
        window_frame = ttk.Frame(savgol_frame)
        window_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(window_frame, text="Window Length:").pack(side='left')
        ttk.Scale(window_frame, from_=5, to=101, variable=self.savgol_window,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(window_frame, textvariable=self.savgol_window).pack(side='right')
        
        # Polynomial order control
        order_frame = ttk.Frame(savgol_frame)
        order_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(order_frame, text="Polynomial Order:").pack(side='left')
        ttk.Scale(order_frame, from_=2, to=5, variable=self.savgol_order,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(order_frame, textvariable=self.savgol_order).pack(side='right')

    def setup_fft_controls(self):
        """Setup FFT filter controls"""
        fft_frame = ttk.LabelFrame(self.frame, text="FFT Filter")
        fft_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(fft_frame, text="Enable",
                       variable=self.use_fft,
                       command=self.on_filter_change).pack(pady=2)
        
        threshold_frame = ttk.Frame(fft_frame)
        threshold_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(threshold_frame, text="Threshold:").pack(side='left')
        ttk.Scale(threshold_frame, from_=0.01, to=1.0, variable=self.fft_threshold,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(threshold_frame, textvariable=self.fft_threshold).pack(side='right')

    def setup_butterworth_controls(self):
        """Setup Butterworth filter controls"""
        butter_frame = ttk.LabelFrame(self.frame, text="Butterworth Filter")
        butter_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(butter_frame, text="Enable",
                       variable=self.use_butterworth,
                       command=self.on_filter_change).pack(pady=2)
        
        cutoff_frame = ttk.Frame(butter_frame)
        cutoff_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(cutoff_frame, text="Cutoff Frequency:").pack(side='left')
        ttk.Scale(cutoff_frame, from_=0.01, to=1.0, variable=self.butter_cutoff,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(cutoff_frame, textvariable=self.butter_cutoff).pack(side='right')
        
        order_frame = ttk.Frame(butter_frame)
        order_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(order_frame, text="Order:").pack(side='left')
        ttk.Scale(order_frame, from_=1, to=10, variable=self.butter_order,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(order_frame, textvariable=self.butter_order).pack(side='right')

    def setup_extract_add_controls(self):
        """Setup Extract-Add filter controls"""
        extract_frame = ttk.LabelFrame(self.frame, text="Extract-Add Filter")
        extract_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(extract_frame, text="Enable",
                       variable=self.use_extract_add,
                       command=self.on_filter_change).pack(pady=2)
        
        # Prominence threshold
        prom_frame = ttk.Frame(extract_frame)
        prom_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(prom_frame, text="Prominence:").pack(side='left')
        ttk.Scale(prom_frame, from_=50, to=500, variable=self.extract_prominence,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(prom_frame, textvariable=self.extract_prominence).pack(side='right')
        
        # Width range controls
        width_frame = ttk.Frame(extract_frame)
        width_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(width_frame, text="Width Range:").pack(side='left')
        ttk.Scale(width_frame, from_=1, to=20, variable=self.extract_width_min,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)
        ttk.Label(width_frame, text="to").pack(side='left')
        ttk.Scale(width_frame, from_=21, to=100, variable=self.extract_width_max,
                 orient='horizontal', command=lambda _: self.on_filter_change()
                 ).pack(side='left', fill='x', expand=True)

    def get_active_filters(self):
        """Get the currently active filters and their parameters"""
        filters = {}
        
        if self.use_savgol.get():
            filters['savgol_params'] = {
                'window_length': self.savgol_window.get(),
                'polyorder': self.savgol_order.get()
            }
        
        if self.use_fft.get():
            filters['fft_threshold'] = self.fft_threshold.get()
        
        if self.use_butterworth.get():
            filters['butter_params'] = {
                'cutoff': self.butter_cutoff.get(),
                'order': self.butter_order.get(),
                'fs': 1000.0  # Default sampling frequency
            }
        
        if self.use_extract_add.get():
            filters['extract_add_params'] = {
                'prominence_threshold': self.extract_prominence.get(),
                'width_range': (self.extract_width_min.get(), 
                              self.extract_width_max.get())
            }
        
        return filters

    def on_filter_change(self, *args):
        """Called when any filter parameter changes"""
        app_logger.debug("Filter parameters changed")
        try:
            # Get current filter settings and update plot
            self.update_callback(self.get_active_filters())
        except Exception as e:
            app_logger.error(f"Error updating filters: {str(e)}")

    def reset_filters(self):
        """Reset all filters to default values"""
        self.use_savgol.set(False)
        self.use_fft.set(False)
        self.use_butterworth.set(False)
        self.use_extract_add.set(False)
        
        # Reset parameters to defaults
        self.savgol_window.set(51)
        self.savgol_order.set(3)
        self.fft_threshold.set(0.2)
        self.butter_cutoff.set(0.1)
        self.butter_order.set(5)
        self.extract_prominence.set(200)
        self.extract_width_min.set(1)
        self.extract_width_max.set(50)
        
        # Update plot
        self.on_filter_change()