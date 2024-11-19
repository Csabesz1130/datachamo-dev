# src/visualization/enhanced_oscilloscope.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from src.utils.logger import app_logger

class EnhancedOscilloscopeView:
    def __init__(self, master, figure_size=(10, 6)):
        """Initialize enhanced oscilloscope view"""
        self.master = master
        
        # Create main frame with tabs
        self.frame = ttk.Frame(master)
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Setup tabs
        self.setup_scope_tab()
        self.setup_measurement_tab()
        self.setup_regression_tab()
        
        # Initialize cursor variables
        self.cursors = {'x': None, 'y': None}
        self.cursor_lines = {'x1': None, 'x2': None, 'y1': None, 'y2': None}
        self.cursor_active = False
        
        app_logger.debug("Enhanced oscilloscope view initialized")

    def setup_scope_tab(self):
        """Setup main oscilloscope display tab"""
        scope_frame = ttk.Frame(self.notebook)
        self.notebook.add(scope_frame, text="Scope")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=self.figure_size, dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Setup canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=scope_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, scope_frame)
        self.toolbar.update()
        
        # Add cursor controls
        cursor_frame = ttk.LabelFrame(scope_frame, text="Cursors")
        cursor_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(cursor_frame, text="Add Time Cursor",
                  command=self.add_time_cursor).pack(side='left', padx=5)
        ttk.Button(cursor_frame, text="Add Voltage Cursor",
                  command=self.add_voltage_cursor).pack(side='left', padx=5)
        ttk.Button(cursor_frame, text="Clear Cursors",
                  command=self.clear_cursors).pack(side='left', padx=5)

    def setup_measurement_tab(self):
        """Setup measurements tab"""
        meas_frame = ttk.Frame(self.notebook)
        self.notebook.add(meas_frame, text="Measurements")
        
        # Create measurements display
        self.meas_text = tk.Text(meas_frame, height=10, width=50)
        self.meas_text.pack(padx=5, pady=5)
        
        # Add measurement controls
        control_frame = ttk.Frame(meas_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Measure Peak-Peak",
                  command=self.measure_peak_peak).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Measure Period",
                  command=self.measure_period).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Signal Statistics",
                  command=self.show_statistics).pack(side='left', padx=5)

    def setup_regression_tab(self):
        """Setup regression analysis tab"""
        reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(reg_frame, text="Regression")
        
        # Regression controls
        control_frame = ttk.LabelFrame(reg_frame, text="Linear Regression Controls")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Window size control
        window_frame = ttk.Frame(control_frame)
        window_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(window_frame, text="Initial Points:").pack(side='left')
        self.window_var = tk.IntVar(value=100)
        ttk.Entry(window_frame, textvariable=self.window_var,
                 width=10).pack(side='left', padx=5)
        
        # Blend control
        self.blend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Smooth Transition",
                       variable=self.blend_var).pack(padx=5, pady=2)
        
        # Apply button
        ttk.Button(control_frame, text="Apply Regression",
                  command=self.apply_regression).pack(pady=5)
        
        # Results display
        self.reg_text = tk.Text(reg_frame, height=10, width=50)
        self.reg_text.pack(padx=5, pady=5)

    def add_time_cursor(self):
        """Add vertical time measurement cursor"""
        if not self.cursor_active:
            self.cursor_active = True
            self.canvas.mpl_connect('button_press_event', self.on_click_time)
            self.master.config(cursor="crosshair")

    def add_voltage_cursor(self):
        """Add horizontal voltage measurement cursor"""
        if not self.cursor_active:
            self.cursor_active = True
            self.canvas.mpl_connect('button_press_event', self.on_click_voltage)
            self.master.config(cursor="crosshair")

    def on_click_time(self, event):
        """Handle time cursor placement"""
        if event.inaxes != self.ax:
            return
            
        if self.cursor_lines['x1'] is None:
            self.cursor_lines['x1'] = self.ax.axvline(x=event.xdata, color='r', linestyle='--')
        elif self.cursor_lines['x2'] is None:
            self.cursor_lines['x2'] = self.ax.axvline(x=event.xdata, color='r', linestyle='--')
            self.measure_time_difference()
            self.cursor_active = False
            self.master.config(cursor="")
        
        self.canvas.draw()

    def on_click_voltage(self, event):
        """Handle voltage cursor placement"""
        if event.inaxes != self.ax:
            return
            
        if self.cursor_lines['y1'] is None:
            self.cursor_lines['y1'] = self.ax.axhline(y=event.ydata, color='b', linestyle='--')
        elif self.cursor_lines['y2'] is None:
            self.cursor_lines['y2'] = self.ax.axhline(y=event.ydata, color='b', linestyle='--')
            self.measure_voltage_difference()
            self.cursor_active = False
            self.master.config(cursor="")
        
        self.canvas.draw()

    def measure_time_difference(self):
        """Measure time difference between cursors"""
        if self.cursor_lines['x1'] and self.cursor_lines['x2']:
            x1 = self.cursor_lines['x1'].get_xdata()[0]
            x2 = self.cursor_lines['x2'].get_xdata()[0]
            dt = abs(x2 - x1)
            self.update_measurements(f"Time Difference: {dt:.3f} ms")

    def measure_voltage_difference(self):
        """Measure voltage difference between cursors"""
        if self.cursor_lines['y1'] and self.cursor_lines['y2']:
            y1 = self.cursor_lines['y1'].get_ydata()[0]
            y2 = self.cursor_lines['y2'].get_ydata()[0]
            dv = abs(y2 - y1)
            self.update_measurements(f"Voltage Difference: {dv:.3f} mV")

    def update_measurements(self, text):
        """Update measurements display"""
        self.meas_text.insert(tk.END, text + "\n")
        self.meas_text.see(tk.END)

    def apply_regression(self):
        """Apply linear regression to initial points"""
        if not hasattr(self, 'data') or self.data is None:
            return
            
        try:
            from src.filtering.regression_filter import LinearRegressionFilter
            
            # Create and apply filter
            reg_filter = LinearRegressionFilter(window_size=self.window_var.get())
            filtered_data = reg_filter.apply_regression_filter(
                self.data,
                self.time_data,
                blend=self.blend_var.get()
            )
            
            # Plot result
            self.ax.plot(self.time_data * 1000, filtered_data, 'r-',
                        label='Regression Fit', alpha=0.7)
            self.canvas.draw()
            
            # Show statistics
            stats = reg_filter.get_regression_stats()
            self.reg_text.delete(1.0, tk.END)
            self.reg_text.insert(tk.END, 
                f"Regression Results:\n"
                f"Slope: {stats['slope']:.4f} mV/ms\n"
                f"Intercept: {stats['intercept']:.4f} mV\n"
                f"Initial Trend: {stats['initial_trend']}\n"
            )
            
        except Exception as e:
            app_logger.error(f"Error applying regression: {str(e)}")
            raise

    def clear_cursors(self):
        """Clear all measurement cursors"""
        for cursor in self.cursor_lines.values():
            if cursor is not None:
                cursor.remove()
        self.cursor_lines = {'x1': None, 'x2': None, 'y1': None, 'y2': None}
        self.cursor_active = False
        self.master.config(cursor="")
        self.canvas.draw()

    def set_data(self, time_data, voltage_data):
        """
        Set new data for display
        
        Args:
            time_data: Array of time points
            voltage_data: Array of voltage values
        """
        try:
            self.data = voltage_data
            self.time_data = time_data
            self.update_plot()
            self.update_statistics()
        except Exception as e:
            app_logger.error(f"Error setting data: {str(e)}")

    def update_plot(self):
        """Update the oscilloscope display"""
        try:
            if not hasattr(self, 'data') or self.data is None:
                return

            self.ax.clear()
            
            # Plot grid
            self.ax.grid(True, which='major', linestyle='-', alpha=0.5)
            self.ax.grid(True, which='minor', linestyle=':', alpha=0.2)
            
            # Plot threshold lines
            self.plot_thresholds()
            
            # Plot signal
            self.ax.plot(self.time_data * 1000, self.data, 'k-', 
                        linewidth=1, label='Signal')
            
            # Set labels and title
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Voltage (mV)')
            self.ax.set_title('Signal Analysis')
            
            # Restore any active cursors
            self.restore_cursors()
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            app_logger.error(f"Error updating plot: {str(e)}")

    def plot_thresholds(self):
        """Plot threshold lines on the oscilloscope"""
        try:
            # Get current threshold values
            vh = getattr(self, 'vh_var', tk.DoubleVar(value=-80)).get()
            vn = getattr(self, 'vn_var', tk.DoubleVar(value=-100)).get()
            vt = getattr(self, 'vt_var', tk.DoubleVar(value=10)).get()
            
            # Plot threshold lines
            self.ax.axhline(y=vh, color='b', linestyle='--', 
                          alpha=0.5, label=f'VH ({vh}mV)')
            self.ax.axhline(y=vn, color='r', linestyle='--', 
                          alpha=0.5, label=f'VN ({vn}mV)')
            self.ax.axhline(y=vt, color='g', linestyle='--', 
                          alpha=0.5, label=f'VT ({vt}mV)')
            
            self.ax.legend()
            
        except Exception as e:
            app_logger.error(f"Error plotting thresholds: {str(e)}")

    def restore_cursors(self):
        """Restore any active measurement cursors after plot update"""
        try:
            for cursor_type, line in self.cursor_lines.items():
                if line is not None:
                    if cursor_type.startswith('x'):
                        x_val = line.get_xdata()[0]
                        self.cursor_lines[cursor_type] = self.ax.axvline(
                            x=x_val, color='r', linestyle='--'
                        )
                    else:
                        y_val = line.get_ydata()[0]
                        self.cursor_lines[cursor_type] = self.ax.axhline(
                            y=y_val, color='b', linestyle='--'
                        )
        except Exception as e:
            app_logger.error(f"Error restoring cursors: {str(e)}")

    def measure_peak_peak(self):
        """Measure and display peak-to-peak voltage"""
        try:
            if not hasattr(self, 'data') or self.data is None:
                return
                
            vpp = np.ptp(self.data)
            vmax = np.max(self.data)
            vmin = np.min(self.data)
            
            self.update_measurements(
                f"Peak-to-Peak Measurements:\n"
                f"Vpp: {vpp:.2f} mV\n"
                f"Vmax: {vmax:.2f} mV\n"
                f"Vmin: {vmin:.2f} mV\n"
            )
            
        except Exception as e:
            app_logger.error(f"Error measuring peak-to-peak: {str(e)}")

    def measure_period(self):
        """Measure signal period using zero crossings"""
        try:
            if not hasattr(self, 'data') or self.data is None:
                return
                
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.signbit(self.data)))[0]
            
            if len(zero_crossings) >= 2:
                # Calculate average period
                periods = np.diff(zero_crossings)
                avg_period = np.mean(periods) * (self.time_data[1] - self.time_data[0]) * 1000
                frequency = 1000 / avg_period  # Convert to Hz
                
                self.update_measurements(
                    f"Period Measurements:\n"
                    f"Average Period: {avg_period:.2f} ms\n"
                    f"Frequency: {frequency:.2f} Hz\n"
                )
            else:
                self.update_measurements("Insufficient zero crossings for period measurement\n")
                
        except Exception as e:
            app_logger.error(f"Error measuring period: {str(e)}")

    def show_statistics(self):
        """Display comprehensive signal statistics"""
        try:
            if not hasattr(self, 'data') or self.data is None:
                return
                
            stats = {
                'Mean': np.mean(self.data),
                'Std Dev': np.std(self.data),
                'RMS': np.sqrt(np.mean(np.square(self.data))),
                'Peak-Peak': np.ptp(self.data),
                'Skewness': float(np.mean(((self.data - np.mean(self.data)) / np.std(self.data)) ** 3)),
                'Kurtosis': float(np.mean(((self.data - np.mean(self.data)) / np.std(self.data)) ** 4))
            }
            
            stats_text = "Signal Statistics:\n"
            for name, value in stats.items():
                stats_text += f"{name}: {value:.2f}\n"
            
            self.meas_text.delete(1.0, tk.END)
            self.meas_text.insert(tk.END, stats_text)
            
        except Exception as e:
            app_logger.error(f"Error showing statistics: {str(e)}")

    def export_data(self):
        """Export measurement data to CSV"""
        try:
            if not hasattr(self, 'data') or self.data is None:
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                data_dict = {
                    'Time_ms': self.time_data * 1000,
                    'Voltage_mV': self.data
                }
                
                # If regression fit exists, include it
                if hasattr(self, 'regression_fit'):
                    data_dict['Regression_Fit'] = self.regression_fit
                
                import pandas as pd
                df = pd.DataFrame(data_dict)
                df.to_csv(filename, index=False)
                
                app_logger.info(f"Data exported to {filename}")
                
        except Exception as e:
            app_logger.error(f"Error exporting data: {str(e)}")

    def export_figure(self):
        """Export current figure as image"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                app_logger.info(f"Figure exported to {filename}")
                
        except Exception as e:
            app_logger.error(f"Error exporting figure: {str(e)}")

    def add_export_controls(self):
        """Add export control buttons"""
        export_frame = ttk.LabelFrame(self.frame, text="Export")
        export_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Data",
                  command=self.export_data).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Export Figure",
                  command=self.export_figure).pack(side='left', padx=5)

    def update_statistics(self):
        """Update statistics display"""
        if self.notebook.select() == self.notebook.tabs()[1]:  # Measurements tab
            self.show_statistics()

    def on_tab_changed(self, event):
        """Handle tab change events"""
        try:
            current_tab = self.notebook.select()
            if current_tab == self.notebook.tabs()[1]:  # Measurements tab
                self.update_statistics()
            elif current_tab == self.notebook.tabs()[2]:  # Regression tab
                if hasattr(self, 'data') and self.data is not None:
                    self.apply_regression()
        except Exception as e:
            app_logger.error(f"Error handling tab change: {str(e)}")

    def __init__(self, master, figure_size=(10, 6)):
    """
    Initialize enhanced oscilloscope view with all controls and displays
    
    Args:
        master: Parent tkinter widget
        figure_size: Tuple of (width, height) for the figure
    """
    self.master = master
    self.figure_size = figure_size
    
    # Create main frame
    self.frame = ttk.Frame(master)
    self.frame.pack(fill='both', expand=True, padx=5, pady=5)
    
    # Create notebook for tabs
    self.notebook = ttk.Notebook(self.frame)
    self.notebook.pack(fill='both', expand=True)
    
    # Initialize data attributes
    self.data = None
    self.time_data = None
    self.regression_fit = None
    
    # Initialize cursor variables
    self.cursors = {'x': None, 'y': None}
    self.cursor_lines = {'x1': None, 'x2': None, 'y1': None, 'y2': None}
    self.cursor_active = False
    
    # Setup main scope tab
    self.scope_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.scope_frame, text="Scope")
    
    # Create matplotlib figure
    self.fig = Figure(figsize=self.figure_size, dpi=100)
    self.ax = self.fig.add_subplot(111)
    
    # Setup canvas
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.scope_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.scope_frame)
    self.toolbar.update()
    
    # Add scope controls
    self.setup_scope_controls()
    
    # Setup measurements tab
    self.meas_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.meas_frame, text="Measurements")
    
    # Create measurements display
    self.meas_text = tk.Text(self.meas_frame, height=10, width=50)
    self.meas_text.pack(padx=5, pady=5)
    
    # Add measurement controls
    self.setup_measurement_controls()
    
    # Setup regression tab
    self.reg_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.reg_frame, text="Regression")
    
    # Add regression controls
    self.setup_regression_controls()
    
    # Create variables for scope settings
    self.setup_scope_variables()
    
    # Add export controls
    self.add_export_controls()
    
    # Bind events
    self.bind_events()
    
    app_logger.debug("Enhanced oscilloscope view initialized")

def setup_scope_controls(self):
    """Setup oscilloscope display controls"""
    control_frame = ttk.LabelFrame(self.scope_frame, text="Display Controls")
    control_frame.pack(fill='x', padx=5, pady=5)
    
    # Time base control
    time_frame = ttk.Frame(control_frame)
    time_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(time_frame, text="Time/div (ms):").pack(side='left')
    self.time_scale = ttk.Scale(time_frame, from_=0.1, to=1000,
                               orient='horizontal',
                               command=self.update_plot)
    self.time_scale.set(100)
    self.time_scale.pack(side='left', fill='x', expand=True)
    
    # Voltage scale control
    volt_frame = ttk.Frame(control_frame)
    volt_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(volt_frame, text="Voltage/div (mV):").pack(side='left')
    self.volt_scale = ttk.Scale(volt_frame, from_=1, to=500,
                               orient='horizontal',
                               command=self.update_plot)
    self.volt_scale.set(20)
    self.volt_scale.pack(side='left', fill='x', expand=True)
    
    # Cursor controls
    cursor_frame = ttk.LabelFrame(control_frame, text="Cursors")
    cursor_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Button(cursor_frame, text="Add Time Cursor",
               command=self.add_time_cursor).pack(side='left', padx=5)
    ttk.Button(cursor_frame, text="Add Voltage Cursor",
               command=self.add_voltage_cursor).pack(side='left', padx=5)
    ttk.Button(cursor_frame, text="Clear Cursors",
               command=self.clear_cursors).pack(side='left', padx=5)

def setup_measurement_controls(self):
    """Setup measurement controls"""
    control_frame = ttk.Frame(self.meas_frame)
    control_frame.pack(fill='x', padx=5, pady=5)
    
    ttk.Button(control_frame, text="Measure Peak-Peak",
               command=self.measure_peak_peak).pack(side='left', padx=5)
    ttk.Button(control_frame, text="Measure Period",
               command=self.measure_period).pack(side='left', padx=5)
    ttk.Button(control_frame, text="Signal Statistics",
               command=self.show_statistics).pack(side='left', padx=5)

def setup_regression_controls(self):
    """Setup regression analysis controls"""
    control_frame = ttk.LabelFrame(self.reg_frame, text="Regression Controls")
    control_frame.pack(fill='x', padx=5, pady=5)
    
    # Window size control
    window_frame = ttk.Frame(control_frame)
    window_frame.pack(fill='x', padx=5, pady=2)
    ttk.Label(window_frame, text="Initial Points:").pack(side='left')
    self.window_var = tk.IntVar(value=100)
    ttk.Entry(window_frame, textvariable=self.window_var,
              width=10).pack(side='left', padx=5)
    
    # Blend control
    self.blend_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(control_frame, text="Smooth Transition",
                    variable=self.blend_var).pack(padx=5, pady=2)
    
    # Apply button
    ttk.Button(control_frame, text="Apply Regression",
               command=self.apply_regression).pack(pady=5)
    
    # Results display
    self.reg_text = tk.Text(self.reg_frame, height=10, width=50)
    self.reg_text.pack(padx=5, pady=5)

def setup_scope_variables(self):
    """Initialize scope settings variables"""
    # Threshold variables
    self.vh_var = tk.DoubleVar(value=-80)  # Holding voltage
    self.vn_var = tk.DoubleVar(value=-100)  # Negative threshold
    self.vt_var = tk.DoubleVar(value=10)   # Trigger voltage
    
    # Display options
    self.show_grid = tk.BooleanVar(value=True)
    self.show_thresholds = tk.BooleanVar(value=True)
    self.auto_scale = tk.BooleanVar(value=True)

def bind_events(self):
    """Bind various events to callbacks"""
    # Tab change event
    self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
    
    # Canvas events
    self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    self.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    # Window resize event
    self.frame.bind('<Configure>', self.on_resize)