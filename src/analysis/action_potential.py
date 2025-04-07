import numpy as np
from scipy import signal
from scipy.integrate import simps
from scipy.stats import linregress
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self):
        """Initialize the action potential processor"""
        self.data = None
        self.time_data = None
        self.params = {
            'n_cycles': 2,
            't1': 100,  # ms
            't2': 100,  # ms
            'V0': -80,  # mV
            'V1': 100,  # mV
            'V2': 10,   # mV
            'cell_area_cm2': 1.0  # Default membrane area in cm²
        }
        
        # Processed data
        self.processed_data = None
        self.baseline = None
        self.cycles = []
        self.cycle_times = []
        self.cycle_indices = []
        
        # Derived curves
        self.blue_curve = None
        self.magenta_curve = None
        self.purple_hyperpol_curve = None
        self.purple_depol_curve = None
        
        # Status bar
        self.status_callback = None
        
        app_logger.debug("Action potential processor initialized")

    def set_data(self, data, time_data):
        """Set input data"""
        self.data = np.array(data)
        self.time_data = np.array(time_data)
        app_logger.debug("Input data set")

    def set_status_callback(self, callback):
        """Set status bar update callback"""
        self.status_callback = callback

    def update_status(self, text):
        """Update status bar text"""
        if self.status_callback:
            self.status_callback(text)

    def analyze(self, **params):
        """Perform action potential analysis"""
        try:
            # Update parameters
            self.params.update(params)
            
            # Process signal
            self.baseline_correction()
            self.find_cycles()
            self.normalize_signal()
            
            # Calculate derived curves
            self.calculate_blue_curve()
            self.calculate_magenta_curve()
            self.calculate_purple_curves()
            
            # Calculate results
            results = self.calculate_integral()
            
            app_logger.info("Analysis completed successfully")
            return self.processed_data
            
        except Exception as e:
            app_logger.error(f"Error in analysis: {str(e)}")
            raise

    def baseline_correction(self):
        """Apply baseline correction"""
        baseline_window = self.data[:1000]
        self.baseline = np.median(baseline_window)
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Baseline correction applied: {self.baseline:.2f}")

    def find_cycles(self):
        """Find action potential cycles"""
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t1_samples = int(self.params['t1'] * sampling_rate / 1000)
        
        # Find peaks with proper threshold
        peaks, _ = signal.find_peaks(-self.processed_data,
                                   prominence=np.std(self.processed_data),
                                   distance=t1_samples)
        
        for i, peak in enumerate(peaks[:self.params['n_cycles']]):
            start = max(0, peak - t1_samples//2)
            end = min(len(self.processed_data), peak + t1_samples*2)
            
            cycle = self.processed_data[start:end]
            cycle_time = self.time_data[start:end] - self.time_data[start]
            
            self.cycles.append(cycle)
            self.cycle_times.append(cycle_time)
            self.cycle_indices.append((start, end))
            app_logger.debug(f"Valid cycle found at index {peak}")

    def normalize_signal(self):
        """Normalize signal to voltage range"""
        if not self.cycles:
            return
            
        baselines = [np.median(cycle[:100]) for cycle in self.cycles]
        baseline = np.mean(baselines)
        min_val = min(np.min(cycle) for cycle in self.cycles)
        
        # Scale to V0-V1 range in mV
        scale_factor = abs(self.params['V1'] - self.params['V0']) / abs(min_val - baseline)
        self.processed_data = (self.processed_data - baseline) * scale_factor + self.params['V0']
        
        # Update cycles with normalized data
        for i, (start, end) in enumerate(self.cycle_indices):
            self.cycles[i] = self.processed_data[start:end]
        
        app_logger.info(f"Signal normalized with scale factor: {scale_factor:.4f}")

    def calculate_blue_curve(self):
        """Calculate voltage-normalized curve"""
        if not self.processed_data is None:
            # Take points 28-827 (800 points)
            self.blue_curve = self.processed_data[28:828]
            app_logger.debug("Blue curve calculated")

    def calculate_magenta_curve(self):
        """Calculate averaged normalized curve"""
        if not self.blue_curve is None:
            # Take points 28-228 (200 points) and average
            self.magenta_curve = np.mean(self.blue_curve[:200])
            app_logger.debug("Magenta curve calculated")

    def calculate_purple_curves(self):
        """Calculate modified peak curves"""
        if not self.processed_data is None:
            # Hyperpolarization curve (points 1028-1227)
            self.purple_hyperpol_curve = self.processed_data[1028:1227]
            
            # Depolarization curve (points 828-1028)
            self.purple_depol_curve = self.processed_data[828:1028]
            
            app_logger.debug("Purple curves calculated")

    def calculate_integral(self):
        """Calculate integral and capacitance"""
        try:
            if not self.cycles:
                return {
                    'integral_value': 'No cycles found',
                    'capacitance_uF_cm2': 'No cycles found',
                    'cycle_indices': []
                }
                
            # Calculate integral for each cycle
            integrals = []
            for cycle, cycle_time in zip(self.cycles, self.cycle_times):
                integral = simps(cycle, cycle_time)
                integrals.append(integral)
                
            # Calculate average integral
            avg_integral = np.mean(integrals)
            
            # Calculate capacitance
            capacitance = avg_integral / (self.params['V1'] - self.params['V0'])
            capacitance_per_area = capacitance / self.params['cell_area_cm2']
            
            return {
                'integral_value': f"{avg_integral:.2f} pA·ms",
                'capacitance_uF_cm2': f"{capacitance_per_area:.2f} μF/cm²",
                'cycle_indices': self.cycle_indices
            }
            
        except Exception as e:
            app_logger.error(f"Error calculating integral: {str(e)}")
            return {
                'integral_value': f'Error: {str(e)}',
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def get_regression_line(self, start_idx, end_idx):
        """Calculate regression line for given range"""
        try:
            x = self.time_data[start_idx:end_idx]
            y = self.processed_data[start_idx:end_idx]
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'x': x,
                'y': slope * x + intercept
            }
            
        except Exception as e:
            app_logger.error(f"Error calculating regression: {str(e)}")
            return None