import numpy as np
from scipy import signal
from scipy.integrate import simps
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self, data, time_data, params=None):
        self.data = np.array(data)
        self.time_data = np.array(time_data)
        self.params = params or {
            'n_cycles': 2,
            't1': 100,  # ms
            't2': 100,  # ms
            'V0': -80,  # mV
            'V1': 100,  # mV
            'V2': 10    # mV
        }
        
        self.processed_data = None
        self.baseline = None
        self.results = None
        
    def process_signal(self):
        self.baseline_correction()
        self.find_cycles()
        self.normalize_signal()
        results = self.calculate_integral()
        return self.processed_data, self.time_data, results
        
    def baseline_correction(self):
        baseline_window = self.data[:1000]
        self.baseline = np.median(baseline_window)
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Baseline correction applied: {self.baseline:.2f}")

    def find_cycles(self):
        sampling_rate = 1.0 / np.mean(np.diff(self.time_data))
        t1_samples = int(self.params['t1'] * sampling_rate / 1000)
        
        # Find peaks and store indices
        peaks, _ = signal.find_peaks(-self.processed_data,
                                   prominence=np.std(self.processed_data),
                                   distance=t1_samples)
        
        self.cycle_indices = []
        for i, peak in enumerate(peaks[:self.params['n_cycles']]):
            start = max(0, peak - t1_samples//2)
            end = min(len(self.processed_data), peak + t1_samples*2)
            self.cycle_indices.append((start, end))

    def normalize_signal(self):
        if not hasattr(self, 'cycle_indices'):
            self.find_cycles()
            
        # Get baseline from pre-cycle regions
        baselines = []
        for start, _ in self.cycle_indices:
            pre_cycle = self.processed_data[max(0, start-100):start]
            baselines.append(np.median(pre_cycle))
        baseline = np.mean(baselines)
        
        # Find hyperpolarization amplitude
        min_vals = []
        for start, end in self.cycle_indices:
            cycle = self.processed_data[start:end]
            min_vals.append(np.min(cycle))
        min_val = np.mean(min_vals)
        
        # Scale to V0-V1 range
        scale_factor = abs(self.params['V1'] - self.params['V0']) / abs(min_val - baseline)
        self.processed_data = (self.processed_data - baseline) * scale_factor
        
        app_logger.info(f"Signal normalized with scale factor: {scale_factor:.4f}")

    def calculate_integral(self):
        if not hasattr(self, 'cycle_indices'):
            return {'integral': 0, 'capacitance': 0}
        
        # Use first cycle
        start, end = self.cycle_indices[0]
        cycle = self.processed_data[start:end]
        cycle_time = self.time_data[start:end] - self.time_data[start]
        
        # Calculate integral and capacitance
        integral = np.trapz(cycle, cycle_time)
        capacitance = abs(integral / (self.params['V1'] - self.params['V0'])) * 1e6
        
        self.results = {
            'integral_value': f"{abs(integral):.6f}",
            'capacitance_value': f"{capacitance:.2f}",
            'cycle_indices': self.cycle_indices,
        }
        
        app_logger.info(f"Integral: {integral:.6f} V·s, Capacitance: {capacitance:.2f} µF/cm²")
        return self.results