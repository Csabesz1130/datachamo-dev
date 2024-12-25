import numpy as np
from scipy import signal
from scipy.integrate import simps
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self, data, time_data, params=None):
        self.data = np.array(data)  # Should be in pA
        self.time_data = np.array(time_data)  # Should be in seconds
        self.params = params or {
            'n_cycles': 2,
            't1': 100,  # ms
            't2': 100,  # ms
            'V0': -80,  # mV
            'V1': 100,  # mV
            'V2': 10,   # mV
            'cell_area_cm2': 1.0  # Default membrane area in cm²
        }
        
        self.processed_data = None
        self.baseline = None
        self.cycles = []
        self.cycle_times = []
        self.cycle_indices = []
        
        app_logger.debug(f"Parameters validated: {self.params}")

    def baseline_correction(self):
        baseline_window = self.data[:1000]
        self.baseline = np.median(baseline_window)
        self.processed_data = self.data - self.baseline
        app_logger.info(f"Baseline correction applied: {self.baseline:.2f}")

    def find_cycles(self):
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

    def process_signal(self):
        """Process the signal and return results."""
        try:
            self.baseline_correction()
            self.find_cycles()
            self.normalize_signal()
            results = self.calculate_integral()
            
            if not results:
                return None, None, {
                    'integral_value': 'No analysis performed',
                    'capacitance_uF_cm2': 'No analysis performed',
                    'cycle_indices': []
                }
                
            return self.processed_data, self.time_data, results
            
        except Exception as e:
            app_logger.error(f"Error in signal processing: {str(e)}")
            return None, None, {
                'integral_value': f'Error: {str(e)}',
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': []
            }

    def calculate_integral(self):
        """Calculate integral and capacitance with proper unit handling."""
        try:
            if not self.cycles:
                return {
                    'integral_value': 'No cycles found',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': []
                }

            # Get first cycle
            cycle = self.cycles[0]
            cycle_time = self.cycle_times[0]

            # Unit conversions
            current_in_A = cycle * 1e-12  # pA to A
            time_in_s = cycle_time  # Already in seconds from data loading
            voltage_diff_in_V = (self.params['V1'] - self.params['V0']) * 1e-3  # mV to V

            # Find event window using appropriate threshold
            peak_current = np.max(np.abs(current_in_A))
            threshold = 0.1 * peak_current  # 10% of peak
            event_mask = np.abs(current_in_A) > threshold

            if not np.any(event_mask):
                return {
                    'integral_value': '0.0000 C',
                    'capacitance_uF_cm2': '0.0000 µF/cm²',
                    'cycle_indices': self.cycle_indices
                }

            # Calculate charge
            charge_C = np.trapz(current_in_A[event_mask], time_in_s[event_mask])
            
            # Calculate capacitance
            total_capacitance_F = abs(charge_C / voltage_diff_in_V)
            total_capacitance_uF = total_capacitance_F * 1e6  # Convert F to µF

            # Use realistic cell area (typical patch size ~100 µm²)
            area_cm2 = self.params.get('cell_area_cm2', 1e-4)  # 100 µm² = 1e-4 cm²
            capacitance_uF_cm2 = total_capacitance_uF / area_cm2

            results = {
                'integral_value': f"{abs(charge_C):.6e} C",
                'capacitance_uF_cm2': f"{capacitance_uF_cm2:.4f} µF/cm²",
                'cycle_indices': self.cycle_indices,
                'raw_values': {
                    'charge_C': charge_C,
                    'capacitance_F': total_capacitance_F,
                    'area_cm2': area_cm2
                }
            }

            app_logger.info(f"Integrated charge: {charge_C:.2e} C, Capacitance: {capacitance_uF_cm2:.4f} µF/cm²")
            return results

        except Exception as e:
            app_logger.error(f"Error calculating integral: {str(e)}")
            return {
                'integral_value': 'Error in calculation',
                'capacitance_uF_cm2': 'Error',
                'cycle_indices': self.cycle_indices
            }