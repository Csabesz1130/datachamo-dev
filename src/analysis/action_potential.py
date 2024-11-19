import numpy as np
from scipy import stats
from src.utils.logger import app_logger

class ActionPotentialProcessor:
    def __init__(self, data, time_data, params=None):
        """Initialize the processor with data and parameters."""
        try:
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
            
            # Calculate voltage differences
            self.delta_V1 = self.params['V1'] - self.params['V0']
            self.delta_V2 = self.params['V2'] - self.params['V0']
            
            self.processed_data = None
            self.processed_time = None
            self.cycles = []
            self.cycle_times = []
            self.baseline = None
            
            app_logger.info("ActionPotentialProcessor initialized with parameters:")
            app_logger.debug(f"Data shape: {self.data.shape}")
            app_logger.debug(f"Time data shape: {self.time_data.shape}")
            app_logger.debug(f"Parameters: {self.params}")
            
        except Exception as e:
            app_logger.error(f"Error initializing ActionPotentialProcessor: {str(e)}")
            raise

    def find_signal_start(self, window_size=20):
        """Find the start of the action potential signal."""
        try:
            if len(self.data) < window_size:
                app_logger.error(f"Data length ({len(self.data)}) is smaller than window_size ({window_size})")
                raise ValueError("Data length is too small for the specified window size")

            slopes = []
            r_values = []
            
            # Calculate slopes for each window
            for i in range(len(self.data) - window_size):
                window_time = self.time_data[i:i+window_size]
                window_data = self.data[i:i+window_size]
                
                slope, intercept, r_value, _, _ = stats.linregress(window_time, window_data)
                slopes.append(slope)
                r_values.append(r_value)
            
            slopes = np.array(slopes)
            slope_changes = np.diff(slopes)
            
            # Find significant changes
            threshold = np.std(slope_changes) * 3
            change_points = np.where(np.abs(slope_changes) > threshold)[0]
            
            if len(change_points) == 0:
                app_logger.warning("No significant slope changes found, using start of data")
                return 0
                
            start_idx = change_points[0]
            app_logger.info(f"Signal start detected at index {start_idx}")
            return start_idx
            
        except Exception as e:
            app_logger.error(f"Error in find_signal_start: {str(e)}")
            raise

    def find_cycles(self):
        """Find and extract individual action potential cycles."""
        try:
            start_idx = self.find_signal_start()
            
            # Calculate minimum points per cycle based on time constants
            sampling_period = np.mean(np.diff(self.time_data))
            min_points_per_cycle = int((self.params['t1'] + self.params['t2']) / 
                                     (sampling_period * 1000))  # Convert to ms
            
            app_logger.debug(f"Minimum points per cycle: {min_points_per_cycle}")
            
            # Reset cycles
            self.cycles = []
            self.cycle_times = []
            current_cycle = []
            current_time = []
            last_cycle_start = start_idx
            
            # Process signal to find cycles
            data = self.processed_data if self.processed_data is not None else self.data
            
            for i in range(start_idx, len(data) - 1):
                if len(self.cycles) >= self.params['n_cycles']:
                    break
                
                current_cycle.append(data[i])
                current_time.append(self.time_data[i])
                
                # Check for cycle completion
                if len(current_cycle) >= min_points_per_cycle:
                    cycle_data = np.array(current_cycle)
                    if np.ptp(cycle_data) > np.std(data) * 2:  # Significant variation
                        self.cycles.append(np.array(current_cycle))
                        self.cycle_times.append(np.array(current_time))
                        current_cycle = []
                        current_time = []
                        last_cycle_start = i
                        app_logger.debug(f"Cycle detected at index {i}")
            
            # Add last cycle if valid
            if len(current_cycle) >= min_points_per_cycle:
                self.cycles.append(np.array(current_cycle))
                self.cycle_times.append(np.array(current_time))
            
            app_logger.info(f"Found {len(self.cycles)} complete cycles")
            
            if len(self.cycles) == 0:
                raise ValueError("No valid cycles found in the data")
                
            return self.cycles, self.cycle_times
            
        except Exception as e:
            app_logger.error(f"Error in find_cycles: {str(e)}")
            raise

    def baseline_correction(self, n_points=20):
        """Correct baseline using the last n points."""
        try:
            if len(self.data) < n_points:
                raise ValueError(f"Data length ({len(self.data)}) is smaller than n_points ({n_points})")
                
            self.baseline = np.mean(self.data[-n_points:])
            self.processed_data = self.data - self.baseline
            app_logger.info(f"Baseline correction applied: {self.baseline:.2f}")
            
        except Exception as e:
            app_logger.error(f"Error in baseline_correction: {str(e)}")
            raise

    def normalize_signal(self):
        """Normalize the signal using delta_V1."""
        try:
            if self.processed_data is None:
                self.baseline_correction()
            
            if self.delta_V1 == 0:
                raise ValueError("delta_V1 is zero, cannot normalize")
                
            self.processed_data = self.processed_data / self.delta_V1
            app_logger.info("Signal normalized using delta_V1")
            
        except Exception as e:
            app_logger.error(f"Error in normalize_signal: {str(e)}")
            raise

    def average_cycles(self):
        """Average the detected cycles."""
        try:
            if not self.cycles:
                self.find_cycles()
            
            if len(self.cycles) == 0:
                raise ValueError("No cycles available for averaging")
            
            # Interpolate cycles to common time base
            min_time = min(t[0] for t in self.cycle_times)
            max_time = max(t[-1] for t in self.cycle_times)
            target_points = min(len(cycle) for cycle in self.cycles)
            common_time = np.linspace(min_time, max_time, target_points)
            
            # Interpolate each cycle to common time base
            interpolated_cycles = []
            for cycle, time in zip(self.cycles, self.cycle_times):
                interpolated = np.interp(common_time, time, cycle)
                interpolated_cycles.append(interpolated)
            
            # Calculate average
            self.processed_data = np.mean(interpolated_cycles, axis=0)
            self.processed_time = common_time
            
            app_logger.info(f"Cycles averaged successfully. Result shape: {self.processed_data.shape}")
            return self.processed_data, self.processed_time
            
        except Exception as e:
            app_logger.error(f"Error in average_cycles: {str(e)}")
            raise

    def calculate_integral(self):
        """Calculate the integral of the processed signal."""
        try:
            if self.processed_data is None or self.processed_time is None:
                self.average_cycles()
            
            # Calculate integral
            integral = np.trapz(self.processed_data, self.processed_time)
            
            # Convert to appropriate units (µC/cm²)
            membrane_capacitance = 1e-6  # 1 µF/cm²
            integral_adjusted = integral * membrane_capacitance
            
            app_logger.info(f"Signal integral calculated: {integral_adjusted:.6f} µC/cm²")
            return integral_adjusted
            
        except Exception as e:
            app_logger.error(f"Error in calculate_integral: {str(e)}")
            raise

    def process_signal(self):
        """Process the entire signal through all steps."""
        try:
            self.baseline_correction()
            self.normalize_signal()
            processed_data, processed_time = self.average_cycles()
            integral = self.calculate_integral()
            
            app_logger.info("Signal processing completed successfully")
            return processed_data, processed_time, integral
            
        except Exception as e:
            app_logger.error(f"Error in process_signal: {str(e)}")
            raise