"""
Derivative analysis module for calculating and comparing curve steepness.
Used for analyzing differences between control and mutant DHPR isoforms.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import stats
from typing import Dict, Tuple, Optional, List
from src.utils.logger import app_logger


class DerivativeAnalyzer:
    """
    Analyzes derivatives of action potential curves to compare steepness
    between control and experimental groups.
    """
    
    def __init__(self):
        """Initialize the derivative analyzer."""
        self.derivatives = {}
        self.max_derivatives = {}
        self.rise_times = {}
        self.analysis_results = {}
        
        # Parameters for analysis
        self.smoothing_window = 5  # For Savitzky-Golay filter
        self.smoothing_order = 3
        self.rise_time_bounds = (0.1, 0.9)  # 10-90% rise time
        
        app_logger.info("DerivativeAnalyzer initialized")
    
    def analyze_curve(self, time: np.ndarray, data: np.ndarray, 
                     curve_name: str, group: str = 'control') -> Dict:
        """
        Perform complete derivative analysis on a curve.
        
        Args:
            time: Time array (in ms)
            data: Current array (in pA)
            curve_name: Identifier for the curve
            group: 'control' or 'mutant'
            
        Returns:
            Dictionary containing all derivative analysis results
        """
        app_logger.info(f"Analyzing derivatives for {curve_name} ({group})")
        
        # Ensure arrays are numpy arrays
        time = np.asarray(time)
        data = np.asarray(data)
        
        # Calculate derivatives
        first_deriv, first_deriv_time = self.calculate_first_derivative(time, data)
        second_deriv, second_deriv_time = self.calculate_second_derivative(
            time, data, first_deriv, first_deriv_time
        )
        
        # Find critical points
        max_slope_data = self.find_max_slope(time, data, first_deriv, first_deriv_time)
        
        # Calculate rise time
        rise_time_data = self.calculate_rise_time(time, data)
        
        # Calculate activation kinetics
        activation_data = self.analyze_activation_phase(
            time, data, first_deriv, first_deriv_time
        )
        
        # Store results
        results = {
            'curve_name': curve_name,
            'group': group,
            'time': time,
            'data': data,
            'first_derivative': first_deriv,
            'first_derivative_time': first_deriv_time,
            'second_derivative': second_deriv,
            'second_derivative_time': second_deriv_time,
            'max_slope': max_slope_data,
            'rise_time': rise_time_data,
            'activation': activation_data,
            'smoothing_params': {
                'window': self.smoothing_window,
                'order': self.smoothing_order
            }
        }
        
        # Store in internal dictionary
        self.analysis_results[curve_name] = results
        
        app_logger.info(f"Derivative analysis complete for {curve_name}")
        return results
    
    def calculate_first_derivative(self, time: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the first derivative (dI/dt) with smoothing.
        
        Args:
            time: Time array (ms)
            data: Current array (pA)
            
        Returns:
            Tuple of (derivative values, corresponding time points)
        """
        # Apply smoothing first to reduce noise
        if len(data) > self.smoothing_window:
            data_smooth = savgol_filter(data, self.smoothing_window, self.smoothing_order)
        else:
            data_smooth = data
            app_logger.warning("Data too short for smoothing, using raw data")
        
        # Calculate derivative using central differences
        dt = np.diff(time)
        dy = np.diff(data_smooth)
        
        # Avoid division by zero
        dt = np.where(dt == 0, 1e-10, dt)
        
        first_derivative = dy / dt  # pA/ms
        
        # Time points for derivative (midpoints)
        derivative_time = time[:-1] + dt / 2
        
        app_logger.debug(f"First derivative calculated: {len(first_derivative)} points")
        return first_derivative, derivative_time
    
    def calculate_second_derivative(self, time: np.ndarray, data: np.ndarray,
                                  first_deriv: Optional[np.ndarray] = None,
                                  first_deriv_time: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the second derivative (d²I/dt²).
        
        Args:
            time: Original time array
            data: Original current array
            first_deriv: Pre-calculated first derivative (optional)
            first_deriv_time: Time points for first derivative (optional)
            
        Returns:
            Tuple of (second derivative values, corresponding time points)
        """
        if first_deriv is None or first_deriv_time is None:
            first_deriv, first_deriv_time = self.calculate_first_derivative(time, data)
        
        # Apply smoothing to first derivative before differentiating again
        if len(first_deriv) > self.smoothing_window:
            first_deriv_smooth = savgol_filter(first_deriv, self.smoothing_window, self.smoothing_order)
        else:
            first_deriv_smooth = first_deriv
        
        # Calculate second derivative
        dt = np.diff(first_deriv_time)
        dy = np.diff(first_deriv_smooth)
        
        dt = np.where(dt == 0, 1e-10, dt)
        second_derivative = dy / dt  # pA/ms²
        
        # Time points for second derivative
        second_deriv_time = first_deriv_time[:-1] + dt / 2
        
        app_logger.debug(f"Second derivative calculated: {len(second_derivative)} points")
        return second_derivative, second_deriv_time
    
    def find_max_slope(self, time: np.ndarray, data: np.ndarray,
                      first_deriv: Optional[np.ndarray] = None,
                      first_deriv_time: Optional[np.ndarray] = None) -> Dict:
        """
        Find the maximum slope (steepest point) of the curve.
        
        Returns:
            Dictionary with max slope information
        """
        if first_deriv is None or first_deriv_time is None:
            first_deriv, first_deriv_time = self.calculate_first_derivative(time, data)
        
        # Find both positive and negative max slopes
        max_positive_idx = np.argmax(first_deriv)
        max_negative_idx = np.argmin(first_deriv)
        
        # Determine which is larger in magnitude
        if abs(first_deriv[max_negative_idx]) > abs(first_deriv[max_positive_idx]):
            max_idx = max_negative_idx
            is_negative = True
        else:
            max_idx = max_positive_idx
            is_negative = False
        
        max_slope = first_deriv[max_idx]
        max_slope_time = first_deriv_time[max_idx]
        
        # Interpolate to find the exact current value at max slope time
        f_interp = interp1d(time, data, kind='linear', fill_value='extrapolate')
        max_slope_current = f_interp(max_slope_time)
        
        result = {
            'max_slope': max_slope,
            'max_slope_time': max_slope_time,
            'max_slope_current': max_slope_current,
            'is_negative': is_negative,
            'max_positive_slope': first_deriv[max_positive_idx],
            'max_negative_slope': first_deriv[max_negative_idx]
        }
        
        app_logger.info(f"Max slope: {max_slope:.2f} pA/ms at {max_slope_time:.2f} ms")
        return result
    
    def calculate_rise_time(self, time: np.ndarray, data: np.ndarray) -> Dict:
        """
        Calculate rise time between specified percentages of peak.
        
        Returns:
            Dictionary with rise time information
        """
        # Find baseline and peak
        baseline = np.median(data[:int(len(data)*0.1)])  # First 10% for baseline
        
        # Determine if this is an upward or downward deflection
        peak_idx = np.argmax(np.abs(data - baseline))
        peak_value = data[peak_idx]
        peak_time = time[peak_idx]
        
        # Calculate threshold levels
        amplitude = peak_value - baseline
        lower_threshold = baseline + amplitude * self.rise_time_bounds[0]  # 10%
        upper_threshold = baseline + amplitude * self.rise_time_bounds[1]  # 90%
        
        # Find crossing times
        if amplitude > 0:  # Positive deflection
            lower_cross = np.where(data >= lower_threshold)[0]
            upper_cross = np.where(data >= upper_threshold)[0]
        else:  # Negative deflection
            lower_cross = np.where(data <= lower_threshold)[0]
            upper_cross = np.where(data <= upper_threshold)[0]
        
        if len(lower_cross) > 0 and len(upper_cross) > 0:
            # Find first crossing of lower threshold and first crossing of upper
            t_10 = time[lower_cross[0]]
            t_90 = time[upper_cross[0]]
            rise_time = t_90 - t_10
        else:
            rise_time = np.nan
            t_10 = np.nan
            t_90 = np.nan
        
        result = {
            'rise_time': rise_time,
            't_10': t_10,
            't_90': t_90,
            'baseline': baseline,
            'peak_value': peak_value,
            'peak_time': peak_time,
            'amplitude': amplitude
        }
        
        app_logger.info(f"Rise time (10-90%): {rise_time:.2f} ms")
        return result
    
    def analyze_activation_phase(self, time: np.ndarray, data: np.ndarray,
                               first_deriv: Optional[np.ndarray] = None,
                               first_deriv_time: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze the activation phase of the action potential.
        
        Returns:
            Dictionary with activation phase parameters
        """
        if first_deriv is None or first_deriv_time is None:
            first_deriv, first_deriv_time = self.calculate_first_derivative(time, data)
        
        # Find activation start (threshold crossing)
        baseline = np.median(data[:int(len(data)*0.1)])
        noise_level = np.std(data[:int(len(data)*0.1)])
        threshold = baseline + 3 * noise_level  # 3 sigma threshold
        
        # Find where signal first exceeds threshold
        threshold_crossings = np.where(np.abs(data - baseline) > np.abs(threshold - baseline))[0]
        
        if len(threshold_crossings) > 0:
            activation_start_idx = threshold_crossings[0]
            activation_start_time = time[activation_start_idx]
            
            # Find peak of activation phase
            peak_idx = np.argmax(np.abs(data - baseline))
            peak_time = time[peak_idx]
            
            # Calculate activation slope in the linear region
            # Find region between 20% and 80% of peak
            amplitude = data[peak_idx] - baseline
            idx_20 = np.argmin(np.abs(data - (baseline + 0.2 * amplitude)))
            idx_80 = np.argmin(np.abs(data - (baseline + 0.8 * amplitude)))
            
            if idx_80 > idx_20:
                # Linear fit in this region
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time[idx_20:idx_80], data[idx_20:idx_80]
                )
                activation_slope = slope
            else:
                activation_slope = np.nan
        else:
            activation_start_time = np.nan
            activation_slope = np.nan
            peak_time = np.nan
        
        result = {
            'activation_start_time': activation_start_time,
            'activation_slope': activation_slope,  # Linear approximation
            'time_to_peak': peak_time - activation_start_time if not np.isnan(activation_start_time) else np.nan
        }
        
        return result
    
    def compare_groups(self, control_curves: List[str], mutant_curves: List[str]) -> Dict:
        """
        Statistical comparison between control and mutant groups.
        
        Args:
            control_curves: List of curve names in control group
            mutant_curves: List of curve names in mutant group
            
        Returns:
            Dictionary with statistical comparison results
        """
        app_logger.info(f"Comparing {len(control_curves)} control vs {len(mutant_curves)} mutant curves")
        
        # Extract max slopes for each group
        control_slopes = []
        mutant_slopes = []
        
        for curve_name in control_curves:
            if curve_name in self.analysis_results:
                control_slopes.append(self.analysis_results[curve_name]['max_slope']['max_slope'])
        
        for curve_name in mutant_curves:
            if curve_name in self.analysis_results:
                mutant_slopes.append(self.analysis_results[curve_name]['max_slope']['max_slope'])
        
        # Convert to numpy arrays
        control_slopes = np.array(control_slopes)
        mutant_slopes = np.array(mutant_slopes)
        
        # Statistical tests
        if len(control_slopes) > 0 and len(mutant_slopes) > 0:
            # T-test
            t_stat, t_pvalue = stats.ttest_ind(control_slopes, mutant_slopes)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(control_slopes, mutant_slopes)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_slopes) - 1) * np.var(control_slopes) + 
                                 (len(mutant_slopes) - 1) * np.var(mutant_slopes)) / 
                                (len(control_slopes) + len(mutant_slopes) - 2))
            cohens_d = (np.mean(mutant_slopes) - np.mean(control_slopes)) / pooled_std
            
            results = {
                'control_mean': np.mean(control_slopes),
                'control_std': np.std(control_slopes),
                'control_n': len(control_slopes),
                'mutant_mean': np.mean(mutant_slopes),
                'mutant_std': np.std(mutant_slopes),
                'mutant_n': len(mutant_slopes),
                'percent_change': ((np.mean(mutant_slopes) - np.mean(control_slopes)) / 
                                 np.abs(np.mean(control_slopes))) * 100,
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'u_statistic': u_stat,
                'u_pvalue': u_pvalue,
                'cohens_d': cohens_d,
                'significant': t_pvalue < 0.05
            }
            
            app_logger.info(f"Group comparison: Control mean={results['control_mean']:.2f}, "
                          f"Mutant mean={results['mutant_mean']:.2f}, "
                          f"p-value={results['t_pvalue']:.4f}")
        else:
            results = {
                'error': 'Insufficient data for comparison'
            }
            app_logger.warning("Insufficient data for group comparison")
        
        return results
    
    def get_derivative_at_time(self, curve_name: str, time_point: float) -> Dict:
        """
        Get derivative value at a specific time point.
        
        Args:
            curve_name: Name of the analyzed curve
            time_point: Time point in ms
            
        Returns:
            Dictionary with derivative values at that time
        """
        if curve_name not in self.analysis_results:
            return {'error': 'Curve not analyzed'}
        
        results = self.analysis_results[curve_name]
        
        # Interpolate to get exact values
        f_data = interp1d(results['time'], results['data'], kind='linear', fill_value='extrapolate')
        f_deriv1 = interp1d(results['first_derivative_time'], results['first_derivative'], 
                           kind='linear', fill_value='extrapolate')
        f_deriv2 = interp1d(results['second_derivative_time'], results['second_derivative'], 
                           kind='linear', fill_value='extrapolate')
        
        return {
            'time': time_point,
            'current': float(f_data(time_point)),
            'first_derivative': float(f_deriv1(time_point)),
            'second_derivative': float(f_deriv2(time_point))
        }
    
    def export_results(self, curve_name: str) -> Dict:
        """
        Export analysis results for a specific curve.
        
        Args:
            curve_name: Name of the curve to export
            
        Returns:
            Dictionary with all analysis results
        """
        if curve_name not in self.analysis_results:
            return {'error': 'Curve not analyzed'}
        
        return self.analysis_results[curve_name]
    
    def clear_results(self):
        """Clear all stored analysis results."""
        self.analysis_results.clear()
        app_logger.info("All derivative analysis results cleared")


# Utility function for batch analysis
def analyze_curves_batch(curves_data: List[Dict], group_assignment: Optional[Dict] = None) -> DerivativeAnalyzer:
    """
    Analyze multiple curves in batch.
    
    Args:
        curves_data: List of dictionaries with 'name', 'time', and 'data' keys
        group_assignment: Optional dictionary mapping curve names to groups
        
    Returns:
        DerivativeAnalyzer instance with all results
    """
    analyzer = DerivativeAnalyzer()
    
    for curve in curves_data:
        name = curve['name']
        group = 'control'  # Default
        
        # Check for group assignment
        if group_assignment and name in group_assignment:
            group = group_assignment[name]
        elif 'cav1.1_delta_e_29' in name.lower():
            group = 'mutant'
        
        analyzer.analyze_curve(
            curve['time'],
            curve['data'],
            name,
            group
        )
    
    return analyzer