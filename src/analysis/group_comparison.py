"""
Group comparison framework for analyzing control vs mutant groups.
Provides statistical analysis and comparison tools for curve data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from src.utils.logger import app_logger
from src.analysis.derivative_analyzer import DerivativeAnalyzer


class GroupComparisonAnalyzer:
    """
    Analyzes and compares data between control and mutant groups.
    Supports automatic group detection and statistical comparisons.
    """
    
    def __init__(self):
        """Initialize the group comparison analyzer."""
        self.control_group = []
        self.mutant_group = []
        self.comparison_results = {}
        self.derivative_analyzer = DerivativeAnalyzer()
        self.group_statistics = {}
        
        # Configuration for group detection
        self.mutant_patterns = [
            'cav1.1_delta_e_29',
            'mutant',
            'δe29',
            'de29'
        ]
        
        # Statistical test parameters
        self.alpha = 0.05  # Significance level
        self.normality_test = 'shapiro'  # or 'normaltest'
        
        app_logger.info("GroupComparisonAnalyzer initialized")
    
    def add_to_group(self, file_path: str, data: Dict[str, Any], 
                     group_type: Optional[str] = None) -> str:
        """
        Add a file to control or mutant group based on naming convention.
        
        Args:
            file_path: Path to the file
            data: Dictionary containing curve data and metadata
            group_type: Optional explicit group assignment ('control' or 'mutant')
            
        Returns:
            The group the file was assigned to
        """
        # Auto-detect group if not specified
        if group_type is None:
            group_type = self._detect_group_type(file_path)
        
        # Validate group type
        if group_type not in ['control', 'mutant']:
            app_logger.warning(f"Invalid group type: {group_type}. Defaulting to control.")
            group_type = 'control'
        
        # Create entry
        entry = {
            'file_path': file_path,
            'data': data,
            'group': group_type,
            'timestamp': pd.Timestamp.now()
        }
        
        # Add to appropriate group
        if group_type == 'control':
            self.control_group.append(entry)
        else:
            self.mutant_group.append(entry)
        
        app_logger.info(f"Added {file_path} to {group_type} group")
        return group_type
    
    def _detect_group_type(self, file_path: str) -> str:
        """
        Auto-detect group type based on filename patterns.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            'control' or 'mutant'
        """
        file_path_lower = file_path.lower()
        
        # Check for mutant patterns
        for pattern in self.mutant_patterns:
            if pattern in file_path_lower:
                app_logger.debug(f"Detected mutant pattern '{pattern}' in {file_path}")
                return 'mutant'
        
        # Default to control
        return 'control'
    
    def compare_derivatives(self) -> Dict:
        """
        Statistical comparison of derivatives between groups.
        
        Returns:
            Dictionary containing comparison results
        """
        app_logger.info("Starting derivative comparison between groups")
        
        # Extract derivative data for each group
        control_derivatives = self._extract_group_derivatives(self.control_group)
        mutant_derivatives = self._extract_group_derivatives(self.mutant_group)
        
        if not control_derivatives or not mutant_derivatives:
            app_logger.warning("Insufficient data for derivative comparison")
            return {'error': 'Insufficient data for comparison'}
        
        # Perform statistical comparisons
        results = {
            'max_slope': self._compare_parameter(
                control_derivatives['max_slopes'],
                mutant_derivatives['max_slopes'],
                'Max Slope'
            ),
            'rise_time': self._compare_parameter(
                control_derivatives['rise_times'],
                mutant_derivatives['rise_times'],
                'Rise Time'
            ),
            'activation_slope': self._compare_parameter(
                control_derivatives['activation_slopes'],
                mutant_derivatives['activation_slopes'],
                'Activation Slope'
            ),
            'time_to_peak': self._compare_parameter(
                control_derivatives['time_to_peaks'],
                mutant_derivatives['time_to_peaks'],
                'Time to Peak'
            ),
            'sample_sizes': {
                'control': len(self.control_group),
                'mutant': len(self.mutant_group)
            }
        }
        
        self.comparison_results = results
        app_logger.info("Derivative comparison completed")
        return results
    
    def _extract_group_derivatives(self, group: List[Dict]) -> Dict:
        """
        Extract derivative parameters from a group.
        
        Args:
            group: List of group entries
            
        Returns:
            Dictionary of derivative parameters
        """
        max_slopes = []
        rise_times = []
        activation_slopes = []
        time_to_peaks = []
        
        for entry in group:
            data = entry['data']
            
            # Analyze derivatives if not already done
            if 'derivative_results' not in data:
                if 'time' in data and 'current' in data:
                    deriv_results = self.derivative_analyzer.analyze_curve(
                        data['time'], 
                        data['current'],
                        entry['file_path'],
                        entry['group']
                    )
                    data['derivative_results'] = deriv_results
            
            if 'derivative_results' in data:
                results = data['derivative_results']
                
                # Extract parameters
                if 'max_slope' in results:
                    max_slopes.append(results['max_slope']['max_slope'])
                
                if 'rise_time' in results:
                    rt = results['rise_time']['rise_time']
                    if not np.isnan(rt):
                        rise_times.append(rt)
                
                if 'activation' in results:
                    act_slope = results['activation']['activation_slope']
                    if not np.isnan(act_slope):
                        activation_slopes.append(act_slope)
                    
                    ttp = results['activation']['time_to_peak']
                    if not np.isnan(ttp):
                        time_to_peaks.append(ttp)
        
        return {
            'max_slopes': np.array(max_slopes),
            'rise_times': np.array(rise_times),
            'activation_slopes': np.array(activation_slopes),
            'time_to_peaks': np.array(time_to_peaks)
        }
    
    def _compare_parameter(self, control_values: np.ndarray, 
                          mutant_values: np.ndarray, 
                          parameter_name: str) -> Dict:
        """
        Compare a single parameter between groups.
        
        Args:
            control_values: Control group values
            mutant_values: Mutant group values
            parameter_name: Name of the parameter
            
        Returns:
            Dictionary with comparison results
        """
        if len(control_values) == 0 or len(mutant_values) == 0:
            return {
                'error': f'Insufficient data for {parameter_name}',
                'parameter': parameter_name
            }
        
        # Basic statistics
        control_mean = np.mean(control_values)
        control_std = np.std(control_values)
        control_sem = stats.sem(control_values)
        
        mutant_mean = np.mean(mutant_values)
        mutant_std = np.std(mutant_values)
        mutant_sem = stats.sem(mutant_values)
        
        # Test for normality
        control_normal = self._test_normality(control_values)
        mutant_normal = self._test_normality(mutant_values)
        
        # Choose appropriate test based on normality
        if control_normal and mutant_normal:
            # Parametric test (t-test)
            t_stat, p_value = stats.ttest_ind(control_values, mutant_values)
            test_name = 'Independent t-test'
        else:
            # Non-parametric test (Mann-Whitney U)
            u_stat, p_value = stats.mannwhitneyu(
                control_values, mutant_values, alternative='two-sided'
            )
            test_name = 'Mann-Whitney U test'
            t_stat = u_stat  # Store as t_stat for consistency
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * control_std**2 + 
             (len(mutant_values) - 1) * mutant_std**2) / 
            (len(control_values) + len(mutant_values) - 2)
        )
        
        if pooled_std > 0:
            cohens_d = (mutant_mean - control_mean) / pooled_std
        else:
            cohens_d = 0
        
        # Confidence intervals
        control_ci = stats.t.interval(
            1 - self.alpha, 
            len(control_values) - 1, 
            loc=control_mean, 
            scale=control_sem
        )
        
        mutant_ci = stats.t.interval(
            1 - self.alpha, 
            len(mutant_values) - 1, 
            loc=mutant_mean, 
            scale=mutant_sem
        )
        
        # Percent change
        if control_mean != 0:
            percent_change = ((mutant_mean - control_mean) / abs(control_mean)) * 100
        else:
            percent_change = np.nan
        
        return {
            'parameter': parameter_name,
            'control': {
                'mean': control_mean,
                'std': control_std,
                'sem': control_sem,
                'n': len(control_values),
                'ci_lower': control_ci[0],
                'ci_upper': control_ci[1],
                'is_normal': control_normal
            },
            'mutant': {
                'mean': mutant_mean,
                'std': mutant_std,
                'sem': mutant_sem,
                'n': len(mutant_values),
                'ci_lower': mutant_ci[0],
                'ci_upper': mutant_ci[1],
                'is_normal': mutant_normal
            },
            'statistics': {
                'test_name': test_name,
                'statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'percent_change': percent_change,
                'significant': p_value < self.alpha
            }
        }
    
    def _test_normality(self, values: np.ndarray) -> bool:
        """
        Test if values follow a normal distribution.
        
        Args:
            values: Array of values to test
            
        Returns:
            True if normal, False otherwise
        """
        if len(values) < 3:
            return False
        
        if self.normality_test == 'shapiro':
            stat, p_value = stats.shapiro(values)
        else:
            stat, p_value = stats.normaltest(values)
        
        return p_value > self.alpha
    
    def generate_comparison_report(self) -> Dict:
        """
        Generate a detailed comparison report with plots and statistics.
        
        Returns:
            Dictionary containing the complete report
        """
        app_logger.info("Generating comparison report")
        
        # Ensure comparison has been run
        if not self.comparison_results:
            self.compare_derivatives()
        
        # Compile report
        report = {
            'summary': self._generate_summary(),
            'detailed_results': self.comparison_results,
            'group_statistics': self._calculate_group_statistics(),
            'recommendations': self._generate_recommendations(),
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'control_n': len(self.control_group),
                'mutant_n': len(self.mutant_group),
                'alpha': self.alpha
            }
        }
        
        return report
    
    def _generate_summary(self) -> str:
        """Generate a text summary of the comparison results."""
        if not self.comparison_results:
            return "No comparison results available."
        
        summary_parts = ["Group Comparison Summary\n" + "="*50 + "\n"]
        
        for param_name, results in self.comparison_results.items():
            if param_name == 'sample_sizes':
                continue
            
            if 'error' in results:
                summary_parts.append(f"\n{param_name}: {results['error']}")
                continue
            
            stats = results['statistics']
            summary_parts.append(f"\n{param_name}:")
            summary_parts.append(f"  Control: {results['control']['mean']:.3f} ± {results['control']['sem']:.3f}")
            summary_parts.append(f"  Mutant: {results['mutant']['mean']:.3f} ± {results['mutant']['sem']:.3f}")
            summary_parts.append(f"  Change: {stats['percent_change']:.1f}%")
            summary_parts.append(f"  p-value: {stats['p_value']:.4f} {'*' if stats['significant'] else ''}")
            summary_parts.append(f"  Effect size (Cohen's d): {stats['cohens_d']:.3f}")
        
        return '\n'.join(summary_parts)
    
    def _calculate_group_statistics(self) -> Dict:
        """Calculate comprehensive statistics for each group."""
        return {
            'control': self._calculate_single_group_stats(self.control_group),
            'mutant': self._calculate_single_group_stats(self.mutant_group)
        }
    
    def _calculate_single_group_stats(self, group: List[Dict]) -> Dict:
        """Calculate statistics for a single group."""
        if not group:
            return {'n': 0, 'files': []}
        
        # Extract all relevant parameters
        parameters = {
            'max_slopes': [],
            'rise_times': [],
            'peak_currents': [],
            'baseline_currents': []
        }
        
        for entry in group:
            data = entry['data']
            
            # Extract parameters from each file
            if 'derivative_results' in data:
                dr = data['derivative_results']
                if 'max_slope' in dr:
                    parameters['max_slopes'].append(dr['max_slope']['max_slope'])
                if 'rise_time' in dr:
                    parameters['rise_times'].append(dr['rise_time']['rise_time'])
            
            if 'current' in data:
                current = np.array(data['current'])
                parameters['peak_currents'].append(np.max(np.abs(current)))
                parameters['baseline_currents'].append(np.median(current[:100]))
        
        # Calculate statistics for each parameter
        stats = {}
        for param_name, values in parameters.items():
            if values:
                values = np.array(values)
                values = values[~np.isnan(values)]  # Remove NaN values
                if len(values) > 0:
                    stats[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'q1': np.percentile(values, 25),
                        'q3': np.percentile(values, 75),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n': len(values)
                    }
        
        return {
            'n': len(group),
            'files': [entry['file_path'] for entry in group],
            'parameters': stats
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate analysis recommendations based on results."""
        recommendations = []
        
        if not self.comparison_results:
            recommendations.append("Run comparison analysis first.")
            return recommendations
        
        # Check sample sizes
        n_control = len(self.control_group)
        n_mutant = len(self.mutant_group)
        
        if n_control < 5 or n_mutant < 5:
            recommendations.append(
                f"Sample sizes are small (control: {n_control}, mutant: {n_mutant}). "
                "Consider collecting more data for robust statistical analysis."
            )
        
        # Check for significant differences
        significant_params = []
        for param_name, results in self.comparison_results.items():
            if param_name != 'sample_sizes' and 'statistics' in results:
                if results['statistics']['significant']:
                    significant_params.append(param_name)
        
        if significant_params:
            recommendations.append(
                f"Significant differences found in: {', '.join(significant_params)}. "
                "Consider further investigation of these parameters."
            )
        else:
            recommendations.append(
                "No significant differences found. Consider increasing sample size "
                "or reviewing experimental conditions."
            )
        
        # Check effect sizes
        large_effects = []
        for param_name, results in self.comparison_results.items():
            if param_name != 'sample_sizes' and 'statistics' in results:
                d = abs(results['statistics']['cohens_d'])
                if d > 0.8:
                    large_effects.append(f"{param_name} (d={d:.2f})")
        
        if large_effects:
            recommendations.append(
                f"Large effect sizes observed for: {', '.join(large_effects)}. "
                "These parameters show substantial differences between groups."
            )
        
        return recommendations
    
    def export_results(self, output_path: str, format: str = 'json') -> bool:
        """
        Export comparison results to file.
        
        Args:
            output_path: Path for output file
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            Success status
        """
        try:
            report = self.generate_comparison_report()
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format == 'csv':
                # Create a summary DataFrame
                summary_data = []
                for param_name, results in self.comparison_results.items():
                    if param_name != 'sample_sizes' and 'statistics' in results:
                        row = {
                            'Parameter': param_name,
                            'Control_Mean': results['control']['mean'],
                            'Control_SEM': results['control']['sem'],
                            'Mutant_Mean': results['mutant']['mean'],
                            'Mutant_SEM': results['mutant']['sem'],
                            'P_Value': results['statistics']['p_value'],
                            'Cohens_D': results['statistics']['cohens_d'],
                            'Percent_Change': results['statistics']['percent_change'],
                            'Significant': results['statistics']['significant']
                        }
                        summary_data.append(row)
                
                df = pd.DataFrame(summary_data)
                df.to_csv(output_path, index=False)
            
            elif format == 'excel':
                # Create Excel file with multiple sheets
                with pd.ExcelWriter(output_path) as writer:
                    # Summary sheet
                    summary_data = []
                    for param_name, results in self.comparison_results.items():
                        if param_name != 'sample_sizes' and 'statistics' in results:
                            row = {
                                'Parameter': param_name,
                                'Control Mean': results['control']['mean'],
                                'Control SEM': results['control']['sem'],
                                'Mutant Mean': results['mutant']['mean'],
                                'Mutant SEM': results['mutant']['sem'],
                                'P-Value': results['statistics']['p_value'],
                                'Cohen\'s D': results['statistics']['cohens_d'],
                                'Percent Change': results['statistics']['percent_change'],
                                'Significant': results['statistics']['significant']
                            }
                            summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # File lists
                    control_files = pd.DataFrame({
                        'Control Files': [e['file_path'] for e in self.control_group]
                    })
                    control_files.to_excel(writer, sheet_name='Control Files', index=False)
                    
                    mutant_files = pd.DataFrame({
                        'Mutant Files': [e['file_path'] for e in self.mutant_group]
                    })
                    mutant_files.to_excel(writer, sheet_name='Mutant Files', index=False)
            
            app_logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error exporting results: {str(e)}")
            return False
    
    def clear_groups(self):
        """Clear all groups and results."""
        self.control_group.clear()
        self.mutant_group.clear()
        self.comparison_results.clear()
        self.group_statistics.clear()
        app_logger.info("All groups and results cleared")
    
    def get_group_summary(self) -> Dict:
        """
        Get a summary of current groups.
        
        Returns:
            Dictionary with group information
        """
        return {
            'control': {
                'count': len(self.control_group),
                'files': [e['file_path'] for e in self.control_group]
            },
            'mutant': {
                'count': len(self.mutant_group),
                'files': [e['file_path'] for e in self.mutant_group]
            },
            'total': len(self.control_group) + len(self.mutant_group)
        }