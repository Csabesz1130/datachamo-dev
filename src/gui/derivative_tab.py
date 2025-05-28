"""
Enhanced UI tab specifically for derivative analysis visualization and control.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.analysis.derivative_analyzer import DerivativeAnalyzer
from src.analysis.group_comparison import GroupComparisonAnalyzer
from src.utils.logger import app_logger


class DerivativeAnalysisTab:
    """
    Enhanced tab for derivative analysis with dedicated controls and visualizations.
    """
    
    def __init__(self, parent, update_callback=None):
        """
        Initialize the derivative analysis tab.
        
        Args:
            parent: Parent widget (notebook)
            update_callback: Callback for updates
        """
        self.parent = parent
        self.update_callback = update_callback
        
        # Create main frame
        self.frame = ttk.Frame(parent)
        
        # Initialize analyzers
        self.derivative_analyzer = DerivativeAnalyzer()
        self.group_comparison = GroupComparisonAnalyzer()
        
        # Data storage
        self.current_results = None
        self.comparison_results = None
        
        # Create UI components
        self.setup_ui()
        
        app_logger.info("Derivative Analysis Tab initialized")
    
    def setup_ui(self):
        """Set up the user interface components."""
        # Create main paned window
        self.paned_window = ttk.PanedWindow(self.frame, orient='horizontal')
        self.paned_window.pack(fill='both', expand=True)
        
        # Left panel - Controls
        self.control_frame = ttk.Frame(self.paned_window)
        self.setup_control_panel()
        
        # Right panel - Visualization
        self.viz_frame = ttk.Frame(self.paned_window)
        self.setup_visualization_panel()
        
        # Add frames to paned window
        self.paned_window.add(self.control_frame, weight=1)
        self.paned_window.add(self.viz_frame, weight=3)
    
    def setup_control_panel(self):
        """Set up the control panel with analysis options."""
        # Analysis Settings
        settings_frame = ttk.LabelFrame(self.control_frame, text="Analysis Settings")
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Derivative order selection
        order_frame = ttk.Frame(settings_frame)
        order_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(order_frame, text="Derivative Order:").pack(side='left')
        self.derivative_order = tk.IntVar(value=1)
        ttk.Radiobutton(order_frame, text="1st", variable=self.derivative_order, 
                       value=1, command=self.update_analysis).pack(side='left')
        ttk.Radiobutton(order_frame, text="2nd", variable=self.derivative_order, 
                       value=2, command=self.update_analysis).pack(side='left')
        
        # Smoothing parameters
        smooth_frame = ttk.Frame(settings_frame)
        smooth_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(smooth_frame, text="Smoothing Window:").pack(side='left')
        self.smooth_window = tk.IntVar(value=5)
        ttk.Spinbox(smooth_frame, from_=3, to=21, increment=2, 
                   textvariable=self.smooth_window, width=5,
                   command=self.update_analysis).pack(side='left', padx=5)
        
        ttk.Label(smooth_frame, text="Order:").pack(side='left')
        self.smooth_order = tk.IntVar(value=3)
        ttk.Spinbox(smooth_frame, from_=1, to=5, 
                   textvariable=self.smooth_order, width=5,
                   command=self.update_analysis).pack(side='left', padx=5)
        
        # Phase selection
        phase_frame = ttk.LabelFrame(settings_frame, text="Phase Selection")
        phase_frame.pack(fill='x', padx=5, pady=5)
        
        self.analyze_depolarization = tk.BooleanVar(value=True)
        self.analyze_repolarization = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(phase_frame, text="Depolarization", 
                       variable=self.analyze_depolarization,
                       command=self.update_analysis).pack(anchor='w')
        ttk.Checkbutton(phase_frame, text="Repolarization", 
                       variable=self.analyze_repolarization,
                       command=self.update_analysis).pack(anchor='w')
        
        # Group assignment
        group_frame = ttk.LabelFrame(self.control_frame, text="Group Assignment")
        group_frame.pack(fill='x', padx=5, pady=5)
        
        self.group_var = tk.StringVar(value="control")
        ttk.Radiobutton(group_frame, text="Control", 
                       variable=self.group_var, value="control").pack(anchor='w')
        ttk.Radiobutton(group_frame, text="Mutant", 
                       variable=self.group_var, value="mutant").pack(anchor='w')
        
        # Auto-detect checkbox
        self.auto_detect_group = tk.BooleanVar(value=True)
        ttk.Checkbutton(group_frame, text="Auto-detect from filename",
                       variable=self.auto_detect_group).pack(anchor='w')
        
        # Statistical tests
        stats_frame = ttk.LabelFrame(self.control_frame, text="Statistical Tests")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        # Test selection
        test_frame = ttk.Frame(stats_frame)
        test_frame.pack(fill='x', padx=5, pady=5)
        
        self.test_type = tk.StringVar(value="auto")
        ttk.Label(test_frame, text="Test Type:").pack(side='left')
        test_combo = ttk.Combobox(test_frame, textvariable=self.test_type,
                                values=["auto", "t-test", "mann-whitney"],
                                state="readonly", width=15)
        test_combo.pack(side='left', padx=5)
        
        # Significance level
        alpha_frame = ttk.Frame(stats_frame)
        alpha_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(alpha_frame, text="Alpha:").pack(side='left')
        self.alpha_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(alpha_frame, from_=0.01, to=0.10, increment=0.01,
                   textvariable=self.alpha_var, width=6,
                   format="%.2f").pack(side='left', padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(button_frame, text="Analyze Current",
                  command=self.analyze_current_data).pack(fill='x', pady=2)
        ttk.Button(button_frame, text="Load Group Data",
                  command=self.load_group_data).pack(fill='x', pady=2)
        ttk.Button(button_frame, text="Compare Groups",
                  command=self.compare_groups).pack(fill='x', pady=2)
        ttk.Button(button_frame, text="Export Results",
                  command=self.export_results).pack(fill='x', pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(self.control_frame, text="Results Summary")
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap='word', height=10)
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_visualization_panel(self):
        """Set up the visualization panel with matplotlib plots."""
        # Create notebook for multiple plots
        self.plot_notebook = ttk.Notebook(self.viz_frame)
        self.plot_notebook.pack(fill='both', expand=True)
        
        # Derivative plot
        self.deriv_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.deriv_frame, text="Derivatives")
        
        self.deriv_fig = Figure(figsize=(8, 6))
        self.deriv_canvas = FigureCanvasTkAgg(self.deriv_fig, master=self.deriv_frame)
        self.deriv_canvas.draw()
        self.deriv_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Comparison plot
        self.comp_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.comp_frame, text="Group Comparison")
        
        self.comp_fig = Figure(figsize=(8, 6))
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=self.comp_frame)
        self.comp_canvas.draw()
        self.comp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Statistics plot
        self.stats_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.stats_frame, text="Statistics")
        
        self.stats_fig = Figure(figsize=(8, 6))
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=self.stats_frame)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def analyze_current_data(self):
        """Analyze the currently loaded data."""
        if self.update_callback:
            # Request current data from main app
            data = self.update_callback('get_current_data')
            if data:
                self.analyze_data(data)
            else:
                messagebox.showwarning("No Data", "No data currently loaded")
    
    def analyze_data(self, data):
        """
        Analyze the given data with current settings.
        
        Args:
            data: Dictionary containing time and current arrays
        """
        try:
            # Update analyzer settings
            self.derivative_analyzer.smoothing_window = self.smooth_window.get()
            self.derivative_analyzer.smoothing_order = self.smooth_order.get()
            
            # Determine group
            group = self.group_var.get()
            if self.auto_detect_group.get() and 'filename' in data:
                group = self.group_comparison._detect_group_type(data['filename'])
            
            # Run analysis
            results = self.derivative_analyzer.analyze_curve(
                data['time'], 
                data['current'],
                data.get('filename', 'current'),
                group
            )
            
            self.current_results = results
            
            # Update displays
            self.update_results_display(results)
            self.update_derivative_plot(results)
            
            # Add to group comparison if needed
            if hasattr(self, 'group_comparison'):
                self.group_comparison.add_to_group(
                    data.get('filename', 'current'),
                    {'time': data['time'], 'current': data['current'], 
                     'derivative_results': results},
                    group
                )
            
        except Exception as e:
            app_logger.error(f"Error analyzing data: {str(e)}")
            messagebox.showerror("Analysis Error", str(e))
    
    def update_results_display(self, results):
        """Update the results text display."""
        self.results_text.delete(1.0, tk.END)
        
        text_parts = ["Derivative Analysis Results\n" + "="*30 + "\n"]
        
        # Max slope
        if 'max_slope' in results:
            ms = results['max_slope']
            text_parts.append(f"\nMaximum Slope:")
            text_parts.append(f"  Value: {ms['max_slope']:.2f} pA/ms")
            text_parts.append(f"  Time: {ms['max_slope_time']:.2f} ms")
            text_parts.append(f"  Current: {ms['max_slope_current']:.2f} pA")
        
        # Rise time
        if 'rise_time' in results:
            rt = results['rise_time']
            if not np.isnan(rt['rise_time']):
                text_parts.append(f"\nRise Time (10-90%):")
                text_parts.append(f"  Duration: {rt['rise_time']:.2f} ms")
                text_parts.append(f"  Start: {rt['t_10']:.2f} ms")
                text_parts.append(f"  End: {rt['t_90']:.2f} ms")
        
        # Activation parameters
        if 'activation' in results:
            act = results['activation']
            text_parts.append(f"\nActivation Phase:")
            if not np.isnan(act['activation_slope']):
                text_parts.append(f"  Slope: {act['activation_slope']:.2f} pA/ms")
            if not np.isnan(act['time_to_peak']):
                text_parts.append(f"  Time to Peak: {act['time_to_peak']:.2f} ms")
        
        self.results_text.insert(1.0, '\n'.join(text_parts))
    
    def update_derivative_plot(self, results):
        """Update the derivative visualization plot."""
        self.deriv_fig.clear()
        
        # Create subplots
        gs = self.deriv_fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        ax1 = self.deriv_fig.add_subplot(gs[0])
        ax2 = self.deriv_fig.add_subplot(gs[1], sharex=ax1)
        ax3 = self.deriv_fig.add_subplot(gs[2], sharex=ax1)
        
        # Plot original signal
        ax1.plot(results['time'], results['data'], 'b-', label='Signal')
        ax1.set_ylabel('Current (pA)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot first derivative
        ax2.plot(results['first_derivative_time'], results['first_derivative'],
                'r-', label='1st Derivative')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('dI/dt (pA/ms)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Mark max slope
        if 'max_slope' in results:
            ms = results['max_slope']
            ax2.plot(ms['max_slope_time'], ms['max_slope'], 'ro', 
                    markersize=8, label=f'Max: {ms["max_slope"]:.1f}')
        
        # Plot second derivative
        ax3.plot(results['second_derivative_time'], results['second_derivative'],
                'g-', label='2nd Derivative')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_ylabel('d²I/dt² (pA/ms²)')
        ax3.set_xlabel('Time (ms)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Hide x labels on upper plots
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        self.deriv_fig.tight_layout()
        self.deriv_canvas.draw()
    
    def load_group_data(self):
        """Load multiple files for group comparison."""
        files = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=[("Data files", "*.csv *.txt *.atf"), ("All files", "*.*")]
        )
        
        if files:
            # Here you would implement file loading logic
            # For now, show a message
            messagebox.showinfo("Load Files", 
                              f"Selected {len(files)} files for analysis")
    
    def compare_groups(self):
        """Run group comparison analysis."""
        try:
            # Run comparison
            self.comparison_results = self.group_comparison.compare_derivatives()
            
            # Update displays
            self.update_comparison_plot()
            self.update_statistics_plot()
            
            # Show summary
            summary = self.group_comparison._generate_summary()
            messagebox.showinfo("Comparison Complete", summary)
            
        except Exception as e:
            app_logger.error(f"Error in group comparison: {str(e)}")
            messagebox.showerror("Comparison Error", str(e))
    
    def update_comparison_plot(self):
        """Update the group comparison visualization."""
        if not self.comparison_results:
            return
        
        self.comp_fig.clear()
        
        # Create box plots for each parameter
        parameters = []
        control_data = []
        mutant_data = []
        
        for param, results in self.comparison_results.items():
            if param != 'sample_sizes' and 'control' in results:
                parameters.append(param.replace('_', ' ').title())
                control_data.append(results['control']['mean'])
                mutant_data.append(results['mutant']['mean'])
        
        if parameters:
            ax = self.comp_fig.add_subplot(111)
            
            x = np.arange(len(parameters))
            width = 0.35
            
            # Create bars
            rects1 = ax.bar(x - width/2, control_data, width, label='Control',
                           color='blue', alpha=0.7)
            rects2 = ax.bar(x + width/2, mutant_data, width, label='Mutant',
                           color='red', alpha=0.7)
            
            # Add significance markers
            for i, (param, results) in enumerate(self.comparison_results.items()):
                if param != 'sample_sizes' and 'statistics' in results:
                    if results['statistics']['significant']:
                        ax.text(i, max(control_data[i], mutant_data[i]) * 1.05,
                               '*', ha='center', va='bottom', fontsize=16)
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Value')
            ax.set_title('Group Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(parameters, rotation=45, ha='right')
            ax.legend()
            
            self.comp_fig.tight_layout()
            self.comp_canvas.draw()
    
    def update_statistics_plot(self):
        """Update the statistics visualization."""
        if not self.comparison_results:
            return
        
        self.stats_fig.clear()
        
        # Create effect size plot
        ax = self.stats_fig.add_subplot(111)
        
        parameters = []
        effect_sizes = []
        p_values = []
        
        for param, results in self.comparison_results.items():
            if param != 'sample_sizes' and 'statistics' in results:
                parameters.append(param.replace('_', ' ').title())
                effect_sizes.append(results['statistics']['cohens_d'])
                p_values.append(results['statistics']['p_value'])
        
        if parameters:
            # Plot effect sizes
            y_pos = np.arange(len(parameters))
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            
            bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.7)
            
            # Add reference lines
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5)
            
            # Labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(parameters)
            ax.set_xlabel("Cohen's d (Effect Size)")
            ax.set_title("Effect Sizes with Significance")
            
            # Add text annotations
            for i, (d, p) in enumerate(zip(effect_sizes, p_values)):
                ax.text(d + 0.05, i, f'p={p:.3f}', 
                       va='center', fontsize=8)
            
            self.stats_fig.tight_layout()
            self.stats_canvas.draw()
    
    def export_results(self):
        """Export analysis results to file."""
        if not self.current_results and not self.comparison_results:
            messagebox.showwarning("No Results", "No results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), 
                      ("CSV files", "*.csv"),
                      ("JSON files", "*.json")]
        )
        
        if filepath:
            try:
                ext = filepath.split('.')[-1].lower()
                
                if self.comparison_results:
                    self.group_comparison.export_results(filepath, ext)
                else:
                    # Export single analysis results
                    if ext == 'json':
                        import json
                        with open(filepath, 'w') as f:
                            json.dump(self.current_results, f, indent=2, default=str)
                    elif ext == 'csv':
                        # Create summary CSV
                        import pandas as pd
                        summary = {
                            'Parameter': [],
                            'Value': []
                        }
                        
                        if 'max_slope' in self.current_results:
                            summary['Parameter'].append('Max Slope')
                            summary['Value'].append(self.current_results['max_slope']['max_slope'])
                        
                        if 'rise_time' in self.current_results:
                            summary['Parameter'].append('Rise Time')
                            summary['Value'].append(self.current_results['rise_time']['rise_time'])
                        
                        pd.DataFrame(summary).to_csv(filepath, index=False)
                
                messagebox.showinfo("Export Complete", f"Results exported to {filepath}")
                
            except Exception as e:
                app_logger.error(f"Error exporting results: {str(e)}")
                messagebox.showerror("Export Error", str(e))
    
    def update_analysis(self, *args):
        """Update analysis with new settings."""
        if self.current_results:
            # Re-analyze with new settings
            data = {
                'time': self.current_results['time'],
                'current': self.current_results['data'],
                'filename': self.current_results.get('curve_name', 'current')
            }
            self.analyze_data(data)