import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.text import Annotation
from src.utils.logger import app_logger

class CurvePointTracker:
    def __init__(self, ax, processor=None):
        """
        Initialize the curve point tracker.
        
        Args:
            ax: Matplotlib axes object
            processor: ActionPotentialProcessor instance
        """
        self.ax = ax
        self.processor = processor
        self._show_annotations = False
        self.annotations = []
        
        # Curve type information
        self.curve_types = {
            'orange': {'name': 'Orange', 'color': 'orange', 'points': 1800},
            'blue': {'name': 'Blue', 'color': 'blue', 'points': 800, 'offset': 28},
            'magenta': {'name': 'Magenta', 'color': 'magenta', 'points': 200, 'offset': 28},
            'purple_hyperpol': {'name': 'Purple Hyperpol', 'color': 'purple', 'points': 199, 'offset': 1028},
            'purple_depol': {'name': 'Purple Depol', 'color': 'purple', 'points': 199, 'offset': 828}
        }
        
        # Slice information
        self._hyperpol_slice = (1028, 1227)
        self._depol_slice = (828, 1028)
        
        # Connect to mouse events
        self.cid = self.ax.figure.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        app_logger.debug("Curve point tracker initialized")

    @property
    def show_annotations(self):
        return self._show_annotations

    @show_annotations.setter
    def show_annotations(self, value):
        self._show_annotations = value
        if not value and self.annotations:
            for ann in self.annotations:
                ann.remove()
            self.annotations = []
            self.ax.figure.canvas.draw_idle()
        app_logger.info(f"Show points toggled: {value}")

    def _on_mouse_move(self, event):
        """Handle mouse movement events"""
        if event.inaxes != self.ax or not self.processor:
            return
            
        # Get nearest point
        point_info = self._get_nearest_point(event.xdata, event.ydata)
        
        # Update status bar
        self._update_status_bar(event, point_info)
        
        # Update annotations if we have point info
        if point_info:
            self._update_annotations(point_info)
        else:
            # Clear annotations if no point is found
            for ann in self.annotations:
                ann.remove()
            self.annotations = []
            self.ax.figure.canvas.draw_idle()

    def _get_nearest_point(self, x, y):
        """Find the nearest point to the cursor position"""
        if not self.processor or not hasattr(self.processor, 'processed_data'):
            return None
            
        # Get data ranges for adaptive thresholds
        time_range = self.processor.time_data.max() - self.processor.time_data.min()
        value_range = self.processor.processed_data.max() - self.processor.processed_data.min()
        
        # Calculate adaptive thresholds
        time_threshold = min(0.05 * time_range, 5)  # 5% of range or 5ms
        value_threshold = max(0.2 * value_range, 50)  # 20% of range or 50pA
        
        # Check each curve type in priority order
        for curve_type in ['orange', 'purple_hyperpol', 'purple_depol', 'magenta', 'blue']:
            curve_info = self.curve_types[curve_type]
            data = getattr(self.processor, f'{curve_type}_curve', None)
            
            if data is None:
                continue
                
            # Calculate distances
            time_distances = np.abs(self.processor.time_data - x)
            value_distances = np.abs(data - y)
            
            # Find points within thresholds
            valid_points = np.where((time_distances <= time_threshold) & 
                                  (value_distances <= value_threshold))[0]
            
            if len(valid_points) > 0:
                # Get the closest point
                distances = np.sqrt(time_distances[valid_points]**2 + 
                                 value_distances[valid_points]**2)
                closest_idx = valid_points[np.argmin(distances)]
                
                # Get corresponding orange point
                orange_idx = self._get_corresponding_orange_point(curve_type, closest_idx)
                
                return {
                    'curve_type': curve_type,
                    'point_idx': closest_idx,
                    'orange_idx': orange_idx,
                    'x': self.processor.time_data[closest_idx],
                    'y': data[closest_idx]
                }
                
        return None

    def _get_corresponding_orange_point(self, curve_type, point_idx):
        """Get the corresponding orange curve point index"""
        if curve_type == 'orange':
            return point_idx
        elif curve_type in ['blue', 'magenta']:
            return point_idx + self.curve_types[curve_type]['offset']
        elif curve_type == 'purple_hyperpol':
            return point_idx + self._hyperpol_slice[0]
        elif curve_type == 'purple_depol':
            return point_idx + self._depol_slice[0]
        return None

    def _update_status_bar(self, event, point_info):
        """Update the status bar with point information"""
        if point_info:
            curve_info = self.curve_types[point_info['curve_type']]
            status_text = (f"Time: {point_info['x']:.1f} ms, "
                         f"Current: {point_info['y']:.1f} pA | "
                         f"{curve_info['name']} Point: {point_info['point_idx'] + 1} "
                         f"(Orange: {point_info['orange_idx'] + 1})")
        else:
            status_text = f"Time: {event.xdata:.1f} ms, Current: {event.ydata:.1f} pA"
            
        # Always update through toolbar
        if hasattr(self.ax.figure.canvas, 'toolbar'):
            self.ax.figure.canvas.toolbar.set_message(status_text)

    def _update_annotations(self, point_info):
        """Update visual annotations for the point"""
        # Clear previous annotations
        for ann in self.annotations:
            ann.remove()
        self.annotations = []
        
        # Create new annotation
        curve_info = self.curve_types[point_info['curve_type']]
        text = (f"{curve_info['name']} Point: {point_info['point_idx'] + 1} "
                f"[Orange: {point_info['orange_idx'] + 1}]")
                
        ann = Annotation(text, 
                        xy=(point_info['x'], point_info['y']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                fc='white', 
                                alpha=0.7),
                        arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0'))
                                     
        ann.set_color(curve_info['color'])
        self.ax.add_artist(ann)
        self.annotations.append(ann)
        
        self.ax.figure.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the event handler"""
        if hasattr(self, 'cid'):
            self.ax.figure.canvas.mpl_disconnect(self.cid)
            app_logger.debug("Curve point tracker disconnected") 