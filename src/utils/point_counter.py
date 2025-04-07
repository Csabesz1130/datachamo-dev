import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.text import Annotation
from src.utils.logger import app_logger
import os, time
print(f"point_counter.py last modified: {time.ctime(os.path.getmtime(__file__))}")


class CurvePointTracker:
    def __init__(self, ax, processor=None):
        """
        Initialize the curve point tracker.
        
        Args:
            ax: Matplotlib axes object
            processor: ActionPotentialProcessor instance (optional)
        """
        self.ax = ax
        self.processor = processor
        self._show_annotations = False
        self.annotations = []
        
        # Curve type information (offsets might still be useful for blue/magenta)
        self.curve_types = {
            'orange': {'name': 'Orange', 'color': 'orange'},
            'blue': {'name': 'Blue', 'color': 'blue', 'offset': 28}, # Keep offset for time mapping
            'magenta': {'name': 'Magenta', 'color': 'magenta', 'offset': 28}, # Keep offset for time mapping
            'purple_hyperpol': {'name': 'Purple Hyperpol', 'color': 'purple'},
            'purple_depol': {'name': 'Purple Depol', 'color': 'purple'}
            # Removed points and fixed offsets for purple as they depend on processor slices
        }
        
        # Slices are now expected to be set on the processor object
        # self._hyperpol_slice = (1028, 1227) # Removed
        # self._depol_slice = (828, 1028)   # Removed
        
        # Connect to mouse events
        self.cid = self.ax.figure.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        app_logger.info("Curve point tracker initialized (will wait for processor)")

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
        # Basic checks
        if event.inaxes != self.ax:
            # app_logger.debug("Mouse outside axes.")
            return
        if not self.processor:
            app_logger.warning("Point tracker has no processor reference.")
            return
        if event.xdata is None or event.ydata is None:
            # app_logger.debug("Mouse event has no data coordinates (maybe outside data range).")
            return

        # app_logger.debug(f"Mouse move event: x={event.xdata:.2f}, y={event.ydata:.2f}")
        
        # Get nearest point
        point_info = self._get_nearest_point(event.xdata, event.ydata)
        
        # Update status bar
        self._update_status_bar(event, point_info)
        
        # Update annotations if we have point info
        if point_info:
            # app_logger.debug(f"Updating annotation for point: {point_info}")
            self._update_annotations(point_info)
        else:
            # app_logger.debug("No point info found, clearing annotations.")
            # Clear annotations if no point is found
            if self.annotations:
                for ann in self.annotations:
                    ann.remove()
                self.annotations = []
                self.ax.figure.canvas.draw_idle()

    def _get_nearest_point(self, x, y):
        """Find the nearest point to the cursor position across all curves"""
        # Check processor and time_data validity
        if not self.processor or not hasattr(self.processor, 'time_data') or self.processor.time_data is None:
            app_logger.warning("Processor or essential time_data not available for point finding.")
            return None

        # Check if x or y is None (redundant due to caller check, but safe)
        if x is None or y is None:
            app_logger.warning("Received None for x or y coordinates.")
            return None
            
        # Calculate a dynamic distance threshold
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            view_width = xlim[1] - xlim[0]
            view_height = ylim[1] - ylim[0]
            if view_width <= 0 or view_height <= 0:
                 max_dist_sq = 100.0 # Fallback threshold
                 app_logger.warning(f"Invalid view limits ({view_width}, {view_height}), using fallback threshold.")
            else:
                # Threshold based on a small fraction of the view diagonal
                diagonal = np.sqrt(view_width**2 + view_height**2)
                threshold_fraction = 0.03 # 3% - adjust if needed
                max_dist_sq = (threshold_fraction * diagonal)**2
        except Exception as e:
            app_logger.error(f"Error calculating dynamic threshold: {e}, using fallback.")
            max_dist_sq = 100.0
            
        app_logger.debug(f"Cursor at ({x:.2f}, {y:.2f}). Max dist sq threshold: {max_dist_sq:.4f}")

        min_dist_sq = np.inf
        best_point_info = None
        found_on_curve = None # Track which curve had the minimum distance
        
        # Check each curve type
        for curve_type in ['orange', 'purple_hyperpol', 'purple_depol', 'magenta', 'blue']:
            app_logger.debug(f"Checking curve: {curve_type}")
            # Use processor directly to get data
            curve_data = getattr(self.processor, f'{curve_type}_curve', None)
            time_data = getattr(self.processor, 'time_data', None) # Get potentially full time data
            
            # Ensure data is valid and has the expected structure
            if curve_data is None:
                app_logger.debug(f" -> Skipping: {curve_type}_curve is None.")
                continue
            if time_data is None:
                 app_logger.debug(f" -> Skipping: time_data is None (should not happen here).")
                 continue # Should have been caught earlier
                
            # Ensure curve_data and time_data are numpy arrays and not empty
            try:
                curve_data = np.asarray(curve_data)
                time_data = np.asarray(time_data) # Ensure processor's time_data is array
                if curve_data.size == 0:
                     app_logger.debug(f" -> Skipping: {curve_type}_curve data array is empty.")
                     continue
                if time_data.size == 0:
                     app_logger.debug(f" -> Skipping: time_data array is empty.")
                     continue # Should not happen if check at start passed

                # --- Determine the correct time subset and curve subset for distance calculation --- 
                idx_offset = 0 # Default offset for time_data indexing
                if curve_type == 'purple_hyperpol':
                    # Get slice dynamically from processor
                    slice_ = getattr(self.processor, '_hyperpol_slice', None)
                    if slice_ is None:
                        app_logger.debug(f" -> Skipping {curve_type}: _hyperpol_slice missing on processor.")
                        continue
                    # Basic sanity check
                    if not isinstance(slice_, tuple) or len(slice_) != 2 or slice_[1] > len(time_data) or slice_[0] >= slice_[1]:
                         app_logger.warning(f"Invalid slice for {curve_type} from processor: {slice_} vs time_data len {len(time_data)}")
                         continue
                    time_subset = time_data[slice_[0]:slice_[1]]
                    curve_subset = curve_data # purple curve is already sliced relative to its time points
                    idx_offset = slice_[0]
                    if len(time_subset) != len(curve_subset):
                        app_logger.warning(f" -> Length mismatch for {curve_type} after getting slice {slice_}: time={len(time_subset)}, curve={len(curve_subset)}")
                        continue
                elif curve_type == 'purple_depol':
                     # Get slice dynamically from processor
                    slice_ = getattr(self.processor, '_depol_slice', None)
                    if slice_ is None:
                        app_logger.debug(f" -> Skipping {curve_type}: _depol_slice missing on processor.")
                        continue
                     # Basic sanity check
                    if not isinstance(slice_, tuple) or len(slice_) != 2 or slice_[1] > len(time_data) or slice_[0] >= slice_[1]:
                         app_logger.warning(f"Invalid slice for {curve_type} from processor: {slice_} vs time_data len {len(time_data)}")
                         continue
                    time_subset = time_data[slice_[0]:slice_[1]]
                    curve_subset = curve_data # purple curve is already sliced relative to its time points
                    idx_offset = slice_[0]
                    if len(time_subset) != len(curve_subset):
                        app_logger.warning(f" -> Length mismatch for {curve_type} after getting slice {slice_}: time={len(time_subset)}, curve={len(curve_subset)}")
                        continue
                elif curve_type == 'magenta':
                    # Magenta often corresponds to the first N points derived from segments
                    num_points = len(curve_data)
                    # Use offset defined in curve_types if needed for time mapping
                    idx_offset = self.curve_types[curve_type].get('offset', 0) 
                    if idx_offset + num_points > len(time_data):
                         app_logger.warning(f" -> Invalid time range for {curve_type} based on offset {idx_offset} and length {num_points} vs time data {len(time_data)}")
                         continue
                    time_subset = time_data[idx_offset : idx_offset + num_points]
                    curve_subset = curve_data
                elif curve_type == 'blue':
                    # Blue corresponds to specific segments concatenated
                    num_points = len(curve_data)
                    # Use offset defined in curve_types if needed for time mapping
                    idx_offset = self.curve_types[curve_type].get('offset', 0)
                    if idx_offset + num_points > len(time_data):
                        app_logger.warning(f" -> Invalid time range for {curve_type} based on offset {idx_offset} and length {num_points} vs time data {len(time_data)}")
                        continue
                    time_subset = time_data[idx_offset : idx_offset + num_points]
                    curve_subset = curve_data
                else: # Orange curve presumably uses full available time_data that corresponds to its length
                     if len(time_data) < len(curve_data):
                         app_logger.warning(f" -> Length mismatch for {curve_type}: time={len(time_data)}, curve={len(curve_data)}")
                         continue
                     # Use the portion of time_data that matches the orange curve length
                     time_subset = time_data[:len(curve_data)] 
                     curve_subset = curve_data
                     idx_offset = 0 # Orange starts at the beginning of its relevant time data
                
                # Final check after determining subsets
                if len(time_subset) == 0 or len(curve_subset) == 0:
                    app_logger.debug(f" -> Skipping {curve_type}: Subset resulted in empty array.")
                    continue
                if len(time_subset) != len(curve_subset):
                    app_logger.warning(f" -> Final length mismatch for {curve_type} after slicing: time={len(time_subset)}, curve={len(curve_subset)}")
                    continue
                app_logger.debug(f" -> Using {len(time_subset)} points for {curve_type} (offset: {idx_offset})")
                    
            except Exception as e:
                app_logger.error(f" -> Error processing data setup for curve {curve_type}: {e}")
                continue

            # Calculate squared distances for the relevant subset
            try:
                dist_sq = (time_subset - x)**2 + (curve_subset - y)**2
            except ValueError as e:
                 app_logger.error(f" -> Error calculating dist_sq for {curve_type}: {e}. Shapes: time={time_subset.shape}, curve={curve_subset.shape}, x={x}, y={y}")
                 continue # Skip this curve if calculation fails
            
            # Find the index of the minimum distance within the subset
            if len(dist_sq) == 0:
                 app_logger.debug(f" -> dist_sq array empty for {curve_type}.")
                 continue # Skip if empty after processing
            closest_idx_in_subset = np.nanargmin(dist_sq)
            min_dist_sq_curve = dist_sq[closest_idx_in_subset]
            
            app_logger.info(f" -> Curve {curve_type}: min_dist_sq={min_dist_sq_curve:.4f} at subset index {closest_idx_in_subset}")

            # If this point is closer than the best found so far
            if min_dist_sq_curve < min_dist_sq:
                min_dist_sq = min_dist_sq_curve
                found_on_curve = curve_type # Track the best curve found so far
                
                # The actual index in the *original* specific curve data array (e.g., purple_curve)
                original_curve_idx = closest_idx_in_subset
                # The corresponding index in the *full time_data* array
                time_data_idx = closest_idx_in_subset + idx_offset
                
                # Simplified: orange index *is* the time_data index relative to the start of analysis time.
                orange_idx = time_data_idx

                # Store info about this potentially best point
                best_point_info = {
                    'curve_type': curve_type,
                    'point_idx': original_curve_idx, # Index within the specific curve data
                    'orange_idx': orange_idx, # Index relative to the start of the orange curve/time_data
                    'x': time_subset[closest_idx_in_subset],
                    'y': curve_subset[closest_idx_in_subset],
                    'dist_sq': min_dist_sq_curve
                }
                app_logger.info(f" -> New best point candidate found on {curve_type}. dist_sq={min_dist_sq_curve:.4f}")
            else:
                 # app_logger.debug(f" -> Point on {curve_type} (dist_sq={min_dist_sq_curve:.4f}) not closer than current best (dist_sq={min_dist_sq:.4f} on {found_on_curve}).") # Kept as DEBUG
                 pass

        # Check distance threshold for the overall best point found
        if best_point_info and best_point_info['dist_sq'] <= max_dist_sq:
            app_logger.info(f"<<< Nearest point ACCEPTED on {best_point_info['curve_type']} at index {best_point_info['point_idx']} (Orange: {best_point_info['orange_idx']}). DistSq: {best_point_info['dist_sq']:.4f} <= Threshold: {max_dist_sq:.4f} >>>")
            # Remove dist_sq before returning if not needed elsewhere
            del best_point_info['dist_sq'] 
            return best_point_info
        elif best_point_info:
            # A point was found, but it was too far
            app_logger.info(f"Nearest point REJECTED (on {best_point_info['curve_type']} idx {best_point_info['point_idx']}) was too far: dist_sq {best_point_info['dist_sq']:.4f} > threshold {max_dist_sq:.4f}")
            return None
        else:
            # No point found on any curve or all failed checks
            app_logger.info("No point found on any curve within threshold after checking all curves.")
            return None

    def _get_corresponding_orange_point(self, curve_type, point_idx):
        """DEPRECATED or needs simplification. Orange index is derived directly in _get_nearest_point now."""
        # This logic is complex and potentially wrong. 
        # The mapping should be based on the time_data index calculated in _get_nearest_point.
        # Returning the time_data index directly as orange_idx is simpler for this structure.
        # app_logger.warning("_get_corresponding_orange_point is deprecated. Use index derived in _get_nearest_point.")
        # Fallback/placeholder - this likely won't be used if _get_nearest_point is correct
        if curve_type == 'orange':
            return point_idx
        else:
            # This is just a guess based on old logic, likely incorrect
            # Correct calculation is time_data_idx = closest_idx_in_subset + idx_offset
            # Need to pass this index from the caller or recalculate offset here.
            # The approach in _get_nearest_point is better.
            pass 
        return None # Indicate deprecated/handled elsewhere

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