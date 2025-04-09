"""
A lila görbék integrálási pontjainak vezérléséhez szükséges osztály.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from src.utils.logger import app_logger

class PurpleIntegrationController:
    """
    Vezérli a lila görbék integrálási kezdőpontjait.
    Vizuális jelzőket biztosít és pontos vezérlést ad az integrálás kezdetének meghatározásához.
    """
    
    def __init__(self, parent, canvas, ax, callback=None):
        """
        Inicializálja a vezérlőt.
        
        Args:
            parent: Szülő widget (általában ActionPotentialTab)
            canvas: Matplotlib canvas
            ax: Matplotlib tengely
            callback: Hívandó függvény az integrálási pontok változásakor
        """
        self.parent = parent
        self.canvas = canvas
        self.ax = ax
        self.update_callback = callback
        
        # Integrálási pontok tárolása
        self.integration_points = {
            'hyperpol': 0,  # Alapértelmezett: görbe eleje
            'depol': 0      # Alapértelmezett: görbe eleje
        }
        
        # Vizuális elemek
        self.markers = {
            'hyperpol': None,
            'depol': None
        }
        self.integration_patches = {
            'hyperpol': None,
            'depol': None
        }
        
        # Állapot követés
        self.processor = None
        self.is_active = False
        self.custom_points_enabled = False
        
        # Létrehozzuk a felhasználói felület vezérlőit
        self._create_ui_controls()
    
    def _create_ui_controls(self):
        """Létrehozza a felhasználói felület vezérlőit."""
        # Fő konténer (összecsukható LabelFrame)
        self.container = ttk.LabelFrame(self.parent.frame, text="Integrálási Pontok")
        
        # Váltó az egyéni integrálási pontok engedélyezéséhez
        self.enable_custom_points = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.container,
            text="Egyéni integrálási pontok engedélyezése",
            variable=self.enable_custom_points,
            command=self._toggle_custom_points
        ).pack(pady=2)
        
        # Vezérlők létrehozása beágyazott keretben
        self.controls_frame = ttk.Frame(self.container)
        self.controls_frame.pack(fill='x', padx=5, pady=2)
        
        # Hiperpolarizációs pont vezérlő
        hyperpol_frame = ttk.Frame(self.controls_frame)
        hyperpol_frame.pack(fill='x', pady=2)
        
        ttk.Label(hyperpol_frame, text="Hiperpol kezdet:").pack(side='left')
        self.hyperpol_point_var = tk.IntVar(value=0)
        self.hyperpol_spinbox = ttk.Spinbox(
            hyperpol_frame,
            from_=0,
            to=199,
            width=5,
            textvariable=self.hyperpol_point_var,
            command=lambda: self._update_integration_point('hyperpol')
        )
        self.hyperpol_spinbox.pack(side='left', padx=5)
        self.hyperpol_spinbox.bind('<Return>', lambda e: self._update_integration_point('hyperpol'))
        
        # Visszaállítás gomb
        ttk.Button(
            hyperpol_frame, 
            text="Visszaállítás",
            width=5,
            command=lambda: self._reset_integration_point('hyperpol')
        ).pack(side='left', padx=5)
        
        # Depolarizációs pont vezérlő
        depol_frame = ttk.Frame(self.controls_frame)
        depol_frame.pack(fill='x', pady=2)
        
        ttk.Label(depol_frame, text="Depol kezdet:").pack(side='left')
        self.depol_point_var = tk.IntVar(value=0)
        self.depol_spinbox = ttk.Spinbox(
            depol_frame,
            from_=0,
            to=199,
            width=5,
            textvariable=self.depol_point_var,
            command=lambda: self._update_integration_point('depol')
        )
        self.depol_spinbox.pack(side='left', padx=5)
        self.depol_spinbox.bind('<Return>', lambda e: self._update_integration_point('depol'))
        
        # Visszaállítás gomb
        ttk.Button(
            depol_frame, 
            text="Visszaállítás",
            width=5,
            command=lambda: self._reset_integration_point('depol')
        ).pack(side='left', padx=5)
        
        # Időértékek megjelenítése
        self.time_info_var = tk.StringVar(value="Hiperpol: 0.0ms, Depol: 0.0ms")
        ttk.Label(
            self.container, 
            textvariable=self.time_info_var,
            font=('TkDefaultFont', 8, 'italic')
        ).pack(pady=2)
        
        # Alkalmaz és Mindent visszaállít gombok
        button_frame = ttk.Frame(self.container)
        button_frame.pack(fill='x', pady=2)
        
        ttk.Button(
            button_frame, 
            text="Mindre alkalmaz",
            command=self._apply_integration_points
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Mindent visszaállít",
            command=self._reset_all_integration_points
        ).pack(side='left', padx=5)
        
        # Kezdetben letiltjuk a vezérlőket
        self._enable_controls(False)
    
    def _toggle_custom_points(self):
        """Váltja az egyéni integrálási pontok engedélyezését."""
        enabled = self.enable_custom_points.get()
        self.custom_points_enabled = enabled
        
        # Engedélyezzük/letiltjuk a vezérlőket
        self._enable_controls(enabled)
        
        # Frissítjük a vizuális jelzőket
        if self.processor:
            if enabled:
                self._add_integration_markers()
            else:
                self._remove_integration_markers()
                
            # Frissítjük az integrálokat ha van callback
            if self.update_callback:
                self.update_callback(self.get_integration_data())
    
    def _enable_controls(self, enabled):
        """Engedélyezi vagy letiltja az integrálási pont vezérlőket."""
        state = 'normal' if enabled else 'disabled'
        
        for widget in self.controls_frame.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Spinbox, ttk.Button)):
                    child.configure(state=state)
    
    def _update_integration_point(self, curve_type, evt=None):
        """Frissíti egy görbe integrálási pontját."""
        if curve_type not in ['hyperpol', 'depol']:
            return
            
        # Érték lekérése a megfelelő vezérlőből
        if curve_type == 'hyperpol':
            new_point = self.hyperpol_point_var.get()
        else:
            new_point = self.depol_point_var.get()
            
        # Határok ellenőrzése
        if self.processor:
            if curve_type == 'hyperpol' and hasattr(self.processor, 'modified_hyperpol'):
                max_point = len(self.processor.modified_hyperpol) - 1
                new_point = min(max_point, max(0, new_point))
            elif curve_type == 'depol' and hasattr(self.processor, 'modified_depol'):
                max_point = len(self.processor.modified_depol) - 1
                new_point = min(max_point, max(0, new_point))
        
        # Érvényesített pont tárolása
        self.integration_points[curve_type] = new_point
        
        # Vezérlő frissítése érvényesített értékkel
        if curve_type == 'hyperpol':
            self.hyperpol_point_var.set(new_point)
        else:
            self.depol_point_var.set(new_point)
        
        # Vizuális jelzők frissítése
        self._update_marker(curve_type)
        
        # Időértékek megjelenítésének frissítése
        self._update_time_display()
        
        # Változásokról értesítés
        if self.update_callback:
            self.update_callback(self.get_integration_data())
    
    def _reset_integration_point(self, curve_type):
        """Visszaállítja egy görbe integrálási pontját alapértelmezettre (0)."""
        if curve_type not in ['hyperpol', 'depol']:
            return
            
        # Visszaállítás alapértelmezettre
        self.integration_points[curve_type] = 0
        
        # Vezérlő frissítése
        if curve_type == 'hyperpol':
            self.hyperpol_point_var.set(0)
        else:
            self.depol_point_var.set(0)
        
        # Vizuális jelzők frissítése
        self._update_marker(curve_type)
        
        # Időértékek megjelenítésének frissítése
        self._update_time_display()
        
        # Változásokról értesítés
        if self.update_callback:
            self.update_callback(self.get_integration_data())
    
    def _reset_all_integration_points(self):
        """Visszaállítja az összes integrálási pontot alapértelmezettre."""
        self._reset_integration_point('hyperpol')
        self._reset_integration_point('depol')
    
    def _apply_integration_points(self):
        """Alkalmazza az aktuális integrálási pontokat a számításokra."""
        # Ez főként vizuális visszajelzés a felhasználónak
        # A tényleges alkalmazás a callback-ben történik
        if self.update_callback:
            self.update_callback(self.get_integration_data(), apply=True)
    
    def _update_marker(self, curve_type):
        """Frissíti egy integrálási pont vizuális jelzőjét."""
        if not self.processor:
            return
            
        # Meglévő jelző eltávolítása
        if self.markers[curve_type]:
            try:
                self.markers[curve_type].remove()
            except:
                pass
            self.markers[curve_type] = None
            
        # Meglévő kitöltés eltávolítása
        if self.integration_patches[curve_type]:
            try:
                self.integration_patches[curve_type].remove()
            except:
                pass
            self.integration_patches[curve_type] = None
        
        # Kihagyás ha az egyéni pontok nincsenek engedélyezve
        if not self.custom_points_enabled:
            self.canvas.draw_idle()
            return
            
        # Adatok lekérése görbetípus alapján
        if curve_type == 'hyperpol':
            if not hasattr(self.processor, 'modified_hyperpol') or not hasattr(self.processor, 'modified_hyperpol_times'):
                return
                
            data = self.processor.modified_hyperpol
            times = self.processor.modified_hyperpol_times
            color = 'blue'
        else:  # depol
            if not hasattr(self.processor, 'modified_depol') or not hasattr(self.processor, 'modified_depol_times'):
                return
                
            data = self.processor.modified_depol
            times = self.processor.modified_depol_times
            color = 'red'
        
        # Aktuális pont lekérése
        point_idx = self.integration_points[curve_type]
        if point_idx >= len(data):
            return
            
        # Végpont lekérése (RangeSelectionManager-ből ha elérhető)
        end_idx = self._get_end_point(curve_type)
        if end_idx is None or end_idx <= point_idx:
            return
            
        # Átváltás milliszekundumra a rajzoláshoz
        x_val = times[point_idx] * 1000
        y_val = data[point_idx]
        
        # Jelző hozzáadása a ponthoz
        self.markers[curve_type] = self.ax.plot(
            x_val, y_val, 
            'o', 
            color=color, 
            markersize=8, 
            markerfacecolor='white',
            markeredgewidth=2
        )[0]
        
        # Kitöltött terület létrehozása az integrálási területhez
        x_vals = times[point_idx:end_idx] * 1000
        y_vals = data[point_idx:end_idx]
        
        # Alap x koordináták létrehozása (nulla mentén)
        base_x = x_vals.copy()
        base_y = np.zeros_like(y_vals)
        
        # Koordináták összevonása sokszög létrehozásához
        vertices = np.column_stack([
            np.concatenate([x_vals, base_x[::-1]]),
            np.concatenate([y_vals, base_y[::-1]])
        ])
        
        # Kitöltés létrehozása
        from matplotlib.patches import Polygon
        self.integration_patches[curve_type] = Polygon(
            vertices,
            alpha=0.2,
            facecolor=color,
            edgecolor='none',
            zorder=0
        )
        self.ax.add_patch(self.integration_patches[curve_type])
        
        # Újrarajzolás
        self.canvas.draw_idle()
    
    def _get_end_point(self, curve_type):
        """Lekéri az integrálási végpontot a RangeSelectionManager-ből ha elérhető."""
        # RangeSelectionManager lekérése a szülőből
        range_manager = None
        if hasattr(self.parent, 'range_selection_manager'):
            range_manager = self.parent.range_selection_manager
        
        if range_manager and hasattr(range_manager, 'get_integration_ranges'):
            ranges = range_manager.get_integration_ranges()
            if curve_type in ranges:
                return ranges[curve_type]['end']
        
        # Alapértelmezett végpont (teljes görbe)
        if curve_type == 'hyperpol' and hasattr(self.processor, 'modified_hyperpol'):
            return len(self.processor.modified_hyperpol)
        elif curve_type == 'depol' and hasattr(self.processor, 'modified_depol'):
            return len(self.processor.modified_depol)
            
        return None
    
    def _add_integration_markers(self):
        """Hozzáadja az összes integrálási jelzőt a rajzhoz."""
        self._update_marker('hyperpol')
        self._update_marker('depol')
    
    def _remove_integration_markers(self):
        """Eltávolítja az összes integrálási jelzőt a rajzról."""
        for curve_type in ['hyperpol', 'depol']:
            if self.markers[curve_type]:
                try:
                    self.markers[curve_type].remove()
                except:
                    pass
                self.markers[curve_type] = None
                
            if self.integration_patches[curve_type]:
                try:
                    self.integration_patches[curve_type].remove()
                except:
                    pass
                self.integration_patches[curve_type] = None
                
        self.canvas.draw_idle()
    
    def _update_time_display(self):
        """Frissíti az időértékek megjelenítését az aktuális integrálási pontokhoz."""
        hyperpol_time = 0.0
        depol_time = 0.0
        
        if self.processor:
            if hasattr(self.processor, 'modified_hyperpol_times'):
                hyperpol_idx = min(self.integration_points['hyperpol'], 
                                len(self.processor.modified_hyperpol_times) - 1)
                hyperpol_time = self.processor.modified_hyperpol_times[hyperpol_idx] * 1000
                
            if hasattr(self.processor, 'modified_depol_times'):
                depol_idx = min(self.integration_points['depol'], 
                              len(self.processor.modified_depol_times) - 1)
                depol_time = self.processor.modified_depol_times[depol_idx] * 1000
        
        self.time_info_var.set(f"Hiperpol: {hyperpol_time:.1f}ms, Depol: {depol_time:.1f}ms")
    
    def set_processor(self, processor):
        """Beállítja az akciós potenciál processzort és frissíti az állapotot."""
        self.processor = processor
        
        # Vezérlők frissítése aktuális adatokkal
        if processor:
            # Spinboxok maximális értékeinek frissítése
            if hasattr(processor, 'modified_hyperpol'):
                max_hyperpol = max(0, len(processor.modified_hyperpol) - 1)
                self.hyperpol_spinbox.configure(to=max_hyperpol)
                
            if hasattr(processor, 'modified_depol'):
                max_depol = max(0, len(processor.modified_depol) - 1)
                self.depol_spinbox.configure(to=max_depol)
            
            # Vizuális jelzők frissítése ha aktív
            if self.custom_points_enabled:
                self._add_integration_markers()
                
            # Időértékek megjelenítésének frissítése
            self._update_time_display()
    
    def get_integration_data(self):
        """Lekéri az aktuális integrálási pont adatokat."""
        return {
            'enabled': self.custom_points_enabled,
            'points': self.integration_points.copy()
        }
    
    def show(self):
        """Megjeleníti az integrálási pont vezérlőket."""
        self.container.pack(fill='x', padx=5, pady=5)
        self.is_active = True
    
    def hide(self):
        """Elrejti az integrálási pont vezérlőket."""
        self.container.pack_forget()
        self.is_active = False
    
    def toggle_visibility(self):
        """Váltja az integrálási pont vezérlők láthatóságát."""
        if self.is_active:
            self.hide()
        else:
            self.show() 