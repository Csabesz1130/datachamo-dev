# test_imports.py
def test_imports():
    try:
        import tkinter
        print("✓ tkinter is available")
        
        import numpy
        print("✓ numpy is available")
        
        import scipy
        print("✓ scipy is available")
        
        import matplotlib
        print("✓ matplotlib is available")
        
        import pandas
        print("✓ pandas is available")
        
        # Test specific modules we use
        from matplotlib.figure import Figure
        print("✓ matplotlib.figure is available")
        
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("✓ matplotlib backend is available")
        
        from scipy.signal import savgol_filter, butter, filtfilt
        print("✓ scipy.signal filters are available")
        
        print("\nAll required packages are properly installed!")
        return True
        
    except ImportError as e:
        print(f"\nError importing package: {str(e)}")
        return False

if __name__ == "__main__":
    test_imports()