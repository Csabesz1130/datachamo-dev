import tkinter as tk
from .gui.app import SignalAnalyzerApp  # Fixed import
from .utils.logger import app_logger

def main():
    try:
        app_logger.info("Starting Signal Analyzer Application")
        root = tk.Tk()
        root.title("Signal Analyzer")
        
        # Set minimum window size
        root.minsize(1200, 800)
        
        # Start maximized
        root.state('zoomed')
        
        # Create and start application
        app = SignalAnalyzerApp(root)
        
        # Start main event loop
        root.mainloop()
        
    except Exception as e:
        app_logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()