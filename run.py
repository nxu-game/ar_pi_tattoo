#!/usr/bin/env python3
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
AR Pi Tattoo - Main entry point
"""

from ar_pi_tattoo.app import ARPiTattooApp

def main():
    """Main entry point for the AR Pi Tattoo application"""
    try:
        # Create and run the application
        app = ARPiTattooApp()
        app.setup()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 