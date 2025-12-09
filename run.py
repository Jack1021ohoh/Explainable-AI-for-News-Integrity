#!/usr/bin/env python3
"""
Script to run the Fake News Detection Streamlit app
"""
import os
import sys
import subprocess


def main():
    """Run the Streamlit application"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the app
    app_path = os.path.join(script_dir, "app", "app.py")

    # Check if app exists
    if not os.path.exists(app_path):
        print(f"Error: App not found at {app_path}")
        sys.exit(1)

    print("Starting Fake News Detection System...")
    print(f"Running app: {app_path}")
    print("-" * 50)

    # Run streamlit
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: streamlit command not found. Please install streamlit:")
        print("  pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
