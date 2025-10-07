#!/usr/bin/env python3

"""
Simple script to play recorded detection videos with VLC.
"""

import os
import sys
import subprocess
import glob
from datetime import datetime


def find_latest_recording():
    """Find the most recent recording file."""
    recordings_dir = os.path.expanduser("~/ros2_ws/recordings")
    if not os.path.exists(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return None
    
    # Find all MP4 files
    pattern = os.path.join(recordings_dir, "detection_recording_*.mp4")
    files = glob.glob(pattern)
    
    if not files:
        print("No recording files found")
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def list_recordings():
    """List all available recordings."""
    recordings_dir = os.path.expanduser("~/ros2_ws/recordings")
    if not os.path.exists(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return []
    
    pattern = os.path.join(recordings_dir, "detection_recording_*.mp4")
    files = glob.glob(pattern)
    
    if not files:
        print("No recording files found")
        return []
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    print("Available recordings:")
    for i, file in enumerate(files, 1):
        filename = os.path.basename(file)
        size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        print(f"{i:2d}. {filename} ({size:.1f} MB) - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return files


def play_video(filepath):
    """Play video using VLC."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    try:
        print(f"Playing: {os.path.basename(filepath)}")
        subprocess.run(['vlc', filepath], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error: VLC failed to play the video")
        return False
    except FileNotFoundError:
        print("Error: VLC not found. Please install VLC media player.")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            list_recordings()
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python3 play_recording.py           # Play latest recording")
            print("  python3 play_recording.py --list    # List all recordings")
            print("  python3 play_recording.py <file>    # Play specific file")
            print("  python3 play_recording.py --help    # Show this help")
            return
        else:
            # Play specific file
            filepath = sys.argv[1]
            if not os.path.isabs(filepath):
                # If relative path, assume it's in recordings directory
                recordings_dir = os.path.expanduser("~/ros2_ws/recordings")
                filepath = os.path.join(recordings_dir, filepath)
            
            play_video(filepath)
            return
    
    # Default: play latest recording
    latest_file = find_latest_recording()
    if latest_file:
        play_video(latest_file)
    else:
        print("No recordings found. Run the detection system first to create recordings.")
        print("Use '--list' to see available recordings.")


if __name__ == '__main__':
    main()
