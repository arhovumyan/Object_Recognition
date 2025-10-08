#!/usr/bin/env python3

"""
Script to analyze detection results from the drone object recognition system.
"""

import json
import csv
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def load_detection_log(json_file):
    """Load detection log from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_session_summary(json_file):
    """Load session summary from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_detections(detection_log):
    """Analyze detection data and generate statistics."""
    if not detection_log:
        print("No detection data found.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detection_log)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("=== DETECTION ANALYSIS ===")
    print(f"Total detections: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.1f} seconds")
    
    # Object type statistics
    print("\n=== OBJECT TYPE STATISTICS ===")
    object_counts = df['class_id'].value_counts()
    for obj_type, count in object_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{obj_type}: {count} detections ({percentage:.1f}%)")
    
    # Confidence statistics
    print("\n=== CONFIDENCE STATISTICS ===")
    print(f"Average confidence: {df['confidence'].mean():.3f}")
    print(f"Min confidence: {df['confidence'].min():.3f}")
    print(f"Max confidence: {df['confidence'].max():.3f}")
    print(f"Std deviation: {df['confidence'].std():.3f}")
    
    # Frame statistics
    print("\n=== FRAME STATISTICS ===")
    print(f"Frames with detections: {df['frame_number'].nunique()}")
    print(f"Total frames: {df['frame_number'].max()}")
    detection_rate = (df['frame_number'].nunique() / df['frame_number'].max()) * 100
    print(f"Detection rate: {detection_rate:.1f}% of frames")
    
    return df


def plot_detection_timeline(df, output_dir):
    """Create timeline plot of detections."""
    if df.empty:
        return
    
    plt.figure(figsize=(15, 8))
    
    # Create timeline plot
    for obj_type in df['class_id'].unique():
        obj_data = df[df['class_id'] == obj_type]
        plt.scatter(obj_data['timestamp'], obj_data['confidence'], 
                   label=obj_type, alpha=0.7, s=50)
    
    plt.xlabel('Time')
    plt.ylabel('Confidence')
    plt.title('Detection Timeline - Confidence over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timeline_file = os.path.join(output_dir, 'detection_timeline.png')
    plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
    print(f"Timeline plot saved: {timeline_file}")
    plt.close()


def plot_object_distribution(df, output_dir):
    """Create distribution plots for object types."""
    if df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Object count pie chart
    object_counts = df['class_id'].value_counts()
    ax1.pie(object_counts.values, labels=object_counts.index, autopct='%1.1f%%')
    ax1.set_title('Object Type Distribution')
    
    # Confidence distribution by object type
    df.boxplot(column='confidence', by='class_id', ax=ax2)
    ax2.set_title('Confidence Distribution by Object Type')
    ax2.set_xlabel('Object Type')
    ax2.set_ylabel('Confidence')
    
    plt.tight_layout()
    distribution_file = os.path.join(output_dir, 'object_distribution.png')
    plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved: {distribution_file}")
    plt.close()


def generate_report(session_summary, detection_log, output_dir):
    """Generate a comprehensive analysis report."""
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== DRONE OBJECT RECOGNITION ANALYSIS REPORT ===\n\n")
        
        # Session information
        f.write("SESSION INFORMATION:\n")
        f.write(f"Start time: {session_summary['session_start']}\n")
        f.write(f"End time: {session_summary['session_end']}\n")
        f.write(f"Duration: {session_summary['duration_seconds']:.1f} seconds\n")
        f.write(f"Total frames: {session_summary['total_frames']}\n")
        f.write(f"Average FPS: {session_summary['average_fps']:.1f}\n\n")
        
        # Camera settings
        f.write("CAMERA SETTINGS:\n")
        camera = session_summary['camera_settings']
        f.write(f"Device: {camera['device']}\n")
        f.write(f"Resolution: {camera['width']}x{camera['height']}\n")
        f.write(f"FPS: {camera['fps']}\n\n")
        
        # Detection statistics
        f.write("DETECTION STATISTICS:\n")
        f.write(f"Total detections: {session_summary['total_detections']}\n")
        
        detection_stats = session_summary['detection_statistics']
        if detection_stats:
            f.write("\nObject type breakdown:\n")
            for obj_type, count in detection_stats.items():
                percentage = (count / session_summary['total_detections']) * 100
                f.write(f"  {obj_type}: {count} ({percentage:.1f}%)\n")
        
        # Performance metrics
        f.write(f"\nPERFORMANCE METRICS:\n")
        detection_rate = (session_summary['total_detections'] / session_summary['total_frames']) * 100
        f.write(f"Detection rate: {detection_rate:.1f}% of frames\n")
        f.write(f"Detections per second: {session_summary['total_detections'] / session_summary['duration_seconds']:.1f}\n")
    
    print(f"Analysis report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze drone object recognition results')
    parser.add_argument('--detection-log', required=True, help='Path to detection log JSON file')
    parser.add_argument('--session-summary', required=True, help='Path to session summary JSON file')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading detection data...")
    detection_log = load_detection_log(args.detection_log)
    session_summary = load_session_summary(args.session_summary)
    
    # Analyze detections
    df = analyze_detections(detection_log)
    
    # Generate plots
    if df is not None and not df.empty:
        print("\nGenerating plots...")
        plot_detection_timeline(df, args.output_dir)
        plot_object_distribution(df, args.output_dir)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_report(session_summary, detection_log, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
