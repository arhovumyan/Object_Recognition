#!/usr/bin/env python3

"""
Detection Logger for Two-Stage Pipeline
Logs all detections, classifications, and payload triggers with timestamps.

Features:
- CSV logging for analysis
- JSON logging for detailed records
- Real-time console output
- Automatic log rotation
- Performance metrics tracking
"""

import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path
import threading


class DetectionLogger:
    """Logger for detection and classification results."""
    
    def __init__(self, log_dir="logs", enable_console=True, enable_csv=True, enable_json=True):
        """
        Initialize detection logger.
        
        Args:
            log_dir: Directory to store log files
            enable_console: Enable console logging
            enable_csv: Enable CSV logging
            enable_json: Enable JSON logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        
        # Create log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.enable_csv:
            self.csv_file = self.log_dir / f"detections_{timestamp}.csv"
            self._init_csv()
        
        if self.enable_json:
            self.json_file = self.log_dir / f"detections_{timestamp}.json"
            self.json_logs = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'yolo_detections': 0,
            'filtered_detections': 0,
            'tent_detections': 0,
            'mannequin_detections': 0,
            'payload_triggers': 0,
            'session_start': datetime.now().isoformat()
        }
        
        self._log_system_start()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'event_type', 'yolo_class', 'yolo_confidence',
                'mobilenet_class', 'mobilenet_confidence', 'bbox_x1', 'bbox_y1',
                'bbox_x2', 'bbox_y2', 'center_x', 'center_y', 'target_type',
                'payload_type', 'is_target', 'inference_time_ms', 'fps',
                'device', 'frame_number'
            ])
    
    def _log_system_start(self):
        """Log system startup information."""
        startup_info = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'system_start',
            'log_dir': str(self.log_dir),
            'csv_enabled': self.enable_csv,
            'json_enabled': self.enable_json,
            'console_enabled': self.enable_console
        }
        
        if self.enable_console:
            print(f"[{startup_info['timestamp']}] SYSTEM START - Detection logging initialized")
            print(f"  Log directory: {self.log_dir}")
            print(f"  CSV logging: {self.enable_csv}")
            print(f"  JSON logging: {self.enable_json}")
        
        if self.enable_json:
            with self.lock:
                self.json_logs.append(startup_info)
    
    def log_yolo_detection(self, class_name, confidence, bbox, frame_number, fps, device):
        """Log YOLO detection result."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'yolo_detection',
            'yolo_class': class_name,
            'yolo_confidence': confidence,
            'bbox_x1': bbox[0], 'bbox_y1': bbox[1], 'bbox_x2': bbox[2], 'bbox_y2': bbox[3],
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'is_target': False,
            'fps': fps,
            'device': device,
            'frame_number': frame_number
        }
        
        self._write_log(log_entry)
        self.stats['yolo_detections'] += 1
        self.stats['total_detections'] += 1
    
    def log_filtered_detection(self, yolo_class, yolo_conf, bbox, frame_number, fps, device):
        """Log filtered detection (passed to MobileNetV3)."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'filtered_detection',
            'yolo_class': yolo_class,
            'yolo_confidence': yolo_conf,
            'bbox_x1': bbox[0], 'bbox_y1': bbox[1], 'bbox_x2': bbox[2], 'bbox_y2': bbox[3],
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'is_target': False,
            'fps': fps,
            'device': device,
            'frame_number': frame_number
        }
        
        self._write_log(log_entry)
        self.stats['filtered_detections'] += 1
    
    def log_classification_result(self, yolo_class, yolo_conf, mobilenet_class, mobilenet_conf,
                                bbox, is_target, target_type, payload_type, inference_time,
                                frame_number, fps, device):
        """Log MobileNetV3 classification result."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'classification_result',
            'yolo_class': yolo_class,
            'yolo_confidence': yolo_conf,
            'mobilenet_class': mobilenet_class,
            'mobilenet_confidence': mobilenet_conf,
            'bbox_x1': bbox[0], 'bbox_y1': bbox[1], 'bbox_x2': bbox[2], 'bbox_y2': bbox[3],
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'target_type': target_type,
            'payload_type': payload_type,
            'is_target': is_target,
            'inference_time_ms': inference_time * 1000,
            'fps': fps,
            'device': device,
            'frame_number': frame_number
        }
        
        self._write_log(log_entry)
        
        if is_target:
            if target_type == 'tent':
                self.stats['tent_detections'] += 1
            elif target_type == 'mannequin':
                self.stats['mannequin_detections'] += 1
    
    def log_payload_trigger(self, target_type, payload_type, confidence, center_x, center_y,
                          frame_number, fps):
        """Log payload drop trigger."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': 'payload_trigger',
            'target_type': target_type,
            'payload_type': payload_type,
            'mobilenet_confidence': confidence,
            'center_x': center_x,
            'center_y': center_y,
            'is_target': True,
            'fps': fps,
            'frame_number': frame_number
        }
        
        self._write_log(log_entry)
        self.stats['payload_triggers'] += 1
    
    def log_performance_update(self, fps, device, frame_number):
        """Log periodic performance updates."""
        if frame_number % 100 == 0:  # Every 100 frames
            timestamp = datetime.now().isoformat()
            
            log_entry = {
                'timestamp': timestamp,
                'event_type': 'performance_update',
                'fps': fps,
                'device': device,
                'frame_number': frame_number,
                'total_detections': self.stats['total_detections'],
                'yolo_detections': self.stats['yolo_detections'],
                'filtered_detections': self.stats['filtered_detections'],
                'tent_detections': self.stats['tent_detections'],
                'mannequin_detections': self.stats['mannequin_detections'],
                'payload_triggers': self.stats['payload_triggers']
            }
            
            self._write_log(log_entry)
    
    def _write_log(self, log_entry):
        """Write log entry to all enabled outputs."""
        with self.lock:
            # Console logging
            if self.enable_console:
                self._console_log(log_entry)
            
            # CSV logging
            if self.enable_csv:
                self._csv_log(log_entry)
            
            # JSON logging
            if self.enable_json:
                self.json_logs.append(log_entry)
    
    def _console_log(self, log_entry):
        """Format and print to console."""
        event_type = log_entry['event_type']
        timestamp = log_entry['timestamp']
        
        if event_type == 'yolo_detection':
            print(f"[{timestamp}] YOLO: {log_entry['yolo_class']} ({log_entry['yolo_confidence']:.2f}) "
                  f"at ({log_entry['center_x']:.0f}, {log_entry['center_y']:.0f})")
        
        elif event_type == 'filtered_detection':
            print(f"[{timestamp}] FILTERED: {log_entry['yolo_class']} -> MobileNetV3")
        
        elif event_type == 'classification_result':
            if log_entry['is_target']:
                print(f"[{timestamp}] 🎯 TARGET: {log_entry['mobilenet_class'].upper()} "
                      f"({log_entry['payload_type']}) conf={log_entry['mobilenet_confidence']:.2f} "
                      f"at ({log_entry['center_x']:.0f}, {log_entry['center_y']:.0f})")
            else:
                print(f"[{timestamp}] CLASSIFIED: {log_entry['mobilenet_class']} "
                      f"({log_entry['mobilenet_confidence']:.2f}) - not target")
        
        elif event_type == 'payload_trigger':
            print(f"[{timestamp}] 🚁 PAYLOAD TRIGGER: {log_entry['payload_type']} "
                  f"for {log_entry['target_type']} at ({log_entry['center_x']:.0f}, {log_entry['center_y']:.0f})")
        
        elif event_type == 'performance_update':
            print(f"[{timestamp}] 📊 PERFORMANCE: FPS={log_entry['fps']:.1f}, "
                  f"Targets={log_entry['tent_detections']}T/{log_entry['mannequin_detections']}M, "
                  f"Triggers={log_entry['payload_triggers']}")
    
    def _csv_log(self, log_entry):
        """Write to CSV file."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Map log entry to CSV row
            row = [
                log_entry.get('timestamp', ''),
                log_entry.get('event_type', ''),
                log_entry.get('yolo_class', ''),
                log_entry.get('yolo_confidence', ''),
                log_entry.get('mobilenet_class', ''),
                log_entry.get('mobilenet_confidence', ''),
                log_entry.get('bbox_x1', ''),
                log_entry.get('bbox_y1', ''),
                log_entry.get('bbox_x2', ''),
                log_entry.get('bbox_y2', ''),
                log_entry.get('center_x', ''),
                log_entry.get('center_y', ''),
                log_entry.get('target_type', ''),
                log_entry.get('payload_type', ''),
                log_entry.get('is_target', ''),
                log_entry.get('inference_time_ms', ''),
                log_entry.get('fps', ''),
                log_entry.get('device', ''),
                log_entry.get('frame_number', '')
            ]
            
            writer.writerow(row)
    
    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            return self.stats.copy()
    
    def save_json_logs(self):
        """Save JSON logs to file."""
        if not self.enable_json:
            return
        
        with self.lock:
            # Add final statistics
            final_stats = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'session_end',
                'session_duration': self._get_session_duration(),
                'final_stats': self.stats
            }
            self.json_logs.append(final_stats)
            
            # Save to file
            with open(self.json_file, 'w') as f:
                json.dump(self.json_logs, f, indent=2)
    
    def _get_session_duration(self):
        """Calculate session duration."""
        start_time = datetime.fromisoformat(self.stats['session_start'])
        return (datetime.now() - start_time).total_seconds()
    
    def print_summary(self):
        """Print session summary."""
        duration = self._get_session_duration()
        
        print("\n" + "="*60)
        print("DETECTION SESSION SUMMARY")
        print("="*60)
        print(f"Session duration: {duration:.1f} seconds")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"YOLO detections: {self.stats['yolo_detections']}")
        print(f"Filtered detections: {self.stats['filtered_detections']}")
        print(f"Tent detections: {self.stats['tent_detections']}")
        print(f"Mannequin detections: {self.stats['mannequin_detections']}")
        print(f"Payload triggers: {self.stats['payload_triggers']}")
        
        if self.enable_csv:
            print(f"CSV log: {self.csv_file}")
        if self.enable_json:
            print(f"JSON log: {self.json_file}")
        print("="*60)


# Global logger instance
_global_logger = None

def get_logger():
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DetectionLogger()
    return _global_logger

def cleanup_logger():
    """Cleanup and save logs."""
    global _global_logger
    if _global_logger:
        _global_logger.save_json_logs()
        _global_logger.print_summary()
        _global_logger = None


if __name__ == "__main__":
    # Test the logger
    logger = DetectionLogger()
    
    # Simulate some detections
    logger.log_yolo_detection("person", 0.85, [100, 100, 200, 200], 1, 30.0, "cuda")
    logger.log_filtered_detection("person", 0.85, [100, 100, 200, 200], 1, 30.0, "cuda")
    logger.log_classification_result("person", 0.85, "mannequin", 0.92, [100, 100, 200, 200],
                                   True, "mannequin", "water_bottle", 0.05, 1, 30.0, "cuda")
    logger.log_payload_trigger("mannequin", "water_bottle", 0.92, 150, 150, 1, 30.0)
    
    # Save and cleanup
    logger.save_json_logs()
    logger.print_summary()
