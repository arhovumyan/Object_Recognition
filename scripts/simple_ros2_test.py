#!/usr/bin/env python3

"""
Simple ROS2 Test Script
This script demonstrates how to test the ROS2 YOLO detection system step by step.
"""

import subprocess
import time
import signal
import sys
import os


class ROS2TestRunner:
    """Simple test runner for ROS2 YOLO detection system."""
    
    def __init__(self):
        self.processes = []
        self.workspace_dir = "/home/aro/Documents/ObjectRec/ros2_ws"
    
    def run_command_background(self, cmd, name):
        """Run a command in the background."""
        print(f"🚀 Starting {name}...")
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append((process, name))
            print(f"✅ {name} started (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_all_processes(self):
        """Stop all background processes."""
        print("\n🛑 Stopping all processes...")
        for process, name in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔪 Force killed {name}")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        self.processes.clear()
    
    def test_individual_components(self):
        """Test individual components."""
        print("\n🧪 Testing Individual Components")
        print("=" * 50)
        
        # Test 1: Mock MAVROS Services
        cmd1 = f"cd {self.workspace_dir} && bash -c 'source install/setup.bash && ros2 run drone_object_recognition mock_mavros_services.py'"
        self.run_command_background(cmd1, "Mock MAVROS Services")
        time.sleep(3)
        
        # Test 2: YOLO Publisher
        cmd2 = f"cd {self.workspace_dir} && bash -c 'source install/setup.bash && ros2 run drone_object_recognition test_yolo_publisher.py'"
        self.run_command_background(cmd2, "YOLO Publisher")
        time.sleep(3)
        
        # Test 3: YOLO Detection Node
        cmd3 = f"cd {self.workspace_dir} && bash -c 'source install/setup.bash && ros2 run drone_object_recognition yolo_detection_ros2.py'"
        self.run_command_background(cmd3, "YOLO Detection Node")
        time.sleep(5)
        
        print("\n✅ All components started successfully!")
        print("Check the output above for any error messages.")
        
        # Let them run for a bit
        print("\n⏳ Letting components run for 10 seconds...")
        time.sleep(10)
    
    def test_topics_and_services(self):
        """Test ROS2 topics and services."""
        print("\n📡 Testing ROS2 Topics and Services")
        print("=" * 50)
        
        commands = [
            ("ros2 topic list", "List all topics"),
            ("ros2 service list", "List all services"),
            ("ros2 topic info /yolo_result", "YOLO result topic info"),
            ("ros2 topic info /mavros/global_position/global", "GPS topic info"),
            ("ros2 service info /mavros/mission/set_current", "Set waypoint service info")
        ]
        
        for cmd, description in commands:
            print(f"\n🔍 {description}")
            try:
                result = subprocess.run(
                    f"cd {self.workspace_dir} && bash -c 'source install/setup.bash && {cmd}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"✅ {description}:")
                    print(result.stdout)
                else:
                    print(f"❌ {description}: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {description}: Timeout")
            except Exception as e:
                print(f"❌ {description}: {e}")
    
    def monitor_detection_results(self):
        """Monitor YOLO detection results."""
        print("\n👁️ Monitoring YOLO Detection Results")
        print("=" * 50)
        print("This will show detection results for 15 seconds...")
        
        try:
            result = subprocess.run(
                f"cd {self.workspace_dir} && bash -c 'source install/setup.bash && timeout 15s ros2 topic echo /yolo_result'",
                shell=True,
                text=True
            )
        except Exception as e:
            print(f"❌ Error monitoring results: {e}")
    
    def run_full_test(self):
        """Run a complete test."""
        print("🤖 ROS2 YOLO Detection System - Full Test")
        print("=" * 60)
        
        try:
            # Start all components
            self.test_individual_components()
            
            # Test topics and services
            self.test_topics_and_services()
            
            # Monitor results
            self.monitor_detection_results()
            
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        finally:
            self.stop_all_processes()
    
    def interactive_menu(self):
        """Interactive test menu."""
        while True:
            print("\n" + "=" * 60)
            print("🎮 ROS2 YOLO Detection Test Menu")
            print("=" * 60)
            print("1. Test Individual Components")
            print("2. Test Topics & Services")
            print("3. Monitor Detection Results")
            print("4. Run Full Test")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                try:
                    self.test_individual_components()
                finally:
                    self.stop_all_processes()
            elif choice == "2":
                self.test_topics_and_services()
            elif choice == "3":
                self.monitor_detection_results()
            elif choice == "4":
                self.run_full_test()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")


def main():
    """Main function."""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n⚠️ Interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    runner = ROS2TestRunner()
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "full":
            runner.run_full_test()
        elif test_type == "components":
            try:
                runner.test_individual_components()
            finally:
                runner.stop_all_processes()
        else:
            print("❌ Unknown test type. Use: full, components, or no arguments for interactive menu")
    else:
        runner.interactive_menu()


if __name__ == "__main__":
    main()
