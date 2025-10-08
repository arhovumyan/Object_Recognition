cmake_minimum_required(VERSION 3.8)
project(drone_object_recognition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/yolo_detector.py
  scripts/mobilenet_classifier.py
  scripts/object_recognition_pipeline.py
  scripts/phone_detection_monitor.py
  scripts/analyze_results.py
  scripts/play_recording.py
  scripts/yolo_detector_wrapper.py
  scripts/mobilenet_classifier_wrapper.py
  scripts/object_recognition_pipeline_wrapper.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install test scripts
install(PROGRAMS
  test_simple.py
  setup.sh
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
