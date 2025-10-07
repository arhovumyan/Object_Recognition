#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
from PIL import Image as PILImage


class MobileNetClassifier(Node):
    """
    ROS2 Node for MobileNetV3 object classification.
    Classifies cropped objects from YOLO detections to determine if they match target objects.
    """

    def __init__(self):
        super().__init__('mobilenet_classifier')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load MobileNetV3 model
        self.get_logger().info("Loading MobileNetV3 model...")
        try:
            self.model = MobileNetV3Small(
                input_shape=(224, 224, 3),
                weights='imagenet',
                include_top=True
            )
            self.get_logger().info("MobileNetV3 model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load MobileNetV3 model: {str(e)}")
            return
        
        # Define target classes and their ImageNet mappings - focus on phone detection
        self.target_mappings = {
            'phone': ['cellular_telephone', 'cellular_phone', 'cellphone', 'mobile_phone', 'telephone'],
            'mouse': ['computer_mouse', 'mouse', 'trackball'],
            'hat': ['hat', 'cap', 'baseball_cap', 'cowboy_hat', 'sombrero']
        }
        
        # Phone-specific classification settings
        self.phone_keywords = ['cellular_telephone', 'cellular_phone', 'cellphone', 'mobile_phone', 'telephone']
        self.phone_detection_count = 0
        self.total_classification_count = 0
        
        # Confidence threshold for classification - lower for phone detection
        self.classification_threshold = 0.2  # Lower threshold for better phone detection
        self.phone_confidence_threshold = 0.15  # Even lower for phone classification
        
        # Subscribers and publishers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.classification_pub = self.create_publisher(
            String,
            '/object_classification',
            10
        )
        
        self.target_found_pub = self.create_publisher(
            Bool,
            '/target_object_found',
            10
        )
        
        self.debug_image_pub = self.create_publisher(
            Image,
            '/classification_debug_image',
            10
        )
        
        # Store latest image for processing
        self.latest_image = None
        self.latest_image_header = None
        
        self.get_logger().info("MobileNet Classifier node initialized")

    def image_callback(self, msg):
        """
        Store the latest camera image for classification.
        """
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {str(e)}")

    def detection_callback(self, msg):
        """
        Process YOLO detections and classify objects.
        """
        if self.latest_image is None:
            self.get_logger().warn("No image available for classification")
            return
        
        try:
            debug_image = self.latest_image.copy()
            target_found = False
            classification_results = []
            
            for detection in msg.detections:
                # Extract bounding box
                center_x = detection.bbox.center.position.x
                center_y = detection.bbox.center.position.y
                size_x = detection.bbox.size_x
                size_y = detection.bbox.size_y
                
                # Calculate crop coordinates
                x1 = int(max(0, center_x - size_x/2))
                y1 = int(max(0, center_y - size_y/2))
                x2 = int(min(self.latest_image.shape[1], center_x + size_x/2))
                y2 = int(min(self.latest_image.shape[0], center_y + size_y/2))
                
                # Crop the object
                cropped_obj = self.latest_image[y1:y2, x1:x2]
                
                if cropped_obj.size == 0:
                    continue
                
                # Classify the cropped object
                classification_result = self.classify_object(cropped_obj)
                
                if classification_result:
                    class_name, confidence = classification_result
                    self.total_classification_count += 1
                    
                    # Check if this is a phone
                    is_phone = self.is_phone(class_name)
                    if is_phone:
                        self.phone_detection_count += 1
                        target_found = True
                        classification_results.append(f"PHONE: {confidence:.3f}")
                        
                        self.get_logger().info(f"PHONE DETECTED: {class_name} (confidence: {confidence:.3f})")
                        
                        # Draw bright yellow bounding box for phones
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 4)
                        cv2.putText(debug_image, f"PHONE: {confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                    else:
                        # Check if this is any other target object
                        if self.is_target_object(class_name):
                            classification_results.append(f"TARGET: {class_name}: {confidence:.3f}")
                            
                            # Draw green bounding box for other target objects
                            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(debug_image, f"TARGET: {class_name}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            classification_results.append(f"{class_name}: {confidence:.3f}")
                            
                            # Draw blue bounding box for non-target objects
                            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            cv2.putText(debug_image, f"{class_name}: {confidence:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Publish classification results
            if classification_results:
                result_msg = String()
                result_msg.data = "; ".join(classification_results)
                self.classification_pub.publish(result_msg)
            
            # Publish target found status
            target_msg = Bool()
            target_msg.data = target_found
            self.target_found_pub.publish(target_msg)
            
            # Log classification statistics every 20 classifications
            if self.total_classification_count > 0 and self.total_classification_count % 20 == 0:
                phone_percentage = (self.phone_detection_count / self.total_classification_count) * 100
                self.get_logger().info(f"Classification stats - Total: {self.total_classification_count}, Phones: {self.phone_detection_count} ({phone_percentage:.1f}%)")
            
            # Publish debug image
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header = self.latest_image_header
                self.debug_image_pub.publish(debug_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish debug image: {str(e)}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing detections: {str(e)}")

    def classify_object(self, cropped_image):
        """
        Classify a cropped object using MobileNetV3.
        """
        try:
            # Resize image to MobileNetV3 input size
            resized_image = cv2.resize(cropped_image, (224, 224))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = PILImage.fromarray(rgb_image)
            
            # Preprocess for MobileNetV3
            img_array = image.img_to_array(pil_image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Decode predictions
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Find the best prediction above threshold, with special handling for phones
            for class_id, class_name, confidence in decoded_predictions:
                # Lower threshold for phone detection
                threshold = self.phone_confidence_threshold if self.is_phone(class_name) else self.classification_threshold
                
                if confidence > threshold:
                    return class_name, confidence
            
            return None
            
        except Exception as e:
            self.get_logger().warn(f"Classification failed: {str(e)}")
            return None

    def is_phone(self, class_name):
        """
        Check if the classified object is a phone.
        """
        return any(keyword in class_name.lower() for keyword in self.phone_keywords)
    
    def is_target_object(self, class_name):
        """
        Check if the classified object is one of our target objects.
        """
        for target, keywords in self.target_mappings.items():
            if any(keyword in class_name.lower() for keyword in keywords):
                return True
        return False


def main(args=None):
    rclpy.init(args=args)
    
    classifier = MobileNetClassifier()
    
    try:
        rclpy.spin(classifier)
    except KeyboardInterrupt:
        pass
    finally:
        classifier.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
