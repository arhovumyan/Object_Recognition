#!/usr/bin/env python3

"""
MobileNetV3 Classifier for Tent/Mannequin Detection
Two-stage detection pipeline: YOLO → MobileNetV3 classification

Features:
- Pre-trained MobileNetV3-Small for efficiency
- GPU/CPU auto-detection and fallback
- Batch inference for multiple crops
- 3-class output: tent/mannequin/other
- Support for fine-tuning with custom dataset
- Optimized for drone real-time processing

Usage:
    classifier = MobileNetV3Classifier()
    predictions = classifier.classify_crops(cropped_images)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time


class MobileNetV3Classifier:
    """MobileNetV3 classifier for tent/mannequin detection."""
    
    def __init__(self, model_path=None, device=None, num_classes=3):
        """
        Initialize MobileNetV3 classifier.
        
        Args:
            model_path: Path to fine-tuned model weights (optional)
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
            num_classes: Number of output classes (3: tent/mannequin/other)
        """
        self.num_classes = num_classes
        self.class_names = ['tent', 'mannequin', 'other']
        
        # Setup device
        self.setup_device(device)
        
        # Initialize model
        self.setup_model(model_path)
        
        # Setup preprocessing
        self.setup_preprocessing()
        
        # Performance tracking
        self.inference_times = []
        self.batch_size = 4  # Optimize for GPU memory
        
    def setup_device(self, device):
        """Setup device with GPU/CPU auto-detection."""
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test GPU compatibility with a simple model operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    test_model = models.mobilenet_v3_small()
                    test_model = test_model.cuda()
                    # Try a forward pass
                    dummy_input = torch.randn(1, 3, 224, 224).cuda()
                    with torch.no_grad():
                        _ = test_model(dummy_input)
                    
                    # If we get here, GPU works
                    del test_tensor, test_model, dummy_input
                    torch.cuda.empty_cache()
                    self.device = 'cuda'
                    print("MobileNetV3: Using GPU acceleration")
                except Exception as e:
                    print(f"MobileNetV3: GPU detected but incompatible: {e}")
                    print("MobileNetV3: Falling back to CPU")
                    self.device = 'cpu'
                    torch.cuda.empty_cache()
            else:
                self.device = 'cpu'
                print("MobileNetV3: Using CPU")
        else:
            self.device = device
            
        print(f"MobileNetV3: Device set to {self.device.upper()}")
    
    def setup_model(self, model_path):
        """Initialize MobileNetV3 model."""
        print("MobileNetV3: Loading model...")
        
        # Load pre-trained MobileNetV3-Small
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify classifier for our 3 classes
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, self.num_classes)
        
        # Load fine-tuned weights if provided
        if model_path and os.path.exists(model_path):
            print(f"MobileNetV3: Loading fine-tuned weights from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print("MobileNetV3: Fine-tuned weights loaded successfully")
            except Exception as e:
                print(f"MobileNetV3: Warning - Could not load fine-tuned weights: {e}")
                print("MobileNetV3: Using pre-trained weights only")
        else:
            if model_path:
                print(f"MobileNetV3: Warning - Model path {model_path} not found")
            print("MobileNetV3: Using pre-trained ImageNet weights")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        print("MobileNetV3: Model loaded and ready for inference")
    
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess single image for MobileNetV3.
        
        Args:
            image: PIL Image or numpy array (BGR format)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Assume BGR format from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or numpy array")
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def classify_single_crop(self, crop):
        """
        Classify a single cropped image.
        
        Args:
            crop: PIL Image or numpy array of cropped region
            
        Returns:
            dict: {'class': class_name, 'confidence': score, 'class_id': id}
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_image(crop).to(self.device)
            
            # Inference
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_id = torch.max(probabilities, 1)
            
            class_id = class_id.item()
            confidence = confidence.item()
            
            # Track performance
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            return {
                'class': self.class_names[class_id],
                'confidence': confidence,
                'class_id': class_id
            }
    
    def classify_crops(self, crops):
        """
        Classify multiple cropped images efficiently.
        
        Args:
            crops: List of PIL Images or numpy arrays
            
        Returns:
            List of classification results
        """
        if not crops:
            return []
        
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for crop in batch_crops:
                tensor = self.preprocess_image(crop)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(batch_input)
                inference_time = time.time() - start_time
                
                # Get predictions for batch
                probabilities = torch.softmax(outputs, dim=1)
                confidences, class_ids = torch.max(probabilities, 1)
                
                # Convert to results
                for j in range(len(batch_crops)):
                    class_id = class_ids[j].item()
                    confidence = confidences[j].item()
                    
                    results.append({
                        'class': self.class_names[class_id],
                        'confidence': confidence,
                        'class_id': class_id
                    })
                
                # Track performance
                avg_time_per_crop = inference_time / len(batch_crops)
                self.inference_times.append(avg_time_per_crop)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
        
        return results
    
    def get_performance_stats(self):
        """Get inference performance statistics."""
        if not self.inference_times:
            return {'avg_time': 0, 'fps': 0}
        
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_time': avg_time,
            'fps': fps,
            'samples': len(self.inference_times)
        }
    
    def crop_from_bbox(self, image, bbox, padding=10):
        """
        Crop image region from bounding box with padding.
        
        Args:
            image: Full image (numpy array)
            bbox: [x1, y1, x2, y2] coordinates
            padding: Pixels to add around bbox
            
        Returns:
            Cropped image region
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop
        crop = image[y1:y2, x1:x1+(x2-x1)]
        
        return crop
    
    def filter_detections_by_class(self, detections, target_classes=['tent', 'mannequin'], 
                                 min_confidence=0.6):
        """
        Filter classification results to only include target classes.
        
        Args:
            detections: List of classification results
            target_classes: Classes to keep
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for detection in detections:
            if (detection['class'] in target_classes and 
                detection['confidence'] >= min_confidence):
                filtered.append(detection)
        
        return filtered


# Import cv2 for image processing
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available. Some functionality may be limited.")
    cv2 = None


def test_classifier():
    """Test the classifier with dummy data."""
    print("Testing MobileNetV3 Classifier...")
    
    # Initialize classifier
    classifier = MobileNetV3Classifier()
    
    # Create dummy test images
    dummy_crops = []
    for i in range(3):
        # Create random RGB image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_crops.append(Image.fromarray(dummy_img))
    
    # Test classification
    results = classifier.classify_crops(dummy_crops)
    
    print("Classification Results:")
    for i, result in enumerate(results):
        print(f"  Crop {i}: {result['class']} (conf: {result['confidence']:.3f})")
    
    # Performance stats
    stats = classifier.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Average inference time: {stats['avg_time']*1000:.2f}ms")
    print(f"  FPS: {stats['fps']:.1f}")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_classifier()
