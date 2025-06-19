import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import time

class FruitVegRecognition:
    def __init__(self):
        # Initialize the MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')
        
        # Dictionary to map general ImageNet categories to fruit/vegetable specific ones
        self.category_mapping = {
            'banana': 'banana',
            'orange': 'orange',
            'apple': 'apple',
            'cucumber': 'cucumber',
            'bell_pepper': 'bell pepper',
            'broccoli': 'broccoli',
            'cabbage': 'cabbage',
            'carrot': 'carrot',
            'tomato': 'tomato',
            'pineapple': 'pineapple'
        }
        
    def preprocess_frame(self, frame):
        # Resize the frame to 224x224 pixels
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to array and expand dimensions
        image_array = img_to_array(rgb_frame)
        image_array = np.expand_dims(image_array, axis=0)
        # Preprocess the image
        processed_image = preprocess_input(image_array)
        return processed_image
    
    def predict(self, frame):
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(processed_frame)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Filter for fruits and vegetables
        for (_, label, probability) in decoded_predictions:
            normalized_label = label.lower().replace('_', ' ')
            for category in self.category_mapping:
                if category in normalized_label:
                    return self.category_mapping[category], probability
        
        return None, 0.0

def main():
    # Initialize the recognition system
    recognition_system = FruitVegRecognition()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting fruit and vegetable recognition system...")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Make copy of frame for display
        display_frame = frame.copy()
        
        # Get prediction
        prediction, confidence = recognition_system.predict(frame)
        
        # Display results on frame
        if prediction:
            text = f"{prediction}: {confidence*100:.2f}%"
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No fruit/vegetable detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Fruit and Vegetable Recognition', display_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()