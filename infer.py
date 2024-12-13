import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image loading and visualization

# Load the trained model
MODEL_DIR = 'exported'
model = tf.keras.models.load_model(MODEL_DIR)

# Function to preprocess input images for inference
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Resize to match model input shape
    image_resized = cv2.resize(image, (128, 128))
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch, image_resized  # Also return resized image for visualization

# Perform inference
def infer_landmarks(image_path):
    input_image, original_image = preprocess_image(image_path)
    predictions = model.predict(input_image)

    # Reshape predictions into (68, 2) landmarks
    landmarks = predictions[0].reshape(-1, 2)

    # Denormalize landmarks to match original image dimensions
    original_height, original_width, _ = original_image.shape
    landmarks[:, 0] *= original_width  # Scale x-coordinates
    landmarks[:, 1] *= original_height  # Scale y-coordinates

    return landmarks, original_image

# Visualize landmarks
def visualize_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
IMAGE_PATH = r'D:\FLD\FLD\OIP.jpeg'  # Replace with the path to your test image
try:
    predicted_landmarks, original_image = infer_landmarks(IMAGE_PATH)
    visualize_landmarks(original_image, predicted_landmarks)
except Exception as e:
    print(f"Error during inference: {e}")
