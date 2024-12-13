import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the trained model
MODEL_DIR = 'exported'  # Update this to your model directory
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
    return image_batch, image  # Also return the original image for visualization

# Perform inference
def infer_landmarks(image_path):
    input_image, original_image = preprocess_image(image_path)
    predictions = model.predict(input_image)

    # Reshape predictions into (68, 2) landmarks
    normalized_landmarks = predictions[0].reshape(-1, 2)

    # Denormalize landmarks to match original image dimensions
    original_height, original_width, _ = original_image.shape
    denormalized_landmarks = normalized_landmarks.copy()
    denormalized_landmarks[:, 0] *= original_width  # Scale x-coordinates
    denormalized_landmarks[:, 1] *= original_height  # Scale y-coordinates

    return normalized_landmarks, denormalized_landmarks, original_image

# Visualize denormalized landmarks on the image
def visualize_landmarks(image, landmarks, output_path=None):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    # Show the result
    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if output_path is provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")

# Visualize normalized landmarks on a unit square
def visualize_normalized_landmarks(normalized_landmarks):
    plt.scatter(normalized_landmarks[:, 0], normalized_landmarks[:, 1], s=10, c='red')
    plt.gca().invert_yaxis()  # Flip y-axis to match image coordinates
    plt.title("Normalized Landmarks")
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.grid()
    plt.show()

# Example usage
IMAGE_PATH = r'D:\FLD\archive\300W-LP\300W_LP\AFW\AFW_111076519_2_4.jpg'  # Replace with the path to your test image

try:
    normalized_landmarks, denormalized_landmarks, original_image = infer_landmarks(IMAGE_PATH)
    
    # Visualize landmarks on the original image
    visualize_landmarks(original_image.copy(), denormalized_landmarks)  # Use a copy to avoid modifying the original
    
    # Optionally visualize normalized landmarks
    visualize_normalized_landmarks(normalized_landmarks)

    # Print landmarks for debugging
    print("Normalized landmarks:\n", normalized_landmarks)
    print("Denormalized landmarks:\n", denormalized_landmarks)

except Exception as e:
    print(f"Error during inference: {e}")
