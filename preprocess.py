import os
import tensorflow as tf
import cv2

def create_tfrecord(image_dir, landmarks_file, output_file):
    """
    Create a TFRecord file from images and landmarks.
    Args:
        image_dir: Directory containing the images.
        landmarks_file: Path to the landmarks text file.
        output_file: Path to save the TFRecord file.
    """
    with tf.io.TFRecordWriter(output_file) as writer:
        with open(landmarks_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                image_path = os.path.join(image_dir, image_name)
                landmarks = list(map(float, parts[1:]))

                # Convert landmarks to float32 (single precision)
                landmarks = [float(x) for x in landmarks]

                # Check if the image exists
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                # Read the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Invalid image: {image_path}")
                    continue

                # Encode the image to bytes
                _, image_encoded = cv2.imencode('.jpg', image)
                image_bytes = image_encoded.tobytes()

                # Serialize the landmarks
                landmarks_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(landmarks, dtype=tf.float32))

                # Create a TFRecord Example
                feature = {
                    'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name.encode()])),
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                    'label/marks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[landmarks_tensor.numpy()])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Write to the TFRecord file
                writer.write(example.SerializeToString())

    print(f"TFRecord saved to {output_file}")


# Example usage
image_dir = r"D:\FLD\archive\300W-LP\300W_LP\AFW"
train_landmarks_file = r"D:\FLD\FLD\landmarks_train.txt"
val_landmarks_file = r"D:\FLD\FLD\landmarks_val.txt"
train_tfrecord_file = r"D:\FLD\FLD\300w_train.record"
val_tfrecord_file = r"D:\FLD\FLD\300w_validation.record"

create_tfrecord(image_dir, train_landmarks_file, train_tfrecord_file)
create_tfrecord(image_dir, val_landmarks_file, val_tfrecord_file)
