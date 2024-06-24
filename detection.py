import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('unet_lane_detection_tusimple.h5')

# Function to preprocess the entire frame
def preprocess_frame(frame, img_size=(256, 256)):
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame / 255.0  # Normalize the image
    return resized_frame, normalized_frame.reshape((1, img_size[0], img_size[1], 3))

# Function to postprocess the predicted mask
def postprocess_mask(predicted_mask, original_shape):
    # Ensure the predicted mask is scaled back correctly
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    # Binarize the resized mask
    predicted_mask_binarized = (predicted_mask_resized > 0.5).astype(np.uint8) * 255

    # Create an updated mask with zeros
    updated_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    bottom_45_percent_start = int(original_shape[0] * 0.65)
    # Retain only the bottom 45% of the detections
    updated_mask[bottom_45_percent_start:, :] = predicted_mask_binarized[bottom_45_percent_start:, :]
    
    return updated_mask

# Path to the input video file
input_video_path = 'input.mp4'
output_video_path = 'input_video_with_bottom_45_inference.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preprocess the entire frame
    resized_frame, preprocessed_frame = preprocess_frame(frame)

    # Debug: Display the preprocessed frame
    if frame_count == 1:
        plt.imshow(preprocessed_frame[0])
        plt.title("Preprocessed Frame")
        plt.show()

    # Run inference on the entire frame
    predicted_mask = model.predict(preprocessed_frame)[0]

    # Debug: Display the predicted mask
    if frame_count == 1:
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()

    # Postprocess the mask to retain only the bottom 45% detections
    predicted_mask_resized = postprocess_mask(predicted_mask, frame.shape[:2])

    # Debug: Display the resized mask
    if frame_count == 1:
        plt.imshow(predicted_mask_resized, cmap='gray')
        plt.title("Resized Mask")
        plt.show()

    # Overlay the mask on the original frame
    overlay = frame.copy()
    overlay[predicted_mask_resized > 0] = [0, 255, 0]  # Set lane pixels to green

    # Write the frame to the output video
    out.write(overlay)

# Release the video capture and writer objects
cap.release()
out.release()

print(f'Output video saved as {output_video_path}')
