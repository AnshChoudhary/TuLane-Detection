import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('unet_lane_detection_tusimple.h5')

# Function to preprocess a frame
def preprocess_frame(frame, img_size=(256, 256)):
    frame_resized = cv2.resize(frame, img_size)
    frame_normalized = frame_resized / 255.0  # Normalize the image
    return frame_normalized.reshape((1, img_size[0], img_size[1], 3))

# Function to postprocess the predicted mask
def postprocess_mask(predicted_mask, original_shape):
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return predicted_mask_resized

# Open the input video
input_video_path = 'input2.mp4'
output_video_path = 'output2_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Run inference
    predicted_mask = model.predict(preprocessed_frame)[0]

    # Postprocess the mask
    predicted_mask_resized = postprocess_mask(predicted_mask, frame.shape)

    # Overlay the mask on the original frame
    overlay = frame.copy()
    overlay[predicted_mask_resized > 0.5] = [0, 255, 0]  # Set lane pixels to green

    # Write the frame to the output video
    out.write(overlay)

# Release the video capture and writer objects
cap.release()
out.release()

print(f'Output video saved as {output_video_path}')
