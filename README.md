# Autonomous-Driving

## U-Net for Lane Detection on TuSimple Dataset
This repository contains an implementation of a U-Net model for lane detection using the TuSimple dataset. The model is trained to segment lane markings in images captured from a vehicle's front-facing camera.

### Overview
U-Net is a convolutional neural network architecture designed for fast and precise image segmentation. This project applies U-Net for the task of lane detection in autonomous driving, using the TuSimple dataset.

### Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/unet-lane-detection.git
    cd unet-lane-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Predicting on sample images from the validation set
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/ed2d201b-44dd-4d8b-9fc0-b8cdd321b7b2)
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/ce4145a9-5ecd-4c1f-9629-9b9721ba9a6b)

### Running inference
Output with the Original Image and its corresponding model-generated mask:
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/e4beaca2-0ab5-49fb-8dd5-298718447e63)

Output with the Original Image and its corresponding model-generated mask being applied to the image:
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/cca4f7d1-9cd2-4eb0-9aa3-148abc7a6165)

