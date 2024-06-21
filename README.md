# Autonomous Driving Research



## U-Net for Lane Detection on TuSimple Dataset
The [U_net_implementation.ipynb](https://github.com/AnshChoudhary/Autonomous-Driving/blob/main/U_net_implementation.ipynb) contains an implementation of a U-Net model for lane detection using the TuSimple dataset. The model is trained to segment lane markings in images captured from a vehicle's front-facing camera.

### Overview
U-Net is a convolutional neural network architecture designed for fast and precise image segmentation. This project applies U-Net for the task of lane detection in autonomous driving, using the TuSimple dataset.

### U-Net Architecture
The U-Net architecture was originally developed for biomedical image segmentation but has since been widely adopted for various image segmentation tasks, including lane detection. It is named "U-Net" because of its U-shaped structure, which is composed of two main parts: the contracting path (encoder) and the expanding path (decoder).

![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/7d973023-aa5c-4fa7-9817-acfdbb880df1)

#### 1. Contracting Path (Encoder)

The contracting path follows the typical architecture of a convolutional network. It consists of repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step, the number of feature channels is doubled.

#### 2. Bottleneck

At the bottom of the U, the network consists of convolutional layers with the maximum number of feature channels but without downsampling. This part captures the most abstract features of the input.

#### 3. Expanding Path (Decoder)

Every step in the expanding path consists of an upsampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels. Then, a concatenation with the corresponding cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. Cropping is necessary due to the loss of border pixels in every convolution.

#### 4. Final Layer

The final layer is a 1x1 convolution that maps each 64-component feature vector to the desired number of classes (e.g., 1 for binary segmentation). This is followed by a sigmoid activation function for binary segmentation tasks.

### Setup & Requirements

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

### Model Performance
| Metric           | Training       | Validation     |
|------------------|----------------|----------------|
| Loss             | 0.0137         | 0.1146         |
| Accuracy         | 0.9715         | 0.9590         |


### Predicting on sample images from the validation set
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/ed2d201b-44dd-4d8b-9fc0-b8cdd321b7b2)
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/ce4145a9-5ecd-4c1f-9629-9b9721ba9a6b)

### Running inference
Output with the Original Image and its corresponding model-generated mask:
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/e4beaca2-0ab5-49fb-8dd5-298718447e63)

Output with the Original Image and its corresponding model-generated mask being applied to the image:
![image](https://github.com/AnshChoudhary/Autonomous-Driving/assets/32743873/cca4f7d1-9cd2-4eb0-9aa3-148abc7a6165)

