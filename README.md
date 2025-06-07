# Face Recognition Attendance System

A Raspberry Pi-based face recognition attendance system that supports real-time face detection, recognition, attendance recording, and features a web management interface.

<img src="https://github.com/xiaxialaolao/Facial_Recognition_Attendance_System/blob/main/Flow%20Diagram/Schematic%20Diagram.png" width="75%">

## System Architecture

The system consists of three main components:

### 1. Face Recognition System (FRAS.py)

Core face recognition engine, responsible for:
- Real-time camera capture
- Face detection and tracking
- Face feature comparison and recognition
- Attendance recording
- GPIO control (LED indicators)
- System resource monitoring

### 2. Face Feature Extraction and Encoding (FEFE.py)

Processes facial images and extracts feature vectors:
- Extracts facial features from image datasets
- Generates 128-dimensional feature vectors
- Saves encoding data for recognition

### 3. Web Management Interface (HTML directory)

Provides a user-friendly management interface:
- User management (add, edit, delete)
- Facial image acquisition
- Attendance record queries
- System log viewing
- System monitoring
- Multi-language support (Chinese, English)

## Main Features

- **Real-time Face Recognition**: Uses efficient face detection and recognition algorithms
- **Multi-person Recognition**: Supports simultaneous recognition of multiple faces
- **High-precision Recognition**: Uses 128-dimensional feature vectors for high accuracy
- **Automatic Attendance Recording**: Automatically records attendance information upon successful recognition
- **Web Management Interface**: Easy-to-use user management and data query interface
- **Mobile Device Compatible**: Responsive design, supports use on mobile devices
- **Multi-language Support**: Supports Chinese and English interfaces
- **System Monitoring**: Real-time monitoring of CPU, memory, temperature, and other system resources
- **Logging**: Detailed recording of system operations and error information

## Usage Instructions

### System Requirements

- Raspberry Pi 5 (recommended) or compatible device
- Python 3.7+
- Webcam or Raspberry Pi camera module
- At least 2GB RAM
- At least 16GB storage space
- MySQL/MariaDB database

## Face Data Collection

### Method 1: Collection via Web Interface

1. Log in to the web interface
2. Navigate to the "Image Acquisition" page
3. Select a user or create a new user
4. Use the webcam to capture facial images
5. The system will automatically process and save the images

### Method 2: Upload Image Files

1. Log in to the web interface
2. Navigate to the "Image Acquisition" page
3. Select a user
4. Click the "Upload Image" button
5. Select image files containing clear faces (supports JPG, JPEG, PNG formats)
6. The system will automatically process and save the images

## System Configuration Options

### FRAS.py Configuration Options

- `CAMERA_ROTATION`: Camera rotation angle (default: 0)
- `CAMERA_RESOLUTION`: Camera resolution (default: (1280, 720))
- `RECOGNITION_TOLERANCE`: Face recognition tolerance value (default: 0.06, lower is stricter)

### Data Collector Configuration Options

Configurable in FEFE.py:
- `IMAGE_DIR`: Directory for storing facial images (default: 'Image_DataSet')
- `ENCODING_DIR`: Directory for storing feature encodings (default: 'Encoding_DataSet')
- `SCAN_INTERVAL`: Time interval for automatically scanning new images, in seconds (default: 60)
- `FORCE_RECOMPUTE`: Whether to force regeneration of all feature encodings (default: False)

## Directory Structure

<img src="https://github.com/xiaxialaolao/Facial_Recognition_Attendance_System/blob/main/Flow%20Diagram/Directory%20Structure.png" width="40%">

## Important Notes

- **This system is primarily intended for school graduation projects**, suitable for learning and research purposes
- The system has certain limitations and room for optimization, not recommended for direct use in actual production environments
- As a teaching demonstration and academic research project, this system implements basic functionality but has room for improvement in stability, security, and performance
- If you want to develop actual applications based on this project, it is recommended to conduct a comprehensive code review and performance optimization
- This system is only for attendance management and should not be used for security-critical applications
- Please properly safeguard user data and comply with relevant privacy regulations
- Regularly backup the database and facial image dataset
- The system requires a stable power supply, it is recommended to use the official power adapter
