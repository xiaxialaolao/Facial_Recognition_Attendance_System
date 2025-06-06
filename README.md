# Face Recognition Attendance System

A Raspberry Pi-based face recognition attendance system that supports real-time face detection, recognition, attendance recording, and features a web management interface.

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

### Initial Setup

1. Clone or download the project to your Raspberry Pi

2. Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv FRAS_env

# Activate virtual environment
source FRAS_env/bin/activate
```

3. Install dependencies:

```bash
# Install required Python packages
pip install face_recognition opencv-python numpy scipy psutil picamera2 mysql-connector-python
```

4. Configure the database:

```bash
# Login to MySQL
mysql -u root -p

# Create database
CREATE DATABASE fras_db;

# Create user and grant privileges
CREATE USER 'fras_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON fras_db.* TO 'fras_user'@'localhost';
FLUSH PRIVILEGES;
```

5. Modify database configuration:
   - Edit the `HTML/config.php` file, update database connection information
   - Edit the `db_connector.py` file, update database connection information

### Starting the System

#### Method 1: Directly Start Components

1. Start the face recognition system:

```bash
# Ensure virtual environment is activated
source FRAS_env/bin/activate

# Start face recognition system
python FRAS.py
```

2. Start the web server (if not automatically started in FRAS.py):

```bash
# Ensure virtual environment is activated
source FRAS_env/bin/activate

# Start web server
python web_server.py
```

3. Run face feature extraction (as needed):

```bash
# Ensure virtual environment is activated
source FRAS_env/bin/activate

# Run face feature extraction
python FEFE.py
```

### Accessing the Web Interface

After the system starts, you can access the web interface via a browser:

```
http://[Raspberry Pi IP address]:80
```

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

```
├── FRAS.py                 # Main face recognition system
├── FEFE.py                 # Face feature extraction and encoding
├── web_server.py           # Web server
├── db_connector.py         # Database connector
├── gpio_control.py         # GPIO control module
├── HTML/                   # Web interface files
│   ├── config.php          # Database configuration
│   ├── dashboard.php       # Dashboard page
│   ├── image_acquisition.php # Image acquisition page
│   ├── login.php           # Login page
│   ├── log.php             # Log viewing page
│   └── ...                 # Other web files
├── Image_DataSet/          # Face image dataset
└── Encoding_DataSet/       # Feature encoding dataset
```

## Common Issues and Solutions

### 1. System Cannot Recognize Faces

- **Note: The face recognition model used in the system is relatively outdated**, with limited accuracy, and the recognition code implementation is not optimal, which may cause recognition issues
- The system works normally under ideal conditions but may perform poorly in complex environments
- Ensure adequate lighting, avoid backlighting
- Adjust camera position to ensure faces are clearly visible
- **Note distance factor**: Excessive distance will result in failed recognition, it's recommended to control an appropriate distance (typically 0.5-1.5 meters is suitable)
- Increase the number of facial images for each user (at least 10 images recommended)
- Try lowering the `RECOGNITION_TOLERANCE` value (modify in FRAS.py)
- Check if the user's feature encoding file exists in the `Encoding_DataSet` directory
- Run `FEFE.py` to regenerate feature encodings
- Consider using more modern face recognition libraries or models to improve accuracy (if needed for practical applications)

### 2. Web Interface Cannot Be Accessed

- Confirm the Web server is started
- Check firewall settings, ensure port 80 is open
- Verify the Raspberry Pi IP address is correct
- Check network connection
- View Web server logs: `tail -f logs/web_server.log`
- Try restarting the Web server: `python web_server.py`

### 3. Image Acquisition Failure

- Ensure the user has sufficient storage space: `df -h`
- Check if the camera connection is normal: `vcgencmd get_camera`
- Verify the user has write permissions to the Image_DataSet directory: `ls -la Image_DataSet/`
- Check error messages in log files: `tail -f logs/system.log`
- Ensure the directory exists: `mkdir -p Image_DataSet`
- Try restarting the system

### 4. System Performance Issues

- Lower camera resolution (current default is 1280x720, can be modified in FRAS.py by changing CAMERA_WIDTH and CAMERA_HEIGHT)
- Reduce the number of faces being recognized simultaneously
- Turn off unnecessary system services: `sudo systemctl disable <service_name>`
- Monitor system resource usage: `top` or `htop`
- Check system temperature: `vcgencmd measure_temp`
- Consider hardware upgrades (more RAM or faster SD card)

### 5. Database Connection Issues

- Check if the database service is running: `sudo systemctl status mysql`
- Verify database credentials are correct (check config.php and db_connector.py)
- Try connecting to the database manually: `mysql -u fras_user -p fras_db`
- Check database logs: `sudo tail -f /var/log/mysql/error.log`
- Restart database service: `sudo systemctl restart mysql`

### 6. Camera Issues

- Check if the camera is being used by other processes: `sudo lsof /dev/video0`
- Verify camera permissions: `ls -la /dev/video*`
- Try reloading the camera module:
  ```bash
  sudo rmmod bcm2835-v4l2
  sudo modprobe bcm2835-v4l2
  ```
- For Pi cameras, ensure the camera interface is enabled in `raspi-config`

## Important Notes

- **This system is primarily intended for school graduation projects**, suitable for learning and research purposes
- The system has certain limitations and room for optimization, not recommended for direct use in actual production environments
- As a teaching demonstration and academic research project, this system implements basic functionality but has room for improvement in stability, security, and performance
- If you want to develop actual applications based on this project, it is recommended to conduct a comprehensive code review and performance optimization
- This system is only for attendance management and should not be used for security-critical applications
- Please properly safeguard user data and comply with relevant privacy regulations
- Regularly backup the database and facial image dataset
- The system requires a stable power supply, it is recommended to use the official power adapter
