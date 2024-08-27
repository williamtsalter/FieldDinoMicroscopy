# FieldDino

FieldDino is a Python-based application that enables the simple control of Dino-LITE microscopes in the field, for fast
phenotyping of large field trials. It is paired with YOLO object detection capabilities. FieldDino is 
designed for field researchers who need to capture, analyze, and annotate microscopic images in real-time.
![image](https://github.com/user-attachments/assets/675d2f4f-eb6c-4b0a-a024-7fed5197da9b)

| Original                                                                                                                        | Processed                                                                                                                      |
|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/user-attachments/assets/d2232762-e3fc-4608-86be-e515e66986c8" alt="1_B_I_R1_20230907" width="300"> | <img src="https://github.com/user-attachments/assets/9bf466e8-6526-40b2-a67f-41fb4833c510" alt="Processed Image"  width="300"> |

## Features

- **Microscope Control**: Interface with Dino-Lite digital microscopes for live image capture.
- **Image Processing**: Capture, save, and analyze both live and static images.
- **YOLO Integration**: Load and run YOLO models for object detection on microscope feeds or static images.
- **Customizable Settings**: Adjust microscope parameters like exposure and LED state.
- **Flexible File Naming**: Customize image filenames with field-specific information.

## Requirements

- Python 3.7-3.11 
- PyQt5
- OpenCV (cv2)
- Ultralytics YOLO
- Dino-Lite SDK (DNX64.dll)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/geezacolleman/FieldDino.git
   cd FieldDino
   ```

2. Install the required Python packages:
   ```
   pip install PyQt5 opencv-python ultralytics
   ```

3. Ensure the Dino-Lite SDK is installed and `DNX64.dll` is in the correct path (
   usually `C:\Program Files\DNX64\DNX64.dll`).

4. Place your YOLO model file (`.pt`) in the project directory or any accessible location.

### Dino-Lite SDK
Access to the Dino-Lite SDK can be requested through the official [Dino Lite website](https://www.dino-lite.com/download06.php).

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. **Microscope Control**:
    - Click "Start Microscope" to initialize the Dino-Lite microscope.
    - Adjust exposure and LED settings in the "Microscope Settings" panel.

3. **Image Capture**:
    - Click "Capture Image" to save the current view and move to the next filename, if an Excel file has been uploaded.
    - Set the save directory and customize filename prefixes in the settings.

4. **YOLO Model Integration**:
    - Load a YOLO model using the "Load Model" button.
    - Set confidence and IOU thresholds.
    - Click "Run Model" to start object detection on the live feed.

5. **Static Image Analysis**:
    - Click "Load Image" to analyze a pre-captured image.
    - Run the YOLO model on the loaded image for object detection.

6. **Settings**:
    - Customize file naming conventions in the "File Naming" section.
    - Set the save directory for captured images.
    - Adjust model parameters in the "YOLO Model Settings" section.

## Troubleshooting

- Ensure the Dino-Lite microscope is properly connected before starting the application.
- If the YOLO model fails to load, check the file path and ensure it's a compatible `.pt` file.
- For issues with image capture, verify that the save directory exists and is writable.

## Contributing

Contributions to FieldDino are welcome! Please fork the repository and submit a pull request with your improvements.

## License
h
[MIT License](LICENSE)

# References
Developed with the help of Anthropic Claude.

Icon Source:
<a href="https://www.flaticon.com/free-icons/jurassic" title="jurassic icons">Jurassic icons created by Marz Gallery -
Flaticon</a>
