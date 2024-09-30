# FieldDino

![Leaf Clip   FieldDino App - 3](https://media.github.sydney.edu.au/user/5402/files/51868d3a-cdb2-4b70-a5eb-a78003cd75b8)

FieldDino is a Python-based application that enables simple control of Dino-LITE microscopes in the field, for fast phenotyping of large scale field trials. Filenames are loaded in from a CSV file based upon the sampling order. The app is also capable of running live YOLO segmentation in the field for measurement of stomatal traits using pre-trained models. FieldDino is designed for field researchers who need to capture, analyse, and annotate microscopic images in real-time.

We also include the 3D print files and instructions for a leaf-clip style microscope mount. We have designed the leaf clip to ease focussing in the field, speeding up image capture and standardising image brightness and scale across large data collection campaigns.

***Note that the FieldDino leaf clip and software were designed to work with the Dino-Lite AM7515 series of microscopes, however, should also work with other microscopes in the Dino-Lite range with minimal modifications.***

# 3D printed leaf clip microscope mount

The leaf clip should be 3D printed in two parts, the handle and the microscope mount/trigger. Note that we have provided different mount files depending on the magnification of your microscope. The mount should be printed in white (to diffuse the microscope LEDs) whilst the colour of the handle does not matter so much. We recommend using PETG filament, using supports whilst printing and using glue on the print bed to improve bed adhesion. We also recommend printing using fine (0.12 mm layers). After 3D printing, remove all support material and clean up print if needed using sandpaper.

|![Leaf clip](https://media.github.sydney.edu.au/user/5402/files/c8961e1d-9aa7-4e52-bae7-caa1de9b325b)|![image](https://media.github.sydney.edu.au/user/5402/files/36536d63-45a8-4553-a85f-f8a076c2ff67)|
|------------- | ------------- |

To assemble the leaf clip you will need some additional items that you can get from a hardware store:
- 3 mm thick EVA foam
- 38.1 mm x 6 mm extension spring
- 1 x 40 mm M5 bolt, 1 x nut and washers
- Threadlock to stop nut from coming loose during use
- Black electrical tape

Once you have all the parts above, you can assemble the leaf clip as follows: 

|Step | Image |
|------------- | ------------- |
|1. Slot the mount into the handle, lining up the bolt hole. You can use washers if there is a biut of a wiggle to the trigger |![step 1](https://media.github.sydney.edu.au/user/5402/files/082bb708-6649-4d78-88ae-3cd41b2da207)|
|2. Using the bolt, nut and washers, secure the handle in place. Use threadlock to secure nut in place without overtightening | ![step 2](https://media.github.sydney.edu.au/user/5402/files/a5985327-bd77-4388-9233-f0070610e6c6)|
|3. Stick gasket foam to the handle and secure spring in place |![step 3](https://media.github.sydney.edu.au/user/5402/files/2ddd63a9-c18a-4757-aa2a-d3331ead39bc)|
|4. Insert microscope into mount and twist to lock in place | ![step 4](https://media.github.sydney.edu.au/user/5402/files/5a51bb9f-d8c9-4081-94bd-21f1d1298104)|
|5. Tape around transparent plastic and sides of mount cap with two layers of black electrical tape | ![step 5](https://media.github.sydney.edu.au/user/5402/files/7d3078b7-16d1-4eee-b6dd-ec7ed80af873)|


# FieldDino Python app

![FieldDino App 1](https://media.github.sydney.edu.au/user/5402/files/f4820bfe-e144-40e0-9e86-57952b2b31e1)


## Features

- **Microscope Control**: Interface with Dino-Lite digital microscopes for live image capture.
- **Image Processing**: Capture, save, and analyze both live and static images.
- **YOLO Integration**: Load and run YOLO models for object detection on microscope feeds or static images.
- **Customizable Settings**: Adjust microscope parameters like exposure and LED state.
- **Flexible File Naming**: Customize image filenames with field-specific information.

## Getting started (Windows)

1. Install Python version 3.7-3.11 (https://www.python.org/)

2. After installation add Python to PATH
    - Find the location of the Python executable file that you just installed, it will likely be in the following location:
        ```
        C:\Users\<USER>\AppData\Local\Programs\Python[version]
        ```
    - Open Windows Control Panel -> User Accounts -> Change my Environment Variables
    - Inside the top window labeled 'User Variables' select 'Path' and edit
    - Select 'Browse' and navigate to the location above and select 'ok'
    - Select 'Move Down' and repeat step 3 for C:\python[version]\Scripts

3. Open a new command prompt (Windows Key + R, cmd.exe)

4. Install virtualenv through the command pip install virtualenv

5. Check that virtualenv is installed through the command pip --version

6. Install virtualenvwrapper-win through the command pip install virtualenvwrapper-win

## Test functionality of virtual environments
1. Create a test environment with the command mkvirtualenv example, and this will make a folder called Envs in your user directory

2. Change your python directories to a project folder in command prompt and run setprojectdir to set that as your development folder.

3. To leave an environment use deactivate

4. You can enter an environment by using workon example and from there can invoke package installations like pip install flask

5. To completely delete an environment make sure you leave the environment first and use rmvirtualenv example to delete the environment

- PyQt5
- OpenCV (cv2)
- Openpyxl
- PyTorch CPU
- Ultralytics YOLO
- Dino-Lite SDK (DNX64.dll)
- .net Framework 3.5 (on Windows)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/geezacolleman/FieldDino.git
   cd FieldDino
   ```

2. Install the required Python packages in the following order:
   ```
   pip install requirements.txt
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

| Original                                                                                                                        | Processed                                                                                                                      |
|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/user-attachments/assets/d2232762-e3fc-4608-86be-e515e66986c8" alt="1_B_I_R1_20230907" width="300"> | <img src="https://github.com/user-attachments/assets/9bf466e8-6526-40b2-a67f-41fb4833c510" alt="Processed Image"  width="300"> |

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
