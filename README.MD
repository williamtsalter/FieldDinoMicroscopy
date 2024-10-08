# FieldDino

![FieldDino](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/Leaf%20Clip%20&%20FieldDino%20App%20-%203.jpeg)

FieldDino is a Python-based application that enables simple control of Dino-LITE microscopes in the field, for fast phenotyping of large scale field trials. Filenames are loaded in from an Excel spreadsheet based upon the sampling order. The app is also capable of running live YOLO segmentation in the field for measurement of stomatal traits using pre-trained models. FieldDino is designed for field researchers who need to capture, analyse, and annotate microscopic images in real-time.

We also include the 3D print files and instructions for a leaf-clip style microscope mount. We have designed the leaf clip to ease focussing in the field, speeding up image capture and standardising image brightness and scale across large data collection campaigns.

***Note that the FieldDino leaf clip and software were designed to work with the Dino-Lite AM7515 series of microscopes, however, should also work with other microscopes in the Dino-Lite range with minimal modifications.***

<details>
  <summary><h1 style="font-size:10vw">3D printed leaf clip microscope mount</h1></summary> 

The leaf clip should be 3D printed in two parts, the handle and the microscope mount/trigger. Note that we have provided different mount files depending on the magnification of your microscope. The mount should be printed in white (to diffuse the microscope LEDs) whilst the colour of the handle does not matter so much. We recommend using PETG filament, using supports whilst printing and using glue on the print bed to improve bed adhesion. We also recommend printing using fine (0.12 mm layers). After 3D printing, remove all support material and clean up print if needed using sandpaper.

|![Leaf clip](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/leaf%20clip.png)|![3D print bed](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/Screenshot%202024-09-30%20094539.png)|
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
|1. Slot the mount into the handle, lining up the bolt hole. You can use washers if there is a biut of a wiggle to the trigger |![step 1](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/step%201.png)|
|2. Using the bolt, nut and washers, secure the handle in place. Use threadlock to secure nut in place without overtightening | ![step 2](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/step%202.png)|
|3. Stick gasket foam to the handle and secure spring in place |![step 3](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/step%203.png)|
|4. Insert microscope into mount and twist to lock in place | ![step 4](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/step%204.png)|
|5. Tape around transparent plastic and sides of mount cap with three layers of black electrical tape | ![step 5](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/step%205.png)|

</details/>
  
<details>
  <summary><h1 style="font-size:10vw">FieldDino Python app <em>(Windows only for now)</em></h1></summary> 

![FieldDino App](https://github.com/williamtsalter/FieldDinoMicroscopy/blob/master/Images/FieldDino%20App%201.png)


## Features

- **Microscope Control**: Interface with Dino-Lite digital microscopes for live image capture.
- **Image Processing**: Capture, save, and analyze both live and static images.
- **YOLO Integration**: Load and run YOLO models for object detection on microscope feeds or static images.
- **Customizable Settings**: Adjust microscope parameters like exposure and LED state.
- **Flexible File Naming**: Customize image filenames with field-specific information.

<details>
  <summary><h2 style="font-size:7vw">FieldDino app prerequisites</h2></summary> 

<details>
  <summary><h3 style="font-size:5vw">Python installation</h3></summary> 

1. Install Python version 3.7-3.10 (https://www.python.org/)

2. Find the location of the Python executable file that you just installed, it will likely be in the following location:
        ```
        C:\Users\<USER>\AppData\Local\Programs\Python<version>
        ```
3. Open Windows Settings -> System -> About -> Advanced System Settings -> Environment Variables
    
4. Inside the upper window labeled 'User Variables' select 'Path' and edit
    
5. Select 'New' and add file directory for the location above
    
6. Select 'New' and repeat previous step for the Scripts folder within the Python<version> directory
  
7. Restart computer

</details>

<details>
  <summary><h3 style="font-size:5vw">Virtualenv install and test</h3></summary> 
1. Open a new command prompt (Windows Key + R, cmd.exe)

2. Install virtualenv through the command ```pip install virtualenv```

3. Check that virtualenv is installed through the command ```pip list```

4. Install virtualenvwrapper-win through the command ```pip install virtualenvwrapper-win```

5. Create a test environment with the command ```mkvirtualenv testenv```. Note this will also make a folder called Envs in your user directory

6. To leave an environment use ```deactivate```

7. You can re-enter an environment by using ```workon <environmentname>``` and from there can invoke package installations like ```pip install```

8. To completely delete an environment make sure you leave the environment first and use ```rmvirtualenv <environmentname>``` to delete the environment

</details>
  
<details>
  <summary><h3 style="font-size:5vw">Install Git for Windows</h3></summary> 
1. In a browser window navigate to the [Git download page](https://git-scm.com/downloads/win), download the standalone installer
    
2. Open the installer executable and follow the on screen prompts
  
3. Restart computer

</details>

<details>
  <summary><h3 style="font-size:5vw">Install .net Framework 3.5</h3></summary> 
1. In a browser window navigate to the [Microsoft Download Centre](https://www.microsoft.com/en-au/download/details.aspx?id=21) to download the installation file

2. Run the installer executable and follow on screen prompts

</details>

<details>
  <summary><h3 style="font-size:5vw">Request and install DinoLite SDK</h3></summary> 
1. In a browser window navigate to the [Dino-Lite software page](https://www.dino-lite.com/download06.php) and request the SDK license agreement
    
2. Once you receive a link to download the SDK, download and unzip the folder, then run the installer executable and follow on screen prompts
    
3. Ensure the Dino-Lite SDK is installed in the correct location, noting that `DNX64.dll` should be in `C:\Program Files\DNX64\DNX64.dll`.

</details>
</details>  

<details>
  <summary><h2 style="font-size:7vw">FieldDino app installation</h2></summary> 

We are now ready to install required packages for FieldDino using a requirements file. For reference, the required packages include:
- PyQt5
- OpenCV
- Openpyxl
- PyTorch CPU
- Ultralytics YOLO

1. Open a new command prompt and clone the GitHub repository:
   ```
   git clone https://github.com/williamtsalter/FieldDino.git
   cd FieldDino
   ```

2. Create a virtual environment for FieldDino using the command ```mkvirtualenv FieldDino```. To open this when you want to use the app, use the command ```workon FieldDino```

3. Install required Python packages from requirements file:
   ```
   pip install -r requirements.txt
   ```
4. If you want to run a model in real time, place your YOLO model file (`.pt`) in the project directory or any accessible location.

5. If you want to quickly run the app after install you can run the FieldDino.bat file. This batch file will open the FieldDino virtual environment then open the app. Note, you can also create a shortcut to this file to run from the Desktop.
        
</details>

<details>
  <summary><h2 style="font-size:7vw">FieldDino app usage</h2></summary> 

1. Open a new command prompt, open the FieldDino virtual environment (```workon FieldDino```), navigate to the FieldDino folder (```cd FieldDino```) then start the application:
   ```
   python app.py
   ```
    <summary><h3 style="font-size:5vw">OR</h2></summary>
   Open the FieldDino folder in File Explorer, run the FieldDino.bat file. Note that you can also create a shortcut to this file on your desktop for quick and easy access.

2. **Microscope Control**:
    - Ensure microscope is connected to the computer.
    - Click "Start Microscope" to initialize the Dino-Lite microscope.
    - Adjust exposure and LED settings in the "Microscope Settings" panel.
    - Note that we have removed the ability to control axial illumination as it gave very harsh reflections.

3. Custom filename loading from Excel or CSV file
    - Select browse to identify an Excel file containing filenames in a column
    - Click read to load in column names, select the column containing filenames
    - Note, if you capture an image you need to replace (e.g., out of focus, captured by accident) you can press back to return to the previous filename in the column
  
4. **Image Capture**:
    - Set the save directory and customize filename prefixes in the settings.
    - Click "Capture Image" to save the current view and move to the next filename, if an Excel file has been uploaded.

5. **YOLO Model Integration**:
    - Load a YOLO model using the "Load Model" button.
    - Set confidence and IOU thresholds.
    - Click "Run Model" to start object detection on the live feed.

| Original                                                                                                                        | Processed                                                                                                                      |
|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/user-attachments/assets/d2232762-e3fc-4608-86be-e515e66986c8" alt="1_B_I_R1_20230907" width="300"> | <img src="https://github.com/user-attachments/assets/9bf466e8-6526-40b2-a67f-41fb4833c510" alt="Processed Image"  width="300"> |

6. **Static Image Analysis**:
    - Click "Load Image" to analyze a pre-captured image.
    - Run the YOLO model on the loaded image for object detection.

7. **Settings**:
    - Customize file naming conventions in the "File Naming" section.
    - Set the save directory for captured images.
    - Adjust model parameters in the "YOLO Model Settings" section.

</details>

<details>
  <summary><h2 style="font-size:7vw">FieldDino app troubleshooting</h2></summary> 
- Ensure the Dino-Lite microscope is properly connected to the computer before starting the application. Note we have tested the app thoroughly with USB cable connections only, not the WI-FI DinoLite models.
- If you still struggle to connect the microscope, you can open the app.py file and edit lines 63 and 64 changing the ```self.device_index``` and ```self.cv2_cam_index``` variable. These tend to depend on the number of cameras embedded or connected to the computer. You can run ```python cameralist.py``` to identify connected cameras and microscopes.
- If the YOLO model fails to load, check the file path and ensure it's a compatible `.pt` file.
- For issues with image capture, verify that the save directory exists and is writable.
        
</details>
</details>

<details>
  <summary><h1 style="font-size:10vw">Contributing to FieldDino</h1></summary>

Contributions to FieldDino are welcome! Please fork the repository and submit a pull request with your improvements.

</details> 

<details>
  <summary><h1 style="font-size:10vw">License</h1></summary>

[MIT Open Source License](https://opensource.org/license/mit)

</details>

<details>
  <summary><h1 style="font-size:10vw">References and credits</h1></summary>        

Icon Source:
<a href="https://www.flaticon.com/free-icons/jurassic" title="jurassic icons">Jurassic icons created by Marz Gallery -
Flaticon</a>
        
</details>
