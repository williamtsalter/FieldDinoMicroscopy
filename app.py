import sys
import os
import pandas as pd

import cv2
import numpy as np

from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QLineEdit, QTextEdit, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QScrollArea,
                             QRadioButton, QButtonGroup, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from dnx64 import DNX64
from ultralytics import YOLO

class FieldDinoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FieldDino")
        self.setGeometry(100, 100, 1400, 800)

        self.driver_path = 'C:\\Users\\txp168\\Documents\\DNX64\\DNX64.dll'
        self.microscope = None
        self.device_index = 0
        self.cv2_cam_index = 1

        self.camera = None
        self.led_state = False
        self.device_name = ""
        self.frame = None
        self.ret = False

        self.model = None
        self.model_running = False
        self.current_image = None

        self.excel_data = None
        self.current_row = 0
        self.column_radio_buttons = []
        self.column_button_group = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def init_ui(self):
        # Set window icon
        self.setWindowIcon(QIcon("dino-icon.png"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: video feed and controls
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 2)  # Video feed takes 2/3 of the width

        # Video feed area
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_stop_button = self.create_large_button("Start Microscope")
        self.start_stop_button.clicked.connect(self.toggle_microscope)
        control_layout.addWidget(self.start_stop_button)

        self.capture_button = self.create_large_button("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        control_layout.addWidget(self.capture_button)

        left_layout.addLayout(control_layout)

        # Message window
        self.message_window = QTextEdit(self)
        self.message_window.setReadOnly(True)
        self.message_window.setMaximumHeight(100)
        left_layout.addWidget(self.message_window)

        # Right side: settings
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout, 1)  # Settings take 1/3 of the width

        # Microscope settings
        microscope_settings_group = QGroupBox("Microscope Settings")
        microscope_settings_layout = QFormLayout()
        self.exposure_spinbox = QSpinBox(self)
        self.exposure_spinbox.setRange(0, 100000)
        self.exposure_spinbox.valueChanged.connect(self.set_exposure)
        self.ae_target_spinbox = QSpinBox(self)
        self.ae_target_spinbox.setRange(10, 50)
        self.ae_target_spinbox.valueChanged.connect(self.set_ae_target)

        self.led_button = QPushButton("Turn On LED", self)
        self.led_button.clicked.connect(self.toggle_led)

        microscope_settings_layout.addRow("Exposure:", self.exposure_spinbox)
        microscope_settings_layout.addRow("AE Target:", self.ae_target_spinbox)
        microscope_settings_layout.addRow("LED Control:", self.led_button)
        microscope_settings_group.setLayout(microscope_settings_layout)
        settings_layout.addWidget(microscope_settings_group)

        # Filename Settings
        filename_group = QGroupBox("Filename Settings")
        filename_layout = QVBoxLayout()

        # Save directory input
        save_dir_layout = QHBoxLayout()
        self.save_dir_edit = QLineEdit(self)
        self.save_dir_edit.setPlaceholderText("Save Directory")
        save_dir_layout.addWidget(self.save_dir_edit)

        self.save_dir_browse_button = QPushButton("Browse", self)
        self.save_dir_browse_button.clicked.connect(self.browse_save_directory)
        save_dir_layout.addWidget(self.save_dir_browse_button)

        filename_layout.addLayout(save_dir_layout)

        # Excel file integration and Read Excel button
        excel_layout = QHBoxLayout()
        self.excel_path_edit = QLineEdit(self)
        self.excel_path_edit.setPlaceholderText("Excel File Path")
        excel_layout.addWidget(self.excel_path_edit)

        self.excel_browse_button = QPushButton("Browse", self)
        self.excel_browse_button.clicked.connect(self.browse_excel)
        excel_layout.addWidget(self.excel_browse_button)

        self.read_button = QPushButton("Read", self)
        self.read_button.clicked.connect(self.read_excel)
        excel_layout.addWidget(self.read_button)

        filename_layout.addLayout(excel_layout)

        # Column selection area
        self.column_selection_group = QGroupBox("Select Filename Column")
        self.column_selection_layout = QVBoxLayout()
        self.column_selection_group.setLayout(self.column_selection_layout)

        # Make the column selection area scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.column_selection_group)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Adjust this value as needed
        filename_layout.addWidget(scroll_area)

        self.filename_edit = QLineEdit(self)
        self.filename_edit.setPlaceholderText("Current Filename")
        self.filename_edit.setReadOnly(True)
        filename_layout.addWidget(self.filename_edit)

        # Navigation buttons
        navigation_layout = QHBoxLayout()
        self.back_button = self.create_large_button("Back")
        self.back_button.clicked.connect(self.previous_filename)
        navigation_layout.addWidget(self.back_button)

        self.next_button = self.create_large_button("Next")
        self.next_button.clicked.connect(self.next_filename)
        navigation_layout.addWidget(self.next_button)

        filename_layout.addLayout(navigation_layout)

        filename_group.setLayout(filename_layout)
        settings_layout.addWidget(filename_group)

        # YOLO Model settings
        model_settings_group = QGroupBox("YOLO Model Settings")
        model_settings_layout = QFormLayout()

        self.model_path_edit = QLineEdit(self)
        model_settings_layout.addRow("Model Path:", self.model_path_edit)

        self.model_browse_button = QPushButton("Browse", self)
        self.model_browse_button.clicked.connect(self.browse_model)
        model_settings_layout.addRow("", self.model_browse_button)

        self.confidence_spinbox = QDoubleSpinBox(self)
        self.confidence_spinbox.setRange(0, 1)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.25)
        model_settings_layout.addRow("Confidence:", self.confidence_spinbox)

        self.iou_spinbox = QDoubleSpinBox(self)
        self.iou_spinbox.setRange(0, 1)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        model_settings_layout.addRow("IOU:", self.iou_spinbox)

        model_control_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model", self)
        self.load_model_button.clicked.connect(self.load_model)
        model_control_layout.addWidget(self.load_model_button)

        self.run_model_button = QPushButton("Run Model", self)
        self.run_model_button.clicked.connect(self.toggle_run_model)
        model_control_layout.addWidget(self.run_model_button)
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)
        model_control_layout.addWidget(self.load_image_button)

        model_settings_layout.addRow("", model_control_layout)

        model_settings_group.setLayout(model_settings_layout)
        settings_layout.addWidget(model_settings_group)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.log_message("FieldDino started. Connect a microscope and click 'Start Microscope'.")

    def log_message(self, message):
        self.message_window.append(message)

    def toggle_microscope(self):
        if self.microscope is None:
            self.start_microscope()
        else:
            self.stop_microscope()

    def start_microscope(self):
        try:
            self.microscope = DNX64(self.driver_path)
            self.microscope.SetVideoDeviceIndex(self.device_index)
            self.microscope.Init()
            if not self.microscope.Init():
                raise Exception("Failed to initialize microscope")

            self.device_name = self.microscope.GetVideoDeviceName(self.device_index)
            self.setWindowTitle(f"FieldDino - Device ID: {self.device_name}")

            self.camera = cv2.VideoCapture(self.cv2_cam_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")

            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

            self.start_stop_button.setText("Stop Microscope")
            self.log_message("Microscope started successfully.")

            # Initialize microscope settings
            self.update_microscope_settings()
            self.microscope.SetLEDState(self.device_index, 0)
            self.led_state = 0

            self.timer.start(33)

        except Exception as e:
            self.log_message(f"Error starting microscope: {str(e)}")
            self.stop_microscope()

    def stop_microscope(self):
        self.timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.microscope:
            self.microscope = None
        self.video_label.clear()
        self.start_stop_button.setText("Start Microscope")
        self.setWindowTitle("FieldDino")
        self.device_name = ""
        self.log_message("Microscope stopped.")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.log_message(f"Image loaded: {file_name}")
            else:
                self.log_message(f"Failed to load image: {file_name}")

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def toggle_run_model(self):
        if not self.model:
            self.log_message("Please load a model first.")
            return

        if not self.model_running:
            self.model_running = True
            self.run_model_button.setText("Stop Model")
            self.log_message("Model running. Processing frames or loaded image...")
            if self.current_image is not None and self.camera is None:
                self.process_image_with_model(self.current_image)
        else:
            self.model_running = False
            self.run_model_button.setText("Run Model")
            self.log_message("Model stopped.")

    def process_image_with_model(self, image):
        if self.model and self.model_running:
            try:
                # Get image dimensions
                height, width = image.shape[:2]

                # Define tile size
                tile_size = 640

                # Create a copy of the original image for drawing results
                annotated_image = image.copy()

                # Process image in tiles
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        # Extract tile
                        tile = image[y:y + tile_size, x:x + tile_size]

                        # Pad tile if it's smaller than tile_size
                        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                            padded_tile[:tile.shape[0], :tile.shape[1], :] = tile
                            tile = padded_tile

                        # Run model on tile
                        results = self.model(tile, conf=self.confidence_spinbox.value(), iou=self.iou_spinbox.value(), imgsz=(640,640))

                        # Adjust bounding boxes for tile position
                        for r in results:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            adjusted_boxes = boxes + np.array([x, y, x, y])

                            # Draw bounding boxes and labels on the annotated image
                            for box in adjusted_boxes:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Add label (assuming the model provides class names)
                                if r.names:
                                    label = r.names[int(r.boxes.cls[0])]
                                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the annotated image
                self.display_image(annotated_image)

                self.log_message("Model processing completed.")
            except Exception as e:
                self.log_message(f"Error processing image with model: {str(e)}")

    def update_frame(self):
        if self.camera:
            self.ret, self.frame = self.camera.read()
            if self.ret:
                if self.model_running and self.model:
                    self.process_image_with_model(self.frame)
                else:
                    self.display_image(self.frame)

            else:
                self.log_message("Failed to capture frame.")
        elif self.current_image is not None and self.model_running and self.model:
            self.process_image_with_model(self.current_image)

    def toggle_led(self):
        if self.microscope:
            try:
                self.led_state = not self.led_state
                self.microscope.SetLEDState(self.device_index, int(self.led_state))
                self.led_button.setText("Turn Off LED" if self.led_state else "Turn On LED")
                self.log_message(f"LED turned {'on' if self.led_state else 'off'}.")
            except Exception as e:
                self.log_message(f"Error toggling LED: {str(e)}")
        else:
            self.log_message("Microscope not connected. Cannot toggle LED.")

    def update_microscope_settings(self):
        if self.microscope:
            try:
                exposure = self.microscope.GetExposureValue(self.device_index)
                self.exposure_spinbox.setValue(exposure)
                ae_target = self.microscope.GetAETarget(self.device_index)
                self.ae_target_spinbox.setValue(ae_target)
                self.led_button.setText("Turn Off LED" if self.led_state else "Turn On LED")
            except Exception as e:
                self.log_message(f"Error updating microscope settings: {str(e)}")

    def set_exposure(self, value):
        if self.microscope:
            try:
                self.microscope.SetExposureValue(self.device_index, value)
                self.log_message(f"Exposure set to {value}")
            except Exception as e:
                self.log_message(f"Error setting exposure: {str(e)}")
        else:
            self.log_message("Microscope not connected. Cannot set exposure.")

    def set_ae_target(self, value):
        if self.microscope:
            try:
                self.microscope.SetAETarget(self.device_index, value)
                self.log_message(f"AE Target set to {value}")
            except Exception as e:
                self.log_message(f"Error setting AE Target: {str(e)}")
        else:
            self.log_message("Microscope not connected. Cannot set AE Target.")

    def capture_image(self):
        if self.ret:
            self.save_image(self.frame)
            self.next_filename()

        elif self.current_image is not None:
            self.save_image(self.current_image)

        else:
            self.log_message("No image to capture. Please start the microscope or load an image.")

    def save_image(self, image):
        save_dir = self.save_dir_edit.text() or "."
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        excel_filename = self.get_current_filename()

        if excel_filename:
            filename = f"{excel_filename}_{timestamp}"
        else:
            filename = timestamp

        filename += ".png"
        full_path = os.path.join(save_dir, filename)
        cv2.imwrite(full_path, image)
        self.log_message(f"Image saved: {full_path}")

    def browse_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.save_dir_edit.setText(directory)
            self.log_message(f"Save directory set to: {directory}")

    def browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if file_name:
            self.model_path_edit.setText(file_name)
            self.log_message(f"Model path set to: {file_name}")

    def load_model(self):
        model_path = self.model_path_edit.text()
        if not model_path:
            self.log_message("Please specify a model path.")
            return

        try:
            self.model = YOLO(model_path)
            self.log_message(f"Model loaded successfully: {model_path}")
        except Exception as e:
            self.log_message(f"Error loading model: {str(e)}")
            self.model = None

    def browse_excel(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_name:
            self.excel_path_edit.setText(file_name)
            self.log_message(f"Excel file selected: {file_name}")
            self.read_excel_columns()

    def read_excel_columns(self):
        try:
            excel_path = self.excel_path_edit.text()

            if not excel_path:
                raise ValueError("Please provide an Excel file path.")

            self.excel_data = pd.read_excel(excel_path)

            # Clear existing radio buttons
            for button in self.column_radio_buttons:
                self.column_selection_layout.removeWidget(button)
                button.deleteLater()
            self.column_radio_buttons.clear()

            if self.column_button_group:
                self.column_button_group.deleteLater()

            # Create new radio buttons for each column
            self.column_button_group = QButtonGroup(self)
            for column in self.excel_data.columns:
                radio_button = QRadioButton(column)
                self.column_radio_buttons.append(radio_button)
                self.column_selection_layout.addWidget(radio_button)
                self.column_button_group.addButton(radio_button)

            self.log_message("Excel columns loaded. Please select the filename column.")
        except Exception as e:
            self.log_message(f"Error reading Excel file: {str(e)}")

    def read_excel(self):
        try:
            if self.excel_data is None:
                raise ValueError("Please load an Excel file first.")

            selected_button = self.column_button_group.checkedButton()
            if not selected_button:
                raise ValueError("Please select a filename column.")

            filename_column = selected_button.text()

            if filename_column not in self.excel_data.columns:
                raise ValueError(f"Column '{filename_column}' not found in the Excel file.")

            self.current_row = 0
            self.update_filename_display()
            self.log_message("Excel file read successfully.")
        except Exception as e:
            self.log_message(f"Error reading Excel file: {str(e)}")
            self.excel_data = None
            self.update_filename_display()

    def update_filename_display(self):
        current_filename = self.get_current_filename()
        if current_filename:
            self.filename_edit.setText(current_filename)
        else:
            self.filename_edit.setText("No Excel file loaded. Using timestamp.")

    def get_current_filename(self):
        if self.excel_data is not None and not self.excel_data.empty:
            selected_button = self.column_button_group.checkedButton()
            if selected_button:
                filename_column = selected_button.text()
                return self.excel_data.iloc[self.current_row][filename_column]
        return None

    def next_filename(self):
        if self.excel_data is not None and not self.excel_data.empty:
            self.current_row = (self.current_row + 1) % len(self.excel_data)
            self.update_filename_display()

    def previous_filename(self):
        if self.excel_data is not None and not self.excel_data.empty:
            self.current_row = (self.current_row - 1) % len(self.excel_data)
            self.update_filename_display()

    def create_large_button(self, text):
        button = QPushButton(text, self)
        button.setFont(QFont('Arial', 14, QFont.Bold))
        button.setFixedHeight(120)  # Make the button three times taller (assuming default height is 40)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return button

    def closeEvent(self, event):
        self.stop_microscope()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FieldDinoApp()
    window.show()
    sys.exit(app.exec_())