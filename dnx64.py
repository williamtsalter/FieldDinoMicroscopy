from typing import List, Tuple, Callable
import ctypes
import cv2

# CONSTANTS
VID_POINTERS: int = 5
VID_PARAMS: int = 4
METHOD_SIGNATURES: dict = {
    "Init": ([], ctypes.c_bool),
    "EnableMicroTouch": ([ctypes.c_bool], ctypes.c_bool),
    "FOVx": ([ctypes.c_int, ctypes.c_double], ctypes.c_double),
    "GetAETarget": ([ctypes.c_int], ctypes.c_long),
    "GetAMR": ([ctypes.c_int], ctypes.c_double),
    "GetAutoExposure": ([ctypes.c_int], ctypes.c_bool),
    "GetConfig": ([ctypes.c_int], ctypes.c_long),
    "GetDeviceId": ([ctypes.c_int], ctypes.c_wchar_p),
    "GetDeviceIDA": ([ctypes.c_int], ctypes.c_char_p),
    "GetExposureValue": ([ctypes.c_int], ctypes.c_long),
    "GetLensPosLimits": ([ctypes.c_long, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long)], ctypes.c_long),
    "GetVideoDeviceCount": ([], ctypes.c_int),
    "GetVideoDeviceIndex": ([], ctypes.c_long),
    "GetVideoDeviceName": ([ctypes.c_int], ctypes.c_wchar_p),
    "GetVideoProcAmp": ([ctypes.c_long], ctypes.c_long),
    "GetVideoProcAmpValueRange": ([ctypes.POINTER(ctypes.c_long) for _ in range(VID_POINTERS)], ctypes.c_long),
    "GetWiFiVideoCaps": ([ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short)], ctypes.c_bool),
    "SetAETarget": ([ctypes.c_int, ctypes.c_long], None),
    "SetAutoExposure": ([ctypes.c_int, ctypes.c_long], None),
    "SetAXILevel": ([ctypes.c_int, ctypes.c_long], None),
    "SetExposureValue": ([ctypes.c_int, ctypes.c_long], None),
    "SetEFLC": ([ctypes.c_int, ctypes.c_long, ctypes.c_long], None), # Not Working
    "SetFLCSwitch": ([ctypes.c_int, ctypes.c_long], None),
    "SetFLCLevel": ([ctypes.c_int, ctypes.c_long], None),
    "SetLEDState": ([ctypes.c_int, ctypes.c_long], None),
    "SetLensInitPos": ([ctypes.c_int], None),
    "SetLensPos": ([ctypes.c_int, ctypes.c_long], None),
    "SetVideoDeviceIndex": ([ctypes.c_int], None),
    "SetVideoProcAmp": ([ctypes.c_long], None),
    "SetEventCallback": ([ctypes.CFUNCTYPE(None)], None),
  }

class DNX64:
    
    def __init__(self, dll_path: str) -> None:
        """
        Initialize the DNX64 class.

        Parameters:
            dll_path (str): Path to the DNX64.dll library file.
        """
        self.dnx64 = ctypes.CDLL(dll_path)
        self.setup_methods()
        
    def setup_methods(self) -> None:
        """
        Set up the signatures for DNX64.dll methods using dictionary constant.
        """
        for method_name, (argtypes, restype) in METHOD_SIGNATURES.items():
            getattr(self.dnx64, method_name).argtypes = argtypes
            getattr(self.dnx64, method_name).restype = restype
    
    def Init(self) -> bool:
        """
        Initialize the control object.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return self.dnx64.Init()
        except OSError as e:
            if e.winerror == -529697949:
                print("DNX64: Error initializing the control object. Is the microscope connected?")
            else:
                # Handle other OSError cases
                print(f"An error occurred: {e}")
            return False


    def EnableMicroTouch(self, boolean: bool) -> bool:
        """
        REQUIRES DINO-LITE WITH MICROTOUCH BUTTON

        Enable or disable the MicroTouch feature.

        Parameters:
            boolean (bool): True to enable, False to disable.

        Returns:
            bool: True if successful, False otherwise.
        """

        return self.dnx64.EnableMicroTouch(boolean)
    
    def FOVx(self, device_index: int, mag: float) -> float:
        """
        Get the field of view, in micrometers, for the specified device and magnification.

        Parameters:
            device_index (int): Index of the device.
            mag (float): Magnification value.

        Returns:
            float: Field of view (FOV) in micrometers (um).
        """
        return self.dnx64.FOVx(device_index, mag)
    
    def GetAMR(self, device_index: int) -> float:
        """
        REQUIRES DINO-LITE WITH AMR FEATURE
        
        Get the Automatic Magnification Reading (AMR) for the specified device.
        
        Parameters:
            device_index (int): Index of the device.

        Returns:
            float: Automatic Magnification Reading (AMR).
        """
        return self.dnx64.GetAMR(device_index)
    
    def GetAutoExposure(self, device_index: int) -> bool:
        """
        Get the auto exposure value for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            int: Auto exposure value. 0 = 0FF, 1 = ON
        """
        return self.dnx64.GetAutoExposure(device_index)
    
    def GetConfig(self, device_index: int) -> int:
        """
        Get the configuration value for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            int: Device configuration in binary format.
        """

        return self.dnx64.GetConfig(device_index)
    
    def GetDeviceID(self, device_index: int) -> str:
        """
        Get the unique device ID for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            str: Device ID.
        """
        return self.dnx64.GetDeviceID(device_index)
    
    def GetDeviceIDA(self, device_index: int) -> str:
        """
        Get the alternate unique device ID for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            str: Alternate device ID.
        """
        return self.dnx64.GetDeviceIDA(device_index)
    
    def GetAETarget(self, device_index: int) -> int:
        """
        Get the Auto Exposure (AE) target value for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            int: AE target value.
        """
        return self.dnx64.GetAETarget(device_index)
    
    def GetExposureValue(self, device_index: int) -> int:
        """
        Get the exposure value for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            int: Exposure value.
        """
        return self.dnx64.GetExposureValue(device_index)
    
    def GetLensPosLimits(self, device_index: int) -> Tuple[int,int]:
        """
        REQUIRES DINO-LITE WITH EDOF FEATURE
        
        Get the lens position limits for the specified device.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            Tuple[int, int]: Upper and lower lens position limits.
        """
        upper_limit, lower_limit = ctypes.c_long(), ctypes.c_long()
        self.dnx64.GetLensPosLimits(device_index, upper_limit, lower_limit)
        return upper_limit.value, lower_limit.value
    
    def GetVideoDeviceCount(self) -> int:
        """
        Get the number of video devices.

        Returns:
            int: Number of video devices.
        """
        self.dnx64.Init()
        return self.dnx64.GetVideoDeviceCount()
    
    def GetVideoDeviceIndex(self) -> int:
        """
        Get the video device index.

        Returns:
            int: Video device index.
        """
        return self.dnx64.GetVideoDeviceIndex()
    
    def GetVideoDeviceName(self, device_index: int) -> str:
        """
        Get the name of the video device for the specified device index.

        Parameters:
            device_index (int): Index of the device.

        Returns:
            str: Name of the video device.
        """
        return self.dnx64.GetVideoDeviceName(device_index)
    
    def GetVideoProcAmp(self, prop_value_index: int) -> int:
        """
        Get the value of a video property.

        Parameters:
            ValueIndex (int): Value index of the video property.

        Returns:
            int: Video processing amplitude of indexed value
        
        """
        return self.dnx64.GetVideoProcAmp(prop_value_index)
    
    def GetVideoProcAmpValueRange(self, prop_value_index: int) -> Tuple[int, int, int, int, int]:
        """
        Get the min, max, stepping and default values for the specified video property.

        Parameters:
            value_index (int): Value index.

        Returns:
            Tuple[int, int, int, int, int]: index, min, max, step, and default
        """
        prop_value_index = ctypes.c_long(prop_value_index)
        params = [ctypes.c_long() for _ in range(VID_PARAMS)]
        self.dnx64.GetVideoProcAmpValueRange(prop_value_index, *params)
        min_val, max_val, stepping, default = [param.value for param in params]
        return prop_value_index.value, min_val, max_val, stepping, default
    
    def GetWiFiImage(self, filename: str) -> bool:
        """
        Retrieve the WiFi image.

        Parameters:
            filename (str): The filename to save the image as.

        Returns:
            bool: Success status (True/False).
        """
        # Convert Python string to ctypes byte array
        filename_bytes = filename.encode('utf-8')
        filename_array = (ctypes.c_byte * len(filename_bytes))(*filename_bytes)

        return self.dnx64.GetWiFiImage(filename_array)
    
    def GetWiFiVideoCaps(self) -> Tuple[int, List[Tuple[int, int]]]:
        """ 
        Retrieves the supported video resolutions for WiFi.

        Returns:
            Tuple[int, List[Tuple[int, int]]]:
                - int: Number of supported resolutions.
                - List[Tuple[int, int]]: List of formatted resolution.
        """ 
        # Fixed-size arrays for width and height.
        width_array = (ctypes.c_short * 5)()
        height_array = (ctypes.c_short * 5)()

        count = ctypes.c_int()

        success = self.dnx64.GetWiFiVideoCaps(ctypes.byref(count), width_array, height_array)

        if not success:
            raise Exception("Failed to retrieve WiFi video capabilities.\n")

        # Convert ctypes arrays to Python lists and then format them
        #resolutions = [f"{width_array[i]} x {height_array[i]}" for i in range(count.value)]
        resolutions = [(width_array[i], height_array[i]) for i in range(count.value)]
        #"GetVideoProcAmpValueRange": ([ctypes.POINTER(ctypes.c_long) for _ in range(VID_POINTERS)], ctypes.c_long),

        return count.value, resolutions
    
    def SetAETarget(self, device_index: int, ae_target: int) -> None:
        """
        Set the Auto Exposure (AE) target value for the specified device.

        Parameters:
            device_index (int): Index of the device.
            ae_target (int): AE target value. Acceptable Range: 16 to 20    
        """
        self.dnx64.SetAETarget(device_index, ae_target)    
    
    def SetAutoExposure(self, device_index:int, ae_state: int) -> None:
        """
        Set the auto exposure value for the specified device.

        Parameters:
            device_index (int): Index of the device.
            ae_state (int): Auto exposure value. Accepts 0 and 1.
        """
        self.dnx64.SetAutoExposure(device_index, ae_state)
        
    def SetAXILevel(self, device_index: int, axi_level: int) -> None:
        """
        REQUIRES DINO-LITE WITH AXI FEATURE
        
        Set the AXI level for the specified device.

        Parameters:
            device_index (int): Index of the device.
            axi_level (int): AXI level. Accepts 0 to 6.
        """
        self.dnx64.SetAXILevel(device_index, axi_level)

    def SetEventCallback(self, external_callback: Callable) -> None:
        """
        Set the callback function for the MicroTouch pressed event.

        Parameters:
            external_callback (Callable): The external callback function.
        """
        self.EventCallback = ctypes.CFUNCTYPE(None)
        self.callback_func = self.EventCallback(external_callback)
        self.dnx64.SetEventCallback(self.callback_func)
    
    def SetExposureValue(self, device_index: int, exposure_value: int) -> None:
        """
        Set the exposure value for the specified device.

        Parameters:
            device_index (int): Index of the device.
            exposure_value (int): Exposure value.
        """
        self.dnx64.SetExposureValue(device_index, exposure_value)
    
    def SetFLCSwitch(self, device_index: int, flc_quadrant: int) -> None:
        """
        REQUIRES DEVICE WITH FLC FEATURE
        
        Set the FLC switch for the specified device.

        Parameters:
            device_index (int): Index of the device.
            flc_quadrant (int): FLC quadrant.
        """
        self.dnx64.SetFLCSwitch(device_index, flc_quadrant)
    
    def SetFLCLevel(self, device_index: int, flc_level: int) -> None:
        """
        Set the FLC level for the specified device.

        Parameters:
            device_index (int): Index of the device.
            flc_level (int): FLC level. Accepts 1 to 6            
        """
        self.dnx64.SetFLCLevel(device_index, flc_level)
        
    def SetLEDState(self, device_index: int, led_state: int) -> None:
        """
        !!! Controllable only when the camera preview is established
        !!! Not applicable to AM211, AM2011, and Dino-Eye serie
        
        Set the LED state for the specified device.

        Parameters:
            device_index (int): Index of the device.
            led_state (int): LED state.
        """
        self.dnx64.SetLEDState(device_index, led_state)

    def SetLensInitPos(self, device_index: int) -> None:
        """
        !!! REQUIRES DEVICE WITH EDOF FEATURE !!!

        Set the lens initialization position for the specified device.

        Parameters:
            device_index (int): Index of the device.
        """
        self.dnx64.SetLensInitPos(device_index)
    
    def SetLensPos(self, device_index: int, lens_position: int) -> None:
        """
        !!! REQUIRES DEVICE WITH EDOF FEATURE !!!

        Set the lens position for the specified device.

        Parameters:
            device_index (int): Index of the device.
            lens_position (int): Lens position.
        """
        self.dnx64.SetLensPos(device_index, lens_position)
        
    def SetVideoDeviceIndex(self, set_device_index: int) -> None:
        """
        Set the video device index.

        Parameters:
            set_device_index (int): Video device index to set.
        """
        self.dnx64.SetVideoDeviceIndex(set_device_index)
        
    def SetVideoProcAmp(self, prop_value_index: int, new_value: int) -> None:
        """
        Set the value for the specified video property.

        Parameters:
            prop_value_index (int): Index of video property.
            new_value (int):  Updated value of video property.
        """
        self.dnx64.SetVideoProcAmp(prop_value_index, new_value)
        
    def SetWiFiVideoRes(self, width: int, height: int) -> bool:
        """
        Set the WiFi video resolution.

        Parameters:
            width (int): The desired video width.
            height (int): The desired video height.

        Returns:
            bool: Success status (True/False).
        """
        return self.dnx64.SetWiFiVideoRes(width, height)
    
    def SetEFLC(self, DeviceIndex: int, Quadrant: int, Value: int) -> None:
        """
        Sets the EFLC value based on the quadrant.

        Parameters:
            DeviceIndex (int): Index of the device.
            Quadrant (int): Quadrant number (1-4).
            Value (int): EFLC value to set.
        """
        self.dnx64.SetEFLC(DeviceIndex, Quadrant, Value)
        
        
