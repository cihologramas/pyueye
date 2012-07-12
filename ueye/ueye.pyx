# Copyright (c) 2010, Combustion Ingenieros Ltda.
# All rights reserved.
#       Redistribution and use in source and binary forms, with or without
#       modification, are permitted provided that the following conditions are
#       met:
#       
#       * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above
#         copyright notice, this list of conditions and the following disclaimer
#         in the documentation and/or other materials provided with the
#         distribution.
#       * Neither the name of the Combustión Ingenieros Ltda. nor the names of its
#         contributors may be used to endorse or promote products derived from
#         this software without specific prior written permission.
#       
#       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#       "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#       A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#       OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#       SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#       LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#       THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#       (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#       OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Ricardo Amezquita Orozco <ramezquitao@cihologramas.com>


# Functions definition
from stdlib cimport *
from python_cobject cimport *
import numpy as npy
cimport numpy as npy

def GetNumberOfCameras():
    '''Returns the number of connected cams
    
    Syntax:
    =======
        
    ncam=GetNumberOfCameras()
    
    Return Value:
    =============
    
    ncam: 
        Number of connected cams
    '''
    cdef INT ncam
    is_GetNumberOfCameras(&ncam)
    return ncam

#~ def GetCameraList(): #NotWorking
    #~ ncam=GetNumberOfCameras()
    #~ print ncam
    #~ if ncam<1: return None
    #~ cdef PUEYE_CAMERA_LIST pucl
    #~ #cdef UEYE_CAMERA_INFO tuci
    #~ #print tuci.dwCameraID
    #~ print sizeof(DWORD)+ ncam* sizeof (UEYE_CAMERA_INFO)
    #~ pucl = <UEYE_CAMERA_LIST *> malloc(sizeof (DWORD) + ncam* sizeof (UEYE_CAMERA_INFO))
    #~ #pucl = <PUEYE_CAMERA_LIST> malloc(10000)
    #~ print 1
    #~ is_GetCameraList(pucl)
    #~ print 2
    #~ return None

def GetDLLVersion():
    ''' Returns the ueye_api.so/dll mayor,minor,version numbers
    
    Syntax:
    =======
    
    mayor,minor,build=GetDLLVersion()
    
    Return Value:
    =============
    
    mayor:
        mayor version number
    minor:
        minor version number
    build:
        build version number
    '''
    cdef INT ver
    ver=is_GetDLLVersion()
    build=ver&0xFFFF
    minor=(ver>>16) & 0xFF
    mayor=(ver>>24) & 0xFF
    return mayor,minor,build

def bitspixel(colormode):
    '''Returns the bits per pixel corresponding a given colormode
    
    Syntax:
    =======
    
    bpp=bitspixel(colormode)
    
    Input Parameters:
    =================
    
    colormode:
        Colormode to check, all the CM_... constants are defined
    
    Return Value:
    =============
    
    bpp:
        Bits per pixel corresponding to the given colormode
    '''
    if colormode==IS_CM_MONO8 or colormode==IS_CM_BAYER_RG8:
        return 8
    elif colormode==IS_CM_MONO12        or colormode==IS_CM_MONO16        \
      or colormode==IS_CM_BAYER_RG12    or colormode==IS_CM_BAYER_RG16    \
      or colormode==IS_CM_BGR555_PACKED or colormode==IS_CM_BGR565_PACKED \
      or colormode==IS_CM_UYVY_PACKED   or colormode==IS_CM_CBYCRY_PACKED:
        return 16
    elif colormode==IS_CM_RGB8_PACKED or colormode==IS_CM_BGR8_PACKED:
        return 24
    elif colormode==IS_CM_RGBA8_PACKED or colormode==IS_CM_BGRA8_PACKED   \
      or colormode==IS_CM_RGBY8_PACKED or colormode==IS_CM_BGRY8_PACKED   \
      or colormode==IS_CM_RGB10V2_PACKED or colormode==IS_CM_BGR10V2_PACKED:
        return 32;
    else: return 8
    
npy.import_array() 

cdef class Cam:
    '''Class used to control a uEye camera

    Syntax:
    =======
    
    cam=Cam(cid)
    
    Input Parameters:
    =================
    
    cid:
        Camera id of the cam to be used. If cid is not given, or cid=0, the first
        available camera is used
    
    Return Value:
    =============
    
    cam:
        Instance to the Cam class assigned to the requested cam.
    '''
    cdef char *Img
    cdef public HIDS cid
    cdef public object SerNo,ID,Version,Date,Select,SensorID,strSensorName,nColorMode,nMaxWidth,nMaxHeight,bMasterGain, bRGain, bGGain, bBGain, bGlobShutter, \
                        bitspixel, colormode
    cdef public INT LineInc, 
    cdef public int ImgMemId
        
    def __init__(self,HIDS cid=0):
        #cdef HWND hWnd
        rv=is_InitCamera(&cid, NULL)
        #IS_SUCCESS
        self.CheckNoSuccess(rv)
        if rv==IS_STARTER_FW_UPLOAD_NEEDED:
            raise Exception("The camera's starter firmware is not compatible with the driver and needs to be updated.")
        self.cid=cid
        
        cdef CAMINFO cInfo
        rv =is_GetCameraInfo(self.cid, &cInfo)
        self.CheckNoSuccess(rv)
        
        self.SerNo=cInfo.SerNo
        self.ID=cInfo.ID
        self.Version=cInfo.Version
        self.Date=cInfo.Date
        self.Select=cInfo.Select
        #pInfo.Type,pInfo.Reserved
        
        cdef SENSORINFO sInfo
        rv=is_GetSensorInfo(self.cid, &sInfo)
        
        self.CheckNoSuccess(rv)
        
        self.SensorID=sInfo.SensorID            # e.g. IS_SENSOR_UI224X_C
        self.strSensorName=<char*>sInfo.strSensorName  # e.g. "UI-224X-C"
        self.nColorMode=sInfo.nColorMode        # e.g. IS_COLORMODE_BAYER
        self.nMaxWidth=sInfo.nMaxWidth          # e.g. 1280
        self.nMaxHeight=sInfo.nMaxHeight        # e.g. 1024
        self.bMasterGain=sInfo.bMasterGain      # e.g. TRUE
        self.bRGain=sInfo.bRGain                # e.g. TRUE
        self.bGGain=sInfo.bGGain                # e.g. TRUE
        self.bBGain=sInfo.bBGain                # e.g. TRUE
        self.bGlobShutter=sInfo.bGlobShutter    # e.g. TRUE
        
        cdef int pid
        
        # Check if the cam is color or bw
        if self.nColorMode == IS_COLORMODE_BAYER:
            colormode= IS_CM_BGR8_PACKED
        elif self.nColorMode == IS_COLORMODE_MONOCHROME:
            colormode= IS_CM_MONO8
        else:
            raise Exception("Colormode not supported")
        
        
        #Set colormode and assign image memory. The image memory assigment is done 
        #in SetColorMode
        self.Img=<char *>NULL
        self.SetColorMode (colormode)    
        
        
        
        rv=is_SetExternalTrigger (self.cid, IS_SET_TRIGGER_OFF)
        self.CheckNoSuccess(rv)
        
        rv=is_FreezeVideo (self.cid, IS_WAIT)
        self.CheckNoSuccess(rv)
        
        #Start with auto BL_Compensation off by default if possible
        
        rv=self.SetBlCompensation(IS_GET_BL_SUPPORTED_MODE, 0)

        if rv & IS_BL_COMPENSATION_OFFSET:
            self.SetBlCompensation(IS_BL_COMPENSATION_DISABLE, 0)

    def __dealloc__(self):
        
        rv=is_FreeImageMem (self.cid,self.Img, self.ImgMemId)
        self.CheckNoSuccess(rv)
    
        rv=is_ExitCamera (self.cid)
        self.CheckNoSuccess(rv)            
       
    
    def GrabImage(self):
        '''Grabs and reads an image from the camera and returns a numpy array
        
        
        Syntax:
        =======
    
        im=cam.GrabImage()
        
        Return Value:
        =============
    
        im:
            Numpy array containing the image data. The data from the driver buffer
            to the numpy array. The returned numpy array is modified each time 
            the method is called. 
            
        Note: The default colormode for color cameras is BGR, so the imshow in numpy
        is not showing the correct colors


        '''
        cdef npy.npy_intp dims3[3]
        
        
        rv= is_FreezeVideo (self.cid, IS_WAIT)
        self.CheckNoSuccess(rv)
        
        if self.colormode==IS_CM_RGB8_PACKED or self.colormode==IS_CM_BGR8_PACKED:
            dims3[0]=self.nMaxHeight
            dims3[1]=self.LineInc/3
            dims3[2]=3
            npy.Py_INCREF( npy.NPY_UINT8 )
            data = npy.PyArray_SimpleNewFromData(3, dims3, npy.NPY_UINT8, self.Img)
            if self.colormode==IS_CM_BGR8_PACKED:
                data=data[:,:,::-1]
        
        elif self.colormode==IS_CM_MONO8:
            dims3[0]=self.nMaxHeight
            dims3[1]=self.LineInc
            npy.Py_INCREF( npy.NPY_UINT8 )
            data = npy.PyArray_SimpleNewFromData(2, dims3, npy.NPY_UINT8, self.Img)
            
        else:
            raise Exception("ColorFormat not suported")
        return data
    
    def GetExposureRange(self):
        '''Returns the exposure range parameters
        
        Using GetExposureRange(), you can query the exposure values 
        (in milliseconds) available for the currently selected timing 
        (pixel clock, frame rate). The available time values are 
        comprised between min and max and can be set in increments 
        defined by the intervall parameter.

        Syntax:
        =======
        
        min,max,interval=cam.GetExposureRange()
        
        
        Return Value:
        =============
        
        min: 
            Minimum available exposure time
        max:
            Maximum available exposure time
        interval:
            Allowed increment
                
        '''

        cdef double min,max,interval
        rv= is_GetExposureRange (self.cid, &min, &max, &interval)
        self.CheckNoSuccess(rv)

        return (min,max,interval)
    
    def SetExposureTime(self,double expo):
        '''Set the exposure time
        
        Using SetExposureTime(), you can set the exposure time 
        (in milliseconds). Since this value depends on the sensor timing, 
        the exposure time actually used may slightly deviate from the 
        value set here. The actual exposure time is returned by the method. 
        In free-running mode (CaptureVideo()),  any modification of the exposure
        time will only become effective when the next image but one is captured. 
        In trigger mode (SetExternalTrigger()), the modification  will be 
        applied to the next image.

        Syntax:
        =======
        
        nexpo=cam.SetExposureTime(expo)
        
        Input Parameters:
        =================
        
        expo: 
            New desired exposure time.

            For expo=0.0, the exposure time is 1/frame rate.

            GET_EXPOSURE_TIME: Returns the current exposure.

            GET_DEFAULT_EXPOSURE: Returns the default exposure time.

            SET_ENABLE_AUTO_SHUTTER: Enables the auto exposure function.

        Return Value:
        =============
        
            nexpo
                The exposure time actually set
        '''
                
        cdef double nexpo
        rv= is_SetExposureTime(self.cid, expo, &nexpo)
        self.CheckNoSuccess(rv)
        return nexpo
    
    def GetFramesPerSecond(self):
        '''Return the frames per second
        
        In live capture mode started by CaptureVideo(), the GetFramesPerSecond() 
        function returns the number of frames actually captured per second.
               
        Syntax:
        =======
        dblFPS = cam.GetFramesPerSecond()
        
        Return Values:
        ==============
            dblFPS: 
                Returns the current frame rate.
        '''
        
        cdef double dblFPS
        rv=is_GetFramesPerSecond (self.cid, &dblFPS)
        self.CheckNoSuccess(rv)
        return dblFPS
        
    def GetFrameTimeRange(self):
        '''Returns the frame rate settings
        
        Using GetFrameTimeRange(), you can read out the frame rate settings 
        which are available for the current pixel clock setting. 
        The returned values indicate the minimum and maximum frame duration 
        in seconds. You can set the frame duration between min and max 
        in increments defined by the intervall parameter.
    
        Syntax:
        =======
        
        min,max,intervall= cam.GetFrameTimeRange()
           
        Return Values:
        ==============
      
        min: 
            Minimum available frame duration.

        max:
            Maximum available frame duration.

        intervall: 
           Increment you can use to change the frame duration.
        
        '''
        
        cdef double min,max,intervall
        rv=is_GetFrameTimeRange (self.cid, &min, &max, &intervall)
        self.CheckNoSuccess(rv)
        return (min,max,intervall)
    
    def SetFrameRate(self, double FPS):
        '''Set the Frame rate
        
        Using SetFrameRate(), you can set the sensor frame rate in 
        freerun mode (live mode). Since this value depends on the sensor
        timing, the exposure time actually used may slightly deviate from 
        the value set here. After you have called the function, the actual 
        frame rate is returned. If the frame rate is set too high, it might not 
        be possible to transfer every single frame. In this case, the effective 
        frame rate may vary from the set value.
       
        The use of the following functions will affect the frame rate:
        
        - SetPixelClock()
        - SetAOI() (if the image size is changed)
        - SetSubSampling()
        - SetBinning()

        Changes made to the window size or the read-out timing 
        (pixel clock frequency) also affect the defined frame rate. For this 
        reason, you need to call SetFrameRate() again after such changes.
       
        Newer driver versions sometimes allow an extended value range for 
        the frame rate setting. We recommend to query the value range 
        every time and set the frame rate explicitly. Changes to the frame 
        rate affect the value ranges of the exposure time. 
        
        After executing SetFrameRate(), calling the function SetExposureTime()
        is recommended in order to keep the defined camera settings.
           
        Syntax:
        =======
        
        newFPS=cam.SetFrameRate(FPS)
        
        Input Parameters:
        =================
    
        FPS: 
            Desired frame rate in frames per second (fps)
        
        Return Values:
            newFPS: 
                Returns the frame rate actually set.
                
                if FPS=GET_FRAMERATE: Returns the current frame rate.
                
                if PFS=GET_DEFAULT_FRAMERATE: Returns the default frame rate.

        '''   
        cdef double newFPS
        rv=is_SetFrameRate (self.cid, FPS, &newFPS)
        self.CheckNoSuccess(rv)
        return newFPS
    
    def GetPixelClockRange(self):
        '''Return the pixel clock range
        
        GetPixelClockRange() returns the adjustable pixel clock range.
        The pixel clock limit values can vary, depending on the camera 
        model and operating mode. 
   
        Syntax:
        =======
        
        pnMin, pnMax=cam.GetPixelClockRange()
        
        Return Values:
        ==============
        
        pnMin: Lower limit value.
        
        pnMax: Upper limit value.
        '''
        
        cdef INT pnMin,pnMax
        rv=is_GetPixelClockRange (self.cid, &pnMin, &pnMax)
        self.CheckNoSuccess(rv)
        return pnMin,pnMax
    
    def SetPixelClock(self,Clock):
        '''Configure the pixel clock
 
        SetPixelClock() sets the frequency used to read out image data 
        from the sensor (pixel clock frequency). Due to an excessive pixel 
        clock for USB cameras, images may get lost during the transfer. 
        If you change the pixel clock on-the-fly, the current image 
        capturing process will be aborted.
        
        Some sensors allow a higher pixel clock setting if binning or 
        subsampling has been activated. If you set a higher pixel clock 
        and then reduce the binning/subsampling factors again, 
        the driver will automatically select the highest possible pixel 
        clock for the new settings.
        
        Changes to the image geometry or pixel clock affect the value ranges 
        of the frame rate and exposure time. After executing SetPixelClock(), 
        calling the following functions is recommended in order to keep 
        the defined camera settings:
    
        - SetFrameRate()
        - SetExposureTime()
        - If you are using the uEye's flash function: SetFlashStrobe()
       
        Syntax:
        =======
        
        rv=cam.SetPixelClock(clock)
        
        Input Parameters:
        =================
        
        Clock: Pixel clock frequency to be set (in MHz)

        Return Values:
        ==============
        
        rv: 
            SUCCESS: Function executed successfully
        
            Current setting when clock = GET_PIXEL_CLOCK
    
            INVALID_MODE: Camera is in standby mode, function not allowed.

            INVALID_PARAMETER: The value for Clock is outside the pixel clock 
            range supported by the camera.
        
        '''
        
        rv= is_SetPixelClock (self.cid, Clock)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetAutoParameter(self,INT param, double pval1, double pval2):
        '''Set automatic parameters for the cam
        
        Using SetAutoParameter(), you can control the automatic gain, 
        exposure shutter, frame rate and white balance control values.
       
        - Control is only active as long as the camera is capturing images.
        - A manual change of the exposure time and gain settings disables 
          the auto functions.
        - When the auto shutter function is enabled, you cannot modify 
          the pixel clock frequency.
        - The auto frame rate function is only available when the auto 
          shutter control is on. 
          Auto frame rate and auto gain cannot be used simultaneously.
        - The auto gain function can only be used for cameras with master 
          gain control set. Auto white balance is only available for cameras
          with hardware RGB gain control set.
        - The sensor's internal auto functions are only supported by the 
          sensor of the UI-122x/522x camera models. 
        
        Syntax:
        =======
        
        pv1,pv2=cam.SetAutoParameters(param, pval1, pval2)
        
        Input Parameters:
        =================
            param: Configure auto control

                SET_ENABLE_AUTO_GAIN  Enables / disables the auto gain function. 
                Control parameter pval1 = 1 enables, 0 disables control

                GET_ENABLE_AUTO_GAIN Returns the current auto gain setting. return 
                value pv1 returns the current value

                SET_ENABLE_AUTO_SENSOR_GAIN Enables / disables the sensor's 
                internal auto gain function. Control parameter pval1 = 1 enables, 
                0 disables control

                GET_ENABLE_AUTO_SENSOR_GAIN Returns the current setting of the 
                sensor's internal auto gain function. Return value pv1 returns 
                the current value

                SET_ENABLE_AUTO_SHUTTER  Enables / disables the auto exposure 
                function. Control parameter pval1= 1 enables, 0 disables control

                GET_ENABLE_AUTO_SHUTTER Returns the current auto exposure setting. 
                Return value pv1 returns the current value

                SET_ENABLE_AUTO_SENSOR_SHUTTER Enables / disables the sensor's 
                internal auto exposure function. Control parameter pval1= 1 enables,
                0 disables control

                GET_ENABLE_AUTO_SENSOR_SHUTTER Returns the current setting of the 
                sensor's internal auto exposure function. Return value pv1 returns 
                the current value

                SET_ENABLE_AUTO_WHITEBALANCE Enables / disables the auto white 
                balance function. Control parameter pval1 = 1 enables, 
                0 disables control

                GET_ENABLE_AUTO_WHITEBALANCE Returns the current auto white balance 
                setting. Return value pv1 returns the current value

                SET_ENABLE_AUTO_FRAMERATE Enables / disables the auto frame rate 
                function. Control parameter pval1 = 1 enables, 0 disables control

                GET_ENABLE_AUTO_FRAMERATE Returns the current auto frame rate 
                setting. Return value pv1 returns the current value

                SET_ENABLE_AUTO_SENSOR_FRAMERATE Enables / disables the sensor's 
                internal auto frame rate function. Control parameter pval1 = 1 
                enables, 0 disables control

                GET_ENABLE_AUTO_SENSOR_FRAMERATE Returns the current setting of 
                the sensor's internal auto frame rate function. Return value pv1
                returns the current value

                SET_AUTO_REFERENCE Sets the setpoint value for auto gain / auto 
                shutter. Control parameter pval1 defines the setpoint value 
                (average image brightness). Independent of pixel bit depth the 
                setpoint range is:

                    0 = black

                    128 = 50% grey (default)
                    
                    255 = white

                GET_AUTO_REFERENCE Returns the setpoint value for auto gain/auto 
                shutter. Return value pv1 returns the current value

                SET_AUTO_GAIN_MAX Sets the upper limit for auto gain. Control 
                parameter pval1 set the valid value for gain (0...100)

                GET_AUTO_GAIN_MAX Returns the upper limit for auto gain. Return value
                pv1 returns the current value

                SET_AUTO_SHUTTER_MAX Sets the upper limit for auto exposure. 
                Control parameter pval1 defines the valid exposure value 
                (0 sets the value continuously to max. exposure)

                GET_AUTO_SHUTTER_MAX Returns the upper limit for auto exposure. 
                Return value pv1 returns the current value

                SET_AUTO_SPEED Sets the speed value for auto gain / exposure. 
                Control parameter pval1 defines the control speed (0...100)

                GET_AUTO_SPEED Returns the speed value for auto gain / exposure. 
                Return value pv1 returns the current value

                SET_AUTO_HYSTERESIS Sets the hysteresis for auto gain / exposure. 
                Control parameter pval1 defines the hysteresis value (default: 2)

                GET_AUTO_HYSTERESIS Returns the hysteresis for auto gain / exposure.
                Return value pv1 returns the current value

                GET_AUTO_HYSTERESIS_RANGE Returns range for the hysteresis value. 
                Return value pv1 returns the minimum value, and pv2 returns the 
                maximum value

                SET_AUTO_SKIPFRAMES Sets the number of frames to be skipped for 
                auto gain / auto exposure. Control parameter pval1 defines the 
                number of frames to be skipped (default: 4)

                GET_AUTO_SKIPFRAMES Returns the number of frames to be skipped 
                for auto gain / auto exposure. Return value pv1 returns the 
                current value

                GET_AUTO_SKIPFRAMES_RANGE Returns range for the number of frames 
                to be skipped. Return value pv1 returns the minimum value, and pv2
                returns the maximum value

                SET_AUTO_BRIGHTNESS_ONCE Enables / disables automatic disabling 
                of auto gain / auto exposure. Control parameter pval1 = 1 enables,
                0 disables control

                GET_AUTO_BRIGHTNESS_ONCE Returns the automatic disable status of 
                auto gain / auto exposure. Return value pv1 returns the current
                value


                SET_AUTO_WB_OFFSET Sets the offset value for the red and blue 
                channels. Control parameter pval1 defines the red level offset 
                (-50...50) and pval2 defines the blue level offset (-50...50)

                GET_AUTO_WB_OFFSET Returns the offset value for the red and blue 
                channels. Return value pv1 returns the red level offset (-50...50)
                and pv2 returns the blue level offset (-50...50)

                SET_AUTO_WB_GAIN_RANGE Sets the gain limits for the auto white 
                balance function. Control parameter pval1 sets the minimum value
                and pval2 sets the maximum value
                
                GET_AUTO_WB_GAIN_RANGE Returns the gain limits for the auto white 
                balance function. Return value pv1 returns the minimum value and
                pv2 returns the maximum value

                SET_AUTO_WB_SPEED Sets the speed value for the auto white balance.
                Control parameter pval1 defines the control speed (0...100)
                
                GET_AUTO_WB_SPEED Returns the speed value for the auto white 
                balance. Return value pv1 returns the current value

                SET_AUTO_WB_HYSTERESIS Sets the hysteresis for auto white balance. 
                Control parameter pval1 defines the hysteresis value (default: 2)

                GET_AUTO_WB_HYSTERESIS Returns the hysteresis for auto white 
                balance. Return value pv1 returns the current value

                GET_AUTO_WB_HYSTERESIS_RANGE Returns range for the hysteresis 
                value. Return value pv1 returns the minimum value and pv2 
                returns the maximum value

                SET_AUTO_WB_SKIPFRAMES Sets the number of frames to be skipped 
                for auto white balance. Control parameter pval1 defines the number 
                of frames to be skipped (default: 4)

                GET_AUTO_WB_SKIPFRAMES Returns the number of frames to be skipped 
                for auto white balance. Return value pv1 returns the current value

                GET_AUTO_WB_SKIPFRAMES_RANGE Returns range for the number of frames
                to be skipped. Return value pv1 returns the minimum value and pv2
                returns the maximum value

                SET_AUTO_WB_ONCE Enables / disables automatic disabling of auto 
                white balance. Control parameter pval1 = 1 enables, 0 disables 
                control

                GET_AUTO_WB_ONCE Returns the automatic disable status of auto 
                white balance. Return value pv1 returns the current value
                
                Pre-defined values for auto gain/auto exposure

                For parameters pval1 and pval, NULL must be passed.

                DEFAULT_AUTO_BRIGHT_REFERENCE Default setpoint value for auto gain /  exposure
                
                MIN_AUTO_BRIGHT_REFERENCE Minimum setpoint value for auto gain /  exposure

                MAX_AUTO_BRIGHT_REFERENCE Maximum setpoint value for auto gain /  exposure

                DEFAULT_AUTO_SPEED Default value for auto speed

                MAX_AUTO_SPEED Maximum value for auto speed
                
                Pre-defined values for auto white balance

                For parameters pval1 and pval, NULL must be passed.

                DEFAULT_WB_OFFSET Default value for auto white balance offset

                MIN_WB_OFFSET Minimum value for auto white balance offset

                MAX_WB_OFFSET Maximum value for auto white balance offset

                DEFAULT_AUTO_WB_SPEED Default value for auto white balance speed

                MIN_AUTO_WB_SPEED Minimum value for auto white balance speed

                MAX_AUTO_WB_SPEED Maximum value for auto white balance speed
        
            pval1: 
                Control parameter, can have a variable value depending on 
                the corresponding auto function. See table above.

            pval2:
                Control parameter, can have a variable value depending on 
                the corresponding auto function. See table above.
        
        Return Values:
        ==============
            pv1:
                Return value, can have a variable value depending on 
                the corresponding auto function. See table above.
            
            pv2:
                Return value, can have a variable value depending on 
                the corresponding auto function. See table above.

        '''
    
        rv= is_SetAutoParameter (self.cid, param, &pval1, &pval2)
        self.CheckNoSuccess(rv)
        return pval1,pval2
    
    def SetBlCompensation(self, INT nEnable, INT offset):
        '''Set black level compensation

        SetBlCompensation() enables the black level correction function 
        which might improve the image quality under certain circumstances. 
        By default, the sensor adjusts the black level value for each pixel 
        automatically. If the environment is very bright, it can be necessary 
        to adjust the black level manually.
    
        Syntax:
        =======
        rv=cam.SetBlCompensation(nEnable, offset)
        
        Input Parameters:
        =================
        
        nEnable:
    
            BL_COMPENSATION_DISABLE: Disables automatic black level correction. 
            The offset value is used as black level instead. This mode is only 
            supported by sensors of the UI-154x/554x series.

            BL_COMPENSATION_ENABLE: Enables automatic black level correction. 
            The offset value is added to the automatic black level value.

            GET_BL_COMPENSATION: Returns the current mode.

            GET_BL_OFFSET: Returns the currently set value for offset.

            GET_BL_DEFAULT_MODE: Returns the default mode.

            GET_BL_DEFAULT_OFFSET: Returns the default value for offset.

            GET_BL_SUPPORTED_MODE: Returns the supported modes.

            Possible values:
    
                BL_COMPENSATION_ENABLE:
                The sensor supports automatic black level correction.

                BL_COMPENSATION_OFFSET:
                For the sensor used, it is also possible to set the offset manual.

        IGNORE_PARAMETER: The nEnable parameter is ignored.

        offset: 
            Contains the offset value used for compensation. Valid values 
            are between 0 and 255.

            IGNORE_PARAMETER: The offset parameter is ignored.

            
        Return Values:
        ==============
        
        rv:
            SUCCESS: Function executed successfully

            Supported modes when used together with GET_BL_SUPPORTED_MODE
    
            Current mode when used together with GET_BL_COMPENSATION
    
            Current offset when used together with GET_BL_OFFSET
        '''
        
        rv=is_SetBlCompensation (self.cid, nEnable, offset, 0)
        self.CheckNoSuccess(rv)
        return rv
    
    def SetGainBoost(self, mode):
        ''' Set the gain boost
        
        In some cameras, SetGainBoost() enables an additional analogue 
        hardware gain boost feature on the sensor. 
        
        Syntax:
        =======    
        
        rv=cam.SetGainBoost(mode)
        
        Input Parameters:
        =================
        
        mode:
            GET_GAINBOOST: Returns the current state of the gain boost function.

            SET_GAINBOOST_ON: Enables the gain boost function.

            SET_GAINBOOST_OFF: Disables the gain boost function.

            GET_SUPPORTED_GAINBOOST: Indicates whether the camera supports a 
            gain boost feature or not.
            
        Return Values:
        ==============
            
            SUCCESS: Function executed successfully
        
            Current setting when used together with GET_GAINBOOST
            Returns 0 if the camera does not support a gain boost feature.

            Current setting when used together with GET_SUPPORTED_GAINBOOST
            Returns SET_GAINBOOST_ON if the function is supported, otherwise 
            it returns SET_GAINBOOST_OFF.
        
        '''
        
        rv= is_SetGainBoost (self.cid, mode)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetGamma(self, INT nGamma):
        ''' Set gamma correction
        
        SetGamma() sets the value for digital gamma correction 
        (brighten dark image portions by applying a non-linear characteristic (LUT)). 
        Valid values are in the range between 0.01 and 10.
        
        Syntax:
        =======
        
        rv=cam.SetGamma(nGamma)
        
        Input Parameters:
        =================
        
        nGamma: 
            Gamma value to be set, multiplied by 100 (Range: 1…1000. 
            Default = 100, corresponds to a gamma value of 1.0)

            GET_GAMMA: Returns the current setting.
        
        Return Value:
        =============
            SUCCESS: Function executed successfully
        
            Current setting when used together with GET_GAMMA        
        '''
        
        rv=is_SetGamma (self.cid, nGamma)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetGlobalShutter(self, INT mode):
        '''Set global shutter        
        
        SetGlobalShutter() enables the Global Start shutter function on some sensors.
        
        ** The Global Start shutter function is only supported in trigger mode 
        (see also SetExternalTrigger())**.
        
        Syntax:
        =======
        
        rv=cam.SetGlobalShutter(mode)
        
        Input Parameters:
        =================
        
        mode:
            GET_GLOBAL_SHUTTER: Returns the current mode or NOT_SUPPORTED 
            if the camera does not support this function.

            SET_GLOBAL_SHUTTER_ON: Enables Global Start shutter mode.

            SET_GLOBAL_SHUTTER_OFF: Disables Global Start shutter mode.

            GET_SUPPORTED_GLOBAL_SHUTTER: Indicates whether the connected camera 
            supports the Global Start shutter or not.
            
        Return Values:
        ==============
        rv:
            SUCCESS: Function executed successfully

            When used together with GET_SUPPORTED_GLOBAL_SHUTTER
            Returns SET_GLOBAL_SHUTTER_ON if this function is supported. 
            Otherwise, it returns SET_GLOBAL_SHUTTER_OFF.
        
        '''
        
        rv=is_SetGlobalShutter (self, mode)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetHardwareGain (self, INT nMaster, INT nRed, INT nGreen, INT nBlue):
        ''' Set hardware gain
        
        SetHardwareGain() controls the sensor gain channels. These can be set 
        between 0% and 100% independently of of each other. The actual gain factor 
        obtained for the value 100% depends on the sensor.

        You can use the GetSensorInfo() function to query the available gain controls.
        Depending on the time when the gain settings are changed, these changes 
        might only become effective when the next image is captured.
        
        ** Enabling hardware gain increases not only the image brightness, 
        but also the image noise. We recommend to use gain values 
        below 50 for normal operation. 
        
        The default setting values for the red, green and blue channel gain factors 
        depend on the colour correction matrix that has been set. 
        If you select a different colour correction matrix, the returned default values 
        might change (see also SetColorCorrection()).**
        
        Syntax:
        =======
        
        rv=cam.SetHardwareGain(nMaster, nRed, nGreen, nBlue)
        
        Input Parameters:
        =================
        
        nMaster: 
            Sets the overall gain factor (0...100).
            
            IGNORE_PARAMETER: The master gain factor will not be changed.
            
            GET_MASTER_GAIN: Returns the master gain factor.
            
            GET_RED_GAIN: Returns the red channel gain factor.
            
            GET_GREEN_GAIN: Returns the green channel gain factor.
            
            GET_BLUE_GAIN: Returns the blue channel gain factor.
            
            GET_DEFAULT_MASTER: Returns the default master gain factor.
            
            GET_DEFAULT_RED: Returns the default red channel gain factor.
            
            GET_DEFAULT_GREEN: Returns the default green channel gain factor.
            
            GET_DEFAULT_BLUE: Returns the default blue channel gain factor.
            
            SET_ENABLE_AUTO_GAIN: Enables the auto gain functionality 
            (see also SetAutoParameter()). You can disable the auto gain functionality
            by setting a value for nMaster.

        nRed: 
            Sets the red channel gain factor (0...100).
        
            IGNORE_PARAMETER: The channel gain factor will not be changed.
        
        nGreen: 
            Sets the green channel gain factor (0...100).
        
            IGNORE_PARAMETER: The green channel gain factor will not be changed.

        nBlue: 
            Sets the blue channel gain factor (0...100).
            IGNORE_PARAMETER: The blue channel gain factor will not be changed.
        
        Return Values:
        ==============
        
        rv:
            SUCCESS: Function executed successfully
        
            Current setting when used together with GET_MASTER_GAIN, GET_RED_GAIN,
            GET_GREEN_GAIN, GET_BLUE_GAIN
    
            INVALID_MODE: Camera is in standby mode, function not allowed.      
        '''
        
        rv=is_SetHardwareGain (self.cid, nMaster, nRed, nGreen, nBlue)
        self.CheckNoSuccess(rv)
        return rv
    
    def SetHardwareGamma (self, INT nMode):
        ''' Set the hardware gamma value
                
        SetHardwareGamma() enables the hardware gamma control feature of the camera.

        ** The SetHardwareGamma() function is only supported by cameras of 
        the GigE uEye series.**
        
        Syntax:
        =======
        
        rv=cam.SetHardwareGamma(nMode)
        
        Input Parameters:
        =================
        
        nMode:
            GET_HW_SUPPORTED_GAMMA: Indicates whether the camera supports hardware 
            gamma control or not.

            SET_HW_GAMMA_ON: Enables the gamma control feature.

            SET_HW_GAMMA_OFF: Disables gamma control.

            GET_HW_GAMMA: Returns the current state of gamma control.
        
        Return Values:
        ==============
        
        rv:
            SUCCESS: Function executed successfully
            
            Current setting when used together with GET_HW_GAMMA
            
            When used together with GET_HW_SUPPORTED_GAMMA: 
                
                SET_HW_GAMMA_ON: The camera supports gamma control.
                
                SET_HW_GAMMA_OFF: The camera does not support gamma control.
            
        '''
        
        rv=is_SetHardwareGamma (self.cid, nMode)
        self.CheckNoSuccess(rv)
        return rv
    
    def SetHWGainFactor (self, INT nMode, INT nFactor):
        ''' Set the hardware gain factor
                
        SetHWGainFactor() uses gain factors to control sensor gain channels. 
        These channels can be set independently of each other. 
        The SetHardwareGain() does not use factors for setting the gain channels, 
        but standardised values between 0 and 100. The actual gain factor is 
        sensor-dependent and can be found in the manual.

        You can use the is_GetSensorInfo() function to query the available gain controls.
        Depending on the time when the gain settings are changed, these changes might 
        only become effective when the next image is captured.
        
        Syntax:
        =======
        
        rv=cam.SetHWGainFactor(nMode, nFactor)
        
        Input Parameters:
        =================
        
        nmode:
            - GET_MASTER_GAIN_FACTOR: Returns the master gain factor.
            - GET_RED_GAIN_FACTOR: Returns the red channel gain factor.
            - GET_GREEN_GAIN_FACTOR: Returns the green channel gain factor.
            - GET_BLUE_GAIN_FACTOR: Returns the blue channel gain factor.
            - SET_MASTER_GAIN_FACTOR: Sets the master gain factor.
            - SET_RED_GAIN_FACTOR: Sets the red channel gain factor.
            - SET_GREEN_GAIN_FACTOR: Sets the green channel gain factor.
            - SET_BLUE_GAIN_FACTOR: Sets the blue channel gain factor.
            - GET_DEFAULT_MASTER_GAIN_FACTOR: Returns the default master gain factor.
            - GET_DEFAULT_RED_GAIN_FACTOR: Returns the default red channel gain factor.
            - GET_DEFAULT_GREEN_GAIN_FACTOR: Returns the default green channel gain factor.
            - GET_DEFAULT_BLUE_GAIN_FACTOR: Returns the default blue channel gain factor.
            - INQUIRE_MASTER_GAIN_FACTOR: Converts the index value for the master gain factor.
            - INQUIRE_RED_GAIN_FACTOR: Converts the index value for the red channel gain factor.
            - INQUIRE_GREEN_GAIN_FACTOR: Converts the index value for the green channel gain factor.
            - INQUIRE_BLUE_GAIN_FACTOR: Converts the index value for the blue channel gain factor.
        
        nFactor: Gain value (100 = gain factor 1, i. e. no effect)
        
            For converting a gain value from the SetHardwareGain() function, 
            you can set the nMode parameter to one of the INQUIRE_x_FACTOR values. 
            In this case, the value range for nFactor is between 0 and 100.
            To set the gain using SET_..._GAIN_FACTOR, you must set the nFactor 
            parameter to an integer value in the range from 100 to the maximum value. 
            By calling INQUIRE_x_FACTOR and specifying the value 100 for nFactor, 
            you can query the maximum value. A gain value of 100 means no gain, 
            a gain value of 200 means gain to the double level (factor 2), etc.
            
        Return Values:
        ==============
        rv:
            - SUCCESS: Function executed successfully
            - NO_SUCCESS: General error message
            - Current setting when used together with GET_MASTER_GAIN_FACTOR,
              GET_RED_GAIN_FACTOR, GET_GREEN_GAIN_FACTOR, GET_BLUE_GAIN_FACTOR
            - Defined setting when used together with SET_MASTER_GAIN_FACTOR,
              SET_RED_GAIN_FACTOR, SET_GREEN_GAIN_FACTOR, SET_BLUE_GAIN_FACTOR.
            - Default setting when used together with GET_DEFAULT_MASTER_GAIN_FACTOR,
              GET_DEFAULT_RED_GAIN_FACTOR, GET_DEFAULT_GREEN_GAIN_FACTOR, 
              GET_DEFAULT_BLUE_GAIN_FACTOR.
            - When used together with INQUIRE_MASTER_GAIN_FACTOR, INQUIRE_RED_GAIN_FACTOR,
              INQUIRE_GREEN_GAIN_FACTOR, INQUIRE_BLUE_GAIN_FACTOR Converted gain 
              index 
        
        '''
        
        rv=is_SetHWGainFactor (self.cid, nMode, nFactor)
        self.CheckNoSuccess(rv)
        return rv
        
    def ResetToDefault (self):
        ''' Reset parameters to default
        
        ResetToDefault() resets all parameters to the camera-specific defaults 
        as specified by the driver. By default, the camera uses full resolution, 
        a medium speed and colour level gain values adapted to daylight exposure. 
        All optional features are disabled.
        
        Syntax:
        =======
        rv=cam.ResetToDefault()
        
        Return Values:
        ==============
        
        rv:
            SUCCESS: Function executed successfully
        '''
        
        rv=is_ResetToDefault (self.cid)
        self.CheckNoSuccess(rv)
    
    def SetAOI (self, INT type, INT pXPos, INT pYPos, INT pWidth, INT pHeight):
        ''' Set the area of interest
        
        SetAOI() can be used to set the size and position of an area of 
        interest (AOI) within an image. The following AOIs can be defined:
        
        - Image AOI – display of an image portion
        - Auto Brightness AOI – reference area of interest for automatic 
          brightness control
        - Auto Whitebalance AOI – reference area of interest of automatic 
          white balance control
        
        ** By default, the window size for auto AOIs is always maximum, 
        i.e. it corresponds to the current image AOI.
        After a change to the image geometry (by resetting an image AOI, 
        by binning or sub-sampling), the auto AOIs will always be reset 
        to the image AOI value (i.e. to maximum size). This means that it 
        might be necessary to set the AOIs for the auto features again manually.
        
        Changes to the image geometry or pixel clock affect the value ranges 
        of the frame rate and exposure time. After executing SetAOI(), 
        calling the following functions is recommended in order to keep 
        the defined camera settings:
        
        - SetFrameRate()
        - SetExposureTime()
        - If you are using the uEye's flash function: SetFlashStrobe()**
        
        Syntax:
        =======
        XPos, YPos, Width, Height= cam.SetAOI (type, pXPos, pYPos, pWidth, pHeight)
        
        Input Parameters:
        =================
        

        The pXPos and pYPos parameters represent an offset with respect 
        to the upper left image corner. The cut window is copied to the 
        start position in the memory. If you want the image to be copied 
        to the same offset within the memory, you can link the new position 
        with a logical OR to the SET_IMAGEPOS_X_ABS and SET_IMAGEPOS_Y_ABS 
        parameters.
    
        type:
            - SET_IMAGE_AOI: Sets an image AOI.
            - GET_IMAGE_AOI: Returns the current image AOI.
            - SET_AUTO_BRIGHT_AOI: Sets average AOI values for auto gain and auto shutter.
            - GET_AUTO_BRIGHT_AOI: Returns the current auto brightness AOI.
            - SET_AUTO_WB_AOI: Sets an auto white balance AOI.
            - GET_AUTO_WB_AOI: Returns the current auto white balance AOI.
        
        XPos: 
            Horizontal position of the AOI
            0...XPosMax| SET_IMAGEPOS_X_ABS: Applies the absolute position to 
            the memory as well.
        
        pYPos: 
            Vertical position of the AOI
            0...YPosMax| SET_IMAGEPOS_Y_ABS: Applies the absolute position to 
            the memory as well.
        
        pWidth: 
            Width of the AOI
                
        pHeight: 
            Height of the AOI
        
        Return Values:
        ==============
        
        Returns the actual XPos, YPos, Width, Height, or the cofigured values when 
        type is GET...
        '''
        
        rv= is_SetAOI (self.cid, type, &pXPos, &pYPos, &pWidth, &pHeight)
        self.CheckNoSuccess(rv)
        
        return pXPos, pYPos, pWidth, pHeight
        
    def SetImagePos (self, INT x, INT y):
        '''Set the image position
                
        SetImagePos() determines the position of an area of interest (AOI) 
        in the display window. When used together with the is_SetAOI() 
        function, you can cut out an area of interest of the full video image.

        To avoid a positional mismatch between the display area and the image area, 
        make sure to call the functions in the correct order. Starting from the 
        original image, it is mandatory to keep to the following order:
        
        - SetAOI()
        - SetImagePos()
        
        ** With SetAOI(), you can set the position and size of an area of interest 
        using a single function call.
        
        Changes to the image geometry or pixel clock affect the value ranges of 
        the frame rate and exposure time. After executing SetBinning(), 
        calling the following functions is recommended in order to keep 
        the defined camera settings:

        - SetFrameRate()
        - SetExposureTime()
        - If you are using the uEye's flash function: SetFlashStrobe()**
        
        Syntax:
        =======
        
        rv= cam.SetImagePos(x,y)
        
        Input Parameters:
        =================
        
        The x and y parameters represent an offset with respect to the upper left 
        image corner. The cut window is copied to the start position in the memory.
        If you want the image to be copied to the same offset within the memory, 
        you can link the new position with a logical OR to the SET_IMAGE_POS_X_ABS 
        and SET_IMAGE_POS_Y_ABS parameters.
        
        x:
            - 0...xMax: Sets the horizontal position
            - 0...xMax | IS_SET_IMAGE_POS_X_ABS: Applies the absolute position 
              to the memory as well.
            - GET_IMAGE_POS_X: Returns the current x position.
            - GET_IMAGE_POS_X_MIN: Returns the minimum value for the horizontal 
              AOI position.
            - GET_IMAGE_POS_X_MAX: Returns the maximum value for the horizontal 
              AOI position.
            - GET_IMAGE_POS_X_INC: Returns the increment for the horizontal AOI 
              position.
            - GET_IMAGE_POS_X_ABS: Returns the absolute horizontal position in 
              the memory.
            - GET_IMAGE_POS_Y: Returns the current Y position.
            - GET_IMAGE_POS_Y_MIN: Returns the minimum value for the vertical 
              AOI position.
            - GET_IMAGE_POS_Y_MAX: Returns the maximum value for the vertical 
              AOI position.
            - GET_IMAGE_POS_Y_INC: Returns the increment for the vertical AOI 
              position.
            - GET_IMAGE_POS_Y_ABS: Returns the absolute vertical position in the 
              memory.
        
        y:
            - 0...yMax: Sets the vertical position
            - 0...yMax| IS_SET_IMAGE_POS_Y_ABS: Applies the absolute position to 
              the memory as well.
            - 0: When returning settings via parameter x (s. above)
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - INVALID_PARAMETER: Parameters x or y are invalid  (x, y < 0)
            - Current setting when used together with GET_IMAGE_POS parameters
            - INVALID_MODE: Camera is in standby mode, function not allowed.
        
        '''
        
        rv= is_SetImagePos (self.cid, x, y)
        
        self.CheckNoSuccess(rv)
    
    def SetBinning (self, INT mode):
        ''' Set or get the binning mode   
        
        Using SetBinning(), you can enable the binning mode both in horizontal 
        and in vertical direction. This way, the image size in the binning 
        direction can be reduced without scaling down the area of interest. 
        Depending on the sensor used, the sensitivity or the frame rate can be 
        increased while binning is enabled.

        To enable horizontal and vertical binning at the same time, you can 
        link the horizontal and vertical binning parameters by a logical OR.

        The adjustable binning factors of each sensor are listed in the manual.
        
        ** Some sensors allow a higher pixel clock setting if binning or 
        subsampling has been activated. If you set a higher pixel clock 
        and then reduce the binning/subsampling factors again, the driver will 
        automatically select the highest possible pixel clock for the new settings.
        
        Changes to the image geometry or pixel clock affect the value ranges of 
        the frame rate and exposure time. After executing SetBinning(), 
        calling the following functions is recommended in order to keep 
        the defined camera settings:

        - SetFrameRate()
        - SetExposureTime()
        - If you are using the uEye's flash function: SetFlashStrobe()
        
        Syntax:
        =======
        rv=cam.SetBinning(mode)
        
        Input Parameters:
        =================
        mode:
            - BINNING_DISABLE: Disables binning.
            - BINNING_2X_VERTICAL: Enables vertical binning with factor 2.
            - BINNING_3X_VERTICAL: Enables vertical binning with factor 3.
            - BINNING_4X_VERTICAL: Enables vertical binning with factor 4.
            - BINNING_6X_VERTICAL: Enables vertical binning with factor 6.
            - BINNING_2X_HORIZONTAL: Enables horizontal binning with factor 2.
            - BINNING_3X_HORIZONTAL: Enables horizontal binning with factor 3.
            - BINNING_4X_HORIZONTAL: Enables horizontal binning with factor 4.
            - BINNING_6X_HORIZONTAL: Enables horizontal binning with factor 6.
            - GET_BINNING: Returns the current setting.
            - GET_BINNING_FACTOR_VERTICAL: Returns the vertical binning factor.
            - GET_BINNING_FACTOR_HORIZONTAL: Returns the horizontal binning factor.
            - GET_SUPPORTED_BINNING: Returns the supported binning modes.
            - GET_BINNING_TYPE: Indicates whether the camera uses colour-proof binning 
              (IS_BINNING_COLOR) or not (IS_BINNING_MONO).
            
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - Current setting when used together with GET_BINNING, 
              GET_BINNING_FACTOR_VERTICAL, GET_BINNING_FACTOR_HORIZONTAL
            - When used with GET_BINNING_TYPE returns BINNING_COLOR if the camera 
              uses colour-proof subsampling, otherwise BINNING_MONO is returned.
            - When used with GET_SUPPORTED_BINNING, returns the supported 
              subsampling modes linked by logical ORs.
        '''
        
        rv=is_SetBinning (self.cid, mode)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetRopEffect (self, INT effect, INT param):
        '''Set or get ROP effect

        SetRopEffect() enables functions for real-time image geometry 
        modification (Rop = raster operation).
        
        Syntax:
        =======
        
        rv=cam.SetRopEffect(effect, param)
        
        Input Parameters:
        =================
        
        effect:
            - SET_ROP_MIRROR_UPDOWN: Mirrors the image along the horizontal axis.
            - SET_ROP_MIRROR_LEFTRIGHT: Mirrors the image along the vertical axis.
              Depending on the sensor, this operation is performed in the camera 
              or in the PC software.
            - GET_ROP_EFFECT: Returns the current settings.

        param: 
            Turns the Rop effect on / off. 0 = Turn off , 1 = Turn on

        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - NO_SUCCESS: General error message
            - Current setting when used together with SET_ROP_EFFECT
            - INVALID_MODE: Camera is in standby mode, function not allowed.
        
        '''
        
        rv= is_SetRopEffect (self.cid, effect,param, 0)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetSubSampling (self, INT mode):
        '''Set or get subsampling
        
        Using SetSubSampling(), you can enable sub-sampling mode both in horizontal 
        and in vertical directions. This allows you to reduce the image size 
        in the sub-sampling direction without scaling down the area of interest. 
        In order to simultaneously enable horizontal and vertical sub-sampling, 
        the horizontal and vertical sub-sampling parameters can by linked 
        by a logical OR. Some monochrome sensors are limited by their design 
        to mere colour sub-sampling. In case of fine image structures, 
        this can result in slight artifacts. The adjustable sub-sampling 
        factors of each sensor are listed in the manual.
        ** Some sensors allow a higher pixel clock setting if binning or 
        subsampling has been activated. If you set a higher pixel clock 
        and then reduce the binning/subsampling factors again, 
        the driver will automatically select the highest possible pixel 
        clock for the new settings.
        Changes to the image geometry or pixel clock affect the value ranges 
        of the frame rate and exposure time. After executing SetBinning(), 
        calling the following functions is recommended in order to keep 
        the defined camera settings:
        
        - SetFrameRate()
        - SetExposureTime()
        - If you are using the uEye's flash function: SetFlashStrobe()
        
        Syntax:
        =======
        
        rv=cam.SetSubSampling(mode)
        
        Input Parameters:
        =================
        
        mode:
            - SUBSAMPLING_DISABLE: Disables sub-sampling.
            - SUBSAMPLING_2X_VERTICAL: Enables vertical sub-sampling with factor 2.
            - SUBSAMPLING_3X_VERTICAL: Enables vertical sub-sampling with factor 3.
            - SUBSAMPLING_4X_VERTICAL: Enables vertical sub-sampling with factor 4.
            - SUBSAMPLING_5X_VERTICAL: Enables vertical sub-sampling with factor 5.
            - SUBSAMPLING_6X_VERTICAL: Enables vertical sub-sampling with factor 6.
            - SUBSAMPLING_8X_VERTICAL: Enables vertical sub-sampling with factor 8.
            - SUBSAMPLING_16X_VERTICAL: Enables vertical sub-sampling with factor 16.
            - SUBSAMPLING_2X_HORIZONTAL: Enables horizontal sub-sampling with factor 2.
            - SUBSAMPLING_3X_HORIZONTAL: Enables horizontal sub-sampling with factor 3.
            - SUBSAMPLING_4X_HORIZONTAL: Enables horizontal sub-sampling with factor 4.
            - SUBSAMPLING_5X_HORIZONTAL: Enables horizontal sub-sampling with factor 5.
            - SUBSAMPLING_6X_HORIZONTAL: Enables horizontal sub-sampling with factor 6.
            - SUBSAMPLING_8X_HORIZONTAL: Enables horizontal sub-sampling with factor 8.
            - SUBSAMPLING_16X_HORIZONTAL: Enables horizontal sub-sampling with factor 16.
            - GET_SUBSAMPLING: Returns the current setting.
            - GET_SUBSAMPLING_FACTOR_VERTICAL: Returns the vertical sub-sampling factor
            - GET_SUBSAMPLING_FACTOR_HORIZONTAL: Returns the horizontal sub-sampling factor
            - GET_SUBSAMPLING_TYPE: Indicates whether the camera uses colour-proof sub-sampling.
            - GET_SUPPORTED_SUBSAMPLING: Returns the supported sub-sampling modes.
                
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - NO_SUCCESS: General error message
            - Current setting when used together with GET_SUBSAMPLING
            - When used with GET_SUBSAMPLING_TYPE returns SUBSAMPLING_COLOR 
              if the camera uses colour-proof sub-sampling, else SUBSAMPLING_MONO 
            - When used with GET_SUPPORTED_SUBSAMPLING returns the supported 
              sub-sampling modes linked by logical ORs      
        '''
    
        rv=is_SetSubSampling (self.cid, mode)
        
        self.CheckNoSuccess(rv)
        return rv
    
    def GetTimeout (self, UINT nMode):
        '''Get user defined timeout
        
        Using GetTimeout(), you can read out user-defined timeout values from the uEye API.
        
        Syntax:
        =======
        
        pTimeout=cam.GetTimeout(nMode)
        
        Input Parameters:
        =================
        
        nMode: 
            Selects the timeout value to be returned
            - TRIGGER_TIMEOUT: Returns the timeout value in ms for triggered 
              image capture
        
            
        Return Values:
        ==============
        
        pTimeout:
            Timeout value. Returns 0 if the default value of the uEye API is used.
        '''
        cdef UINT pTimeout
        rv= is_GetTimeout (self.cid, nMode, & pTimeout)
        self.CheckNoSuccess(rv)
        return pTimeout
        
    def SetTimeout (self, UINT nMode, UINT Timeout):
        '''Set timeout value
        
        Using SetTimeout(), you can change user-defined timeout values of 
        the uEye API. If no user-defined timeout is set, the default value 
        of the uEye  API is used for the relevant timeout.

        ** The user-defined timeout only applies to the specified camera 
        at runtime of the program. **
        
        Syntax:
        =======
        
        rv=cam.SetTimeout(nMode, Timeout)
        
        Input Parameters:
        =================
        nMode: 
            Selects the timeout value to be set
            - TRIGGER_TIMEOUT: Sets the timeout value for triggered image capture
            
        Timeout: 
            Timeout value in 10 ms. Value range [0; 4...429496729] 
            (corresponds to 40 ms to approx. 1193 hours) 0 = use default value 
            of the uEye API, For 1...3, the value 4 is used.
        
        Return Values:
        SUCCESS: Function executed successfully
   
        NOT_SUPPORTED: The value for nMode is invalid
                
        '''
    
        rv=is_SetTimeout (self.cid, nMode, Timeout)
        self.CheckNoSuccess(rv)
        if  rv!= IS_SUCCESS:
            raise Exception("Could not set timeout")
        return rv
            
    def CaptureVideo (self, INT Wait):
        '''Capture Video
        
        CaptureVideo() digitises video images in real time and transfers 
        the images to an allocated image memory or, if Direct3D is used, 
        to the graphics card. The image data (DIB mode) is stored in the 
        memory created using AllocImageMem() and designated as active image 
        memory using SetImageMem(). Using GetImageMem(), you can query 
        the memory address. If ring buffering is used, the image capturing 
        function cycles through all image memories used for storing the 
        images of a capture sequence in an endless loop. Sequence memories 
        locked by LockSeqBuf() will be skipped. If the last available sequence 
        memory has been filled, the sequence event or message will be triggered. 
        Capturing always starts with the first element of the sequence.
        
        Syntax:
        =======
        
        rv=cam.CaptureVideo(Wait)
             
        
        Input Parameters:
        =================
        
        Wait:
            - DONT_WAIT:
            - WAIT: 
            - Time t:
            
            Timeout value for image capture  (see also the How To Proceed: 
            Timeout Values for Image Capture section of manual)
            - GET_LIVE: Returns if live capture is enabled.   
        
        Return Values:
        ==============
        rv:
            - SUCCESS: Function executed successfully
            - When used with GET_LIVE: TRUE if live capture is enabled
        
        '''
        
        rv=is_CaptureVideo (self.cid, Wait)
        self.CheckNoSuccess(rv)
        return rv
    
    def FreezeVideo (self, INT Wait):
        ''' Freeze video capture
        
        FreezeVideo() acquires a single image from the camera. In DIB mode, 
        the image is stored in the active image memory. If ring buffering 
        is used in DIB mode, the captured image is transferred to the next 
        available image memory of the sequence. Once the last available 
        sequence memory has been filled, the sequence event or message 
        will be triggered.

        In Direct3D mode, the is directly copied to the graphics card buffer 
        and then displayed.

        Image capture will be started by a trigger if you previously enabled 
        the trigger mode using SetExternalTrigger(). A hardware triggered 
        image acquisition can be cancelled using StopLiveVideo() if exposure 
        has not started yet.
        
        Syntax:
        =======
        rv=FreezeVideo(Wait)
        
        Input Parameters:
        =================
        Wait:
            - DONT_WAIT: 
            - WAIT
            - Time t

            Timeout value for image capture (see also the How To Proceed: 
            Timeout Values for Image Capture section in the manual)
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully

        '''
        
        rv=is_FreezeVideo (self.cid, Wait)
        self.CheckNoSuccess(rv)
        
    def StopLiveVideo (self, INT Wait):
        '''Stop Live Video
                  
        StopLiveVideo() stops live mode or cancels a hardware triggered image 
        capture in case the exposure has not yet started.
        
        Syntax:
        =======
        
        rv=cam.StopLiveVideo(Wait)
        
        Input Parameters:
        =================
        
        Wait:
            - WAIT: The function waits until the image save is complete.
            - DONT_WAIT:The function returns immediately. Digitising the image 
              is completed in the background.
            - FORCE_VIDEO_STOP: Digitising is stopped immediately.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
        '''
        
        rv= is_StopLiveVideo (self.cid, Wait)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetExternalTrigger (self, INT nTriggerMode):
        '''Set External Triger
        
        Using SetExternalTrigger(), you can activate the trigger mode.
        If the camera is in standby mode, it quits this mode and activates 
        trigger mode.

        In hardware trigger mode, image capture is delayed for each function 
        call until the selected trigger event has occurred.
        
        In software trigger mode, an image is captured immediately when 
        FreezeVideo() is called, or a continuous triggered capture is started 
        when CaptureVideo() is called. In hardware trigger mode, you can 
        use the ForceTrigger() command to trigger an image capture even if 
        no electric signal is present.

        When you disable the trigger functionality, you can statically 
        query the signal level at the trigger input. This option causes 
        the camera to change to freerun mode.
        
        ** For hardware reasons, the board-level versions of the USB uEye LE 
        cameras can only be triggered on the falling edge.**
        
        Syntax:
        =======
        
        rv=cam.SetExternalTrigger(nTriggerMode)
        
        Input Parameters:
        =================
        +--------------------------+-------------------+-----------------------+
        |nTriggerMode              |  Trigger mode     |    Trigger event      |
        +--------------------------+-------------------+-----------------------+
        |SET_TRIGGER_OFF           |  Off              |          -            |    
        +--------------------------+-------------------+-----------------------+
        |SET_TRIGGER_HI_LO         |  Hardware trigger |  Falling signal edge  | 
        +--------------------------+-------------------+-----------------------+
        |SET_TRIGGER_LO_HI         |  Hardware trigger |  Rising signal edge   |
        +--------------------------+-------------------+-----------------------+
        |SET_TRIGGER_HI_LO_SYNC    | Freerun sync./    | Falling signal edge   |
        |                          | hardware trigger* |                       |
        +--------------------------+-------------------+-----------------------+
        |SET_TRIGGER_LO_HI_SYNC    | Freerun sync./    | Rising signal edge    |
        |                          |  hardware trigger*|                       |
        +--------------------------+-------------------+-----------------------+
        |IS_SET_TRIGGER_SOFTWARE   |Software trigger   |Call of FreezeVideo()  |
        |                          |                   |(single frame mode)    |
        |                          |                   |Call of CaptureVideo() |
        |                          |                   |(continuous mode)      |
        +--------------------------+-------------------+-----------------------+
        
        +--------------------------+-------------------------------------------+
        |GET_EXTERNALTRIGGER       |Returns the trigger mode setting           |
        +--------------------------+-------------------------------------------+
        |GET_TRIGGER_STATUS        |Returns the current signal level at the    |
        |                          |trigger input                              |
        +--------------------------+-------------------------------------------+
        |GET_SUPPORTED_TRIGGER_MODE| Returns the supported trigger modes       |              |
        +--------------------------+-------------------------------------------+

        * The freerun synchronisation mode is currently only supported by the 
        UI-146x/546x series sensors.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - Current setting when used together with GET_EXTERNALTRIGGER
            - When used with GET_TRIGGER_STATUS: Returns the current signal 
              level at the trigger input
            - When used with GET_SUPPORTED_TRIGGER_MODE: Returns the supported 
              modes linked by logical ORs
        '''
    
        rv=is_SetExternalTrigger (self.cid, nTriggerMode)
        self.CheckNoSuccess(rv)
        return rv
            
    def ForceTrigger (self):
        ''' Force Trigger 
        
        You can use ForceTrigger() to force a software-controlled capture
        of an image while a capturing process triggered by hardware is in progress.
        This function can only be used if the triggered capturing process was 
        started using the DONE_WAIT parameter.
        
        
        Syntax:
        =======
        rv= cam.ForceTrigger()
        
        Return Values:
        ==============
        rv:
            - SUCCESS: Function executed successfully
    
        '''
        rv=is_ForceTrigger (self.cid)
        self.CheckNoSuccess(rv)
        
    #~def
    #INT is_GetCaptureErrorInfo (self.cid, UEYE_CAPTURE_ERROR_INFO* CaptureErrorInfo,UINT SizeCaptureErrorInfo)
    
    def LoadBadPixelCorrectionTable (self, char* File):
        '''Load bad pixel correction table
        
        LoadBadPixelCorrectionTable() loads a list of sensor hot pixel coordinates 
        that was previously saved using the SaveBadPixelCorrectionTable() function.
        
        
        Syntax:
        =======
        rv=cam.LoadBadPixelCorrectionTable (File)
        
        Input Values:
        =============
        
        File: 
            string which contains the name of the file where the coordinates are
            stored. You can either pass an absolute or a relative path.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully

        '''
        
        rv= is_LoadBadPixelCorrectionTable (self.cid, File)
        self.CheckNoSuccess(rv)
        return rv 
        
    def SaveBadPixelCorrectionTable (self, char* File):
        '''Save bad pixel correction table
        
        SaveBadPixelCorrectionTable() saves the user-defined hot pixel list 
        to the specified file.
        
        
        Syntax:
        =======
        
        rv=cam.SaveBadPixelCorrectionTable( File) 
        
        Input Values:
        =============
        
        File: 
            String which contains the name of the file where the coordinates are 
            stored. You can either pass an absolute or a relative path.

        Return Values:
        ==============
        
        rv:
            SUCCESS: Function executed successfully
        '''
    
        rv= is_SaveBadPixelCorrectionTable (self.cid, File)
        self.CheckNoSuccess(rv)
        return rv
        
    def SetBadPixelCorrection (self, INT nEnable, INT threshold):
        '''Set bad pixel correction table
        
        SetBadPixelCorrection() enables/disables the software correction 
        of sensor hot pixels.
        
        Syntax:
        =======
        
        rv=SetBadPixelCorrection (nEnable, threshold)
        
        Input Values:
        =============
        
        nEnable:
            - BPC_DISABLE: Disables the correction function.
            - BPC_ENABLE_SOFTWARE: Enables software correction based on the hot 
              pixel list stored in the EEPROM.
            - BPC_ENABLE_USER: Enables software correction based on user-defined 
              values. First, the SetBadPixelCorrectionTable() function must be called.
            - GET_BPC_MODE: Returns the current mode.
            - GET_BPC_THRESHOLD: Returns the current threshold value.
        
        threshold: Currently not used
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - Current mode when used in connection with GET_BPC_MODE
            - Current threshold value when used in connection with GET_BPC_THRESHOLD
        
        '''   
        
        rv=is_SetBadPixelCorrection (self.cid, nEnable, threshold)
        self.CheckNoSuccess(rv)
        return rv
    
    #~ def SetBadPixelCorrectionTable (self, INT nMode, list pixList):
        #~ '''Set or get the bad pixel correction table.
          #~ 
        #~ This method can be used to set the table containing the hot pixel 
        #~ positions which will be used by the user-defined hot pixel correction 
        #~ function. You can enable hot pixel correction by calling 
        #~ self.SetBadPixelCorrection(). Each value in List consists of an 
        #~ integer number. The coordinates are listed first X, then Y.
        #~ 
        #~ A table with 3 hot pixels must contain 6 values and will be 
        #~ structured as follows:
#~ 
        #~ X1 - Y1 - X2 - Y2 - X3 - Y3 
#~ 
        #~ Syntax:
        #~ =======
        #~ 
        #~ rl=cam.SetBadPixelCorrectionTable (nMode, pixList):
        #~ 
        #~ Input Parameters:
        #~ =================
        #~ nMode:
            #~ - SET_BADPIXEL_LIST: Sets a new user-defined list. The List parameter 
              #~ contains the data using the format described above.
            #~ - GET_BADPIXEL_LIST: Returns a python list containig the previously 
              #~ user-defined hot pixel list. In this case List is not used.
            #~ 
        #~ pixList: 
            #~ List containing the the hot pixel table, using the format described 
            #~ above
#~ 
        #~ Return Values:
        #~ ==============
        #~ 
        #~ rl:
            #~ A a python list containig the user-defined hot pixel list.
        #~ 
        #~ '''
        #~ 
        #~ cdef WORD * pList
        #~ 
        #~ if nMode==IS_SET_BADPIXEL_LIST:
            #~ n=len(pixList)
            #~ assert n%2==0, "Table lenght must be even"
            #~ pList= <WORD *>malloc((n+1)*sizeof(WORD))
            #~ pList[0]=n/2
        #~ 
            #~ for i,d in enumerate(pixList):
                #~ pList[i+1]=d
        #~ 
            #~ rv=is_SetBadPixelCorrectionTable (self.cid, nMode, pList)
        #~ 
            #~ self.CheckNoSuccess(rv)
            #~ free(pList)
            #~ rl= pixList
        #~ elif nMode==IS_GET_BADPIXEL_LIST:
            #~ rl=[]
            #~ n=is_SetBadPixelCorrectionTable (self.cid, IS_GET_LIST_SIZE, pList)
            #~ 
            #~ pList= <WORD *>malloc((2*n+1)*sizeof(WORD))
            #~ rv=is_SetBadPixelCorrectionTable (self.cid, nMode, pList)
            #~ self.CheckNoSuccess(rv)
            #~ 
            #~ for i in range(2*n):
                #~ rl.append(pList[i+1])
                #~ free(pList)
            #~ 
        #~ else:
            #~ raise Exception("Invalid nMode Parameter")
        #~ 
        #~ return rl
    
    def SetColorConverter (self, INT ColorMode, INT ConvertMode):
        '''Set color converter

        Using SetColorConverter, you can select the type of Bayer conversion 
        for colour cameras. Software conversion is done on the PC, while 
        hardware conversion (Gigabit Ethernet uEye only) is done in the camera. 
        The use of a larger filter mask results in a higher image quality, 
        but increases the computational load.
        
        ** Hardware conversion is only supported by GigE uEye cameras.
        While free run mode is active, you cannot change the colour conversion 
        type. To do so, you must first stop the capturing process using 
        StopLiveVideo() or set the camera to trigger mode 
        (see also SetExternalTrigger()).
        
        Syntax:
        =======
        
        rv=cam.SetColorConverter (ColorMode, ConvertMode) 

        
        Input Parameters:
        =================
        
        ColorMode: 
            Colour mode for which the converter is to be set.
        
        ConvertMode: 
            Conversion mode selection
            - CONV_MODE_SOFTWARE_3x3: Software conversion using the standard 
              filter mask (default)
            - CONV_MODE_SOFTWARE_5x5: Software conversion using a large filter mask
            - CONV_MODE_HARDWARE_3x3: Hardware conversion using the standard 
              filter mask (GigE uEye only)
            
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - CAPTURE_RUNNING: This function cannot be executed since the camera 
              is currently in live operation
            - INVALID_PARAMETER: The ConvertMode parameter is invalid or not supported
            - INVALID_COLOR_FORMAT: The ColorMode parameter is invalid or not supported
        
        '''
        
        rv=is_SetColorConverter (self.cid, ColorMode, ConvertMode) 
        self.CheckNoSuccess(rv)
        return rv
    
    def SetColorCorrection (self, INT nEnable):
        '''Set color correction
        
        For colour cameras, SetColorCorrection() enables colour correction 
        in the uEye driver. This enhances the rendering of colours for cameras 
        with colour sensors. Colour correction is a digital correction based 
        on a colour matrix which is adjusted individually for each sensor.
        If you perform Bayer conversion for GigE uEye HE colour cameras in 
        the camera itself, colour conversion will automatically also take 
        place in the camera. 
        
        ** After changing this parameter, perform manual or automatic white 
        balancing in order to obtain correct colour rendering 
        (see also SetAutoParameter()).**
        
        Syntax:
        =======
        
        rv=SetColorCorrection (nEnable)
        

        Input Parameters:
        =================
        
        nEnable:
            - CCOR_ENABLE_NORMAL: Enables simple colour correction. This parameter 
              replaces CCOR_ENABLE.
            - CCOR_ENABLE_BG40_ENHANCED: Enables colour correction for cameras with 
              optical IR filter glasses of the BG40 type.
            - CCOR_ENABLE_HQ_ENHANCED: Enables colour correction for cameras with 
              optical IR filter glasses of the HQ type.
            - CCOR_DISABLE: Disables colour correction.
            - GET_CCOR_MODE: Returns the current setting.
            - GET_SUPPORTED_CCOR_MODE: Returns all supported colour correction modes. 
            - GET_DEFAULT_CCOR_MODE: Returns the default colour correction mode.
            
        Return Values:
        ==============
        rv:
            - SUCCESS: Function executed successfully
            - Current setting when used together with GET_CCOR_MODE
            - When used together with GET_SUPPORTED_CCOR_MODE:
                When used for colour cameras and together with GET_SUPPORTED_CCOR_MODE, 
                this parameter returns the supported values linked by a logical OR:
                - CCOR_ENABLE_NORMAL
                - CCOR_ENABLE_BG40_ENHANCED
                - CCOR_ENABLE_HQ_ENHANCED
        
                When used for monochrome cameras, the system returns 0.
            - When used together with GET_DEFAULT_CCOR_MODE:
                When used for colour cameras and together with GET_DEFAULT_CCOR_MODE, 
                this parameter returns the default colour correction mode: 
                CCOR_ENABLE_NORMAL , CCOR_ENABLE_HQ_ENHANCED.
                When used for monochrome cameras, the system returns 0.
        '''
        
        cdef double* factors
        rv= is_SetColorCorrection (self.cid, nEnable, factors)
        self.CheckNoSuccess(rv)
        return rv
    
    def SetColorMode (self, INT Mode):
        '''Set and get color mode
        
        Sets the colour mode to be used when image data are saved or 
        displayed by the graphics card. For this purpose, the allocated 
        image memory must be large enough to accommodate the data with 
        the selected colour mode. When images are transferred directly 
        to the graphics card memory, make sure that the display settings 
        match the colour mode settings. Otherwise, the images will be 
        displayed with altered colours or are not clearly visible.

        For the RGB16 and RGB15 data formats, the MSBs of the internal 
        8-bit R, G and B colours are used.

        Syntax:
        =======
        
        rv=cam.SetColorMode (Mode)
               
        Input Parameters
        ================
        
        Mode: 
            Colour mode to be set
            - CM_BAYER_RG16       -       Raw Bayer (16)
            - CM_BAYER_RG12       -       Raw Bayer (12)
            - CM_BAYER_RG8        -       Raw Bayer (8)
            - CM_MONO16           -       Greyscale (16)
            - CM_MONO12           -       Greyscale (12)
            - CM_MONO8            -       Greyscale (8)
            - CM_RGB10V2_PACKED   -       RGB30 (10 10 10)
            - CM_RGBA8_PACKED     -       RGB32 (8 8 8)
            - CM_RGBY8_PACKED     -       RGBY (8 8 8 8)
            - CM_RGB8_PACKED      -       RGB24 (8 8 8)
            - CM_BGR10V2_PACKED   -       BGR30 (10 10 10)
            - CM_BGRA8_PACKED     -       BGR32 (8 8 8)
            - CM_BGR8_PACKED      -       BGR24 (8 8 8)
            - CM_BGRY8_PACKED     -       BGRY (8 8 8)
            - CM_BGR565_PACKED    -       BGR16 (5 6 5)
            - CM_BGR555_PACKED    -       BGR15 (5 5 5)
            - CM_UYVY_PACKED      -       UYVY (8 8 8 8)
            - CM_UYVY_MONO_PACKED -       UYVY (8 8 8 8)
            - CM_UYVY_BAYER_PACKED-       UYVY (8 8 8 8)
            - CM_CBYCRY_PACKED    -       CbYCrY (8 8 8 8)
            - GET_COLOR_MODE      -       Returns the current setting.
            
        Return Values:
        ==============
        
        rv:
            if Mode != GET_COLORMODE returns Mode
            If Mode == GET_COLORMODE returns the actual colormode
            
        Note:
        =====
        The self.Img memory block gets its size adjusted when the color mode is
        set.
        
        '''
        
        rv=is_SetColorMode (self.cid,  Mode)
        self.CheckNoSuccess(rv)
        if Mode==IS_GET_COLOR_MODE:
            return rv

        # Save information relevant to the colormode
        self.colormode= is_SetColorMode(self.cid, IS_GET_COLOR_MODE)        
        self.bitspixel=bitspixel(self.colormode)
        
        if self.Img!=NULL:
            rv=is_FreeImageMem (self.cid, self.Img, self.ImgMemId)
            self.CheckNoSuccess(rv)
        
        rv=is_AllocImageMem(self.cid, self.nMaxWidth, self.nMaxHeight, self.bitspixel, &self.Img, &self.ImgMemId)
        self.CheckNoSuccess(rv)
        
        rv=is_SetImageMem (self.cid, self.Img, self.ImgMemId)
        self.CheckNoSuccess(rv)
        
        
        rv=is_GetImageMemPitch (self.cid, &self.LineInc)
        self.CheckNoSuccess(rv)
        
        return Mode
    
    def SetConvertParam (self, BOOL ColorCorrection, INT BayerConversionMode, INT ColorMode, INT Gamma, tuple WhiteBalanceMultipliers):
        '''Set convertion parameters
        
        Using SetConvertParam(), you can set the parameters for converting 
        a raw Bayer image to a colour image. To convert the image, use 
        the ConvertImage() function. 
        
        Syntax:
        =======
        
        rv=SetConvertParam (ColorCorrection, BayerConversionMode, 
                  ColorMode, Gamma, WhiteBalanceMultipliers)
        
        
        Input Parameters:
        =================
        
        ColorCorrection: 
            Enables / disables colour correction.

        BayerConversionMode: 
            Sets the Bayer conversion mode.
            - SET_BAYER_CV_BETTER: Better quality
            - SET_BAYER_CV_BEST: Optimum quality (higher CPU load)

        ColorMode: 
            Sets the colour mode for the output image.

        Gamma: 
            Gamma value multiplied by 100. Range: [1…1000]

        WhiteBalanceMultipliers: 
            Tuple containing the red, green and blue gain values
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - INVALID_COLOR_FORMAT: Invalid ColorMode parameter
            - INVALID_PARAMETER: Other invalid parameter.
        
        '''
        cdef double WBM[3]
        assert len(WhiteBalanceMultipliers)==3,"the tuple WhiteBalanceMultipliers must contain 3 double numbers"
        WBM[0]=WhiteBalanceMultipliers[0]
        WBM[1]=WhiteBalanceMultipliers[1]
        WBM[2]=WhiteBalanceMultipliers[2]
        rv=is_SetConvertParam (self.cid, ColorCorrection, BayerConversionMode, ColorMode, Gamma, WBM)
        self.CheckNoSuccess(rv)
        return rv
            
    def SetEdgeEnhancement (self, INT nEnable):
        '''Set edge enhancement 
        
        SetEdgeEnhancement() enables a software edge filter. Due to Bayer 
        format colour conversion, the original edges of a colour image 
        may easily become blurred. By enabling the digital edge filter, 
        you can optimise edge representation. This function causes a higher 
        CPU load.

        If you perform Bayer conversion for GigE uEye HE colour cameras 
        in the camera itself, edge enhancement will automatically also take 
        place in the camera. In this case, the CPU load will not increase.
        
        Syntax:
        =======
        
        RV=SetEdgeEnhancement (nEnable)
        
        Input Parameters:
        =================
        
        nEnable:
            - EDGE_EN_DISABLE: Disables the edge filter.
            - EDGE_EN_STRONG: Enables strong edge enhancement.
            - EDGE_EN_WEAK: Enables weaker edge enhancement.
            - GET_EDGE_ENHANCEMENT: Returns the current setting.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - Current setting when used together with GET_EDGE_ENHANCEMENT
        
        '''
    
        rv=is_SetEdgeEnhancement (self.cid, nEnable)
        self.CheckNoSuccess(rv)
        return rv 

    def SetSaturation (self, INT ChromU, INT ChromV):
        '''Set saturation
        
        Using SetSaturation(), you can set the software colour saturation. 
        
        In the YUV format, colour information (i.e. the colour difference signals) 
        is provided by the U and V channels. In the U channel, this information 
        results from the difference between the blue level and Y (luminance), 
        in the V channel from the difference between the red level and Y.

        For use in other colour formats than YUV, U and V are converted 
        using a driver matrix.
        
        
        Syntax:
        =======
        
        rv=cam.SetSaturation (ChromU, ChromV)
                
        Input Parameters:
        =================
        
        ChromU:
            U saturation: value multiplied by 100.
            
            Range: [MIN_SATURATION … MAX_SATURATION]
            
            GET_SATURATION_U: Returns the current value for the U saturation.

        ChromV:
            V saturation: value multiplied by 100.
            
            Range: [MIN_SATURATION … MAX_SATURATION]
            
            GET_SATURATION_V: Returns the current value for the V saturation.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - Current setting when used together with, GET_SATURATION_U, 
              GET_SATURATION_V
            - INVALID_PARAMETER: Invalid value for the ChromU or ChromV parameter.
        
        '''
    
        rv=is_SetSaturation (self.cid, ChromU, ChromV) 
        self.CheckNoSuccess(rv)
        return rv 
    
    
    def SetSensorTestImage (self, INT TestImage, INT Param):
        '''Set the sensor test image
        
        SetSensorTestImage() enables a test image function in the sensor. 
        You can select different test images. The test images supported 
        by a particular camera can be queried using the GetSupportedTestImages()
        function. For some test images, the Param  parameter provides 
        additional options. If the test image does not support additional parameters, 
        Param will be ignored.
        
        Syntax:
        =======
        
        rv=cam=SetSensorTestImage (TestImage, Param) 
        
        Input Parameters:
        =================
        
        TestImage: 
            The test image to be set. See also GetSupportedTestImages().

        Param: 
            Additional parameter for used to modify the test image. Not available 
            for all test images.
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - INVALID_PARAMETER: The Param parameter is not within the allowed 
              value range.
            - NOT_SUPPORTED: The test image function is not supported by the camera.
        
        '''
        
        rv=is_SetSensorTestImage (self.cid, TestImage, Param)
        self.CheckNoSuccess(rv)
        return rv  

    def  CameraStatus(self, INT nInfo, ULONG ulValue):
        '''Get the camera status
        
        Using CameraStatus(), you can query and partly set various status 
        information and settings.


        Syntax:
        =======
        rv= cam.CameraStatus(nInfo, ulValue) 
        
        Input Parameters:
        =================
        
        nInfo:
            - FIFO_OVR_CNT: Number of FIFO overruns. Is increased if image data 
              gets lost because the USB bus is congested.
            - SEQUENCE_CNT: Returns the sequence count. For CaptureVideo(), this 
              parameter is set to 0. Each time the sequence buffer (image counter) 
              changes, the counter is increased by 1.
            - SEQUENCE_SIZE: Returns the number of sequence buffers.
            - EXT_TRIGGER_EVENT_CNT: Returns the camera's internal count of external 
              trigger events.
            - TRIGGER_MISSED: Returns the number of unprocessed trigger signals. 
              Is reset to 0 after each call.
            - LAST_CAPTURE_ERROR: Returns the last image capture error, e.g. after 
              a 'transfer failed' event. For a list of all possible error events, 
              see GetCaptureErrorInfo().
            - PARAMETER_SET_1: Indicates whether parameter set 1 including camera 
              settings is present on the camera (read-only). 
              See also SaveParameters().
            
              Return values:
                - TRUE        Parameter set 1 present
                - FALSE       Parameter set 1 not present

            - PARAMETER_SET_2: Indicates whether parameter set 2 including camera 
              settings is present on the camera (read-only). 
              See also SaveParameters().
             
              Return values:
              
              - TRUE        Parameter set 2 present
              - FALSE       Parameter set 2 not present

            - STANDBY: Sets the camera to standby mode.
                Return values:
                - TRUE        Camera changes to standby mode
                - FALSE       The camera changes to freerun mode

            - STANDBY_SUPPORTED: Queries whether the camera supports standby mode (read-only).
                Return values:
                - TRUE        The camera supports standby mode
                - FALSE       The camera does not support standby mode
        
        Return Values:
        ==============
        rv:
            - SUCCESS: Function executed successfully
            - Returns the information specified by nInfo: Only if ulValue = GET_STATUS
            - When used with LAST_CAPTURE_ERROR returns the last image capture 
            error. For a list of all possible error events, see GetCaptureErrorInfo().
        
        '''
        rv=is_CameraStatus(self.cid, nInfo, ulValue)
        self.CheckNoSuccess(rv)
        return rv  

    def GetBusSpeed(self):
        '''Get bus speed
        
        Using GetBusSpeed(), you can query whether a camera is connected 
        to a USB 2.0 host controller.
        When the value 0 is passed for hCam, the function checks whether 
        a USB 2.0 controller is present in the system.

        Syntax:
        =======
        
        rv=cam.GetBusSpeed ()
        
        Return Values:
        ==============
        
        rv:
            - SUCCESS: Function executed successfully
            - USB_10: The controller to which the camera is connected does not 
              support USB 2.0.
            - USB_20: The camera is connected to a USB 2.0 controller.
                
        '''
        ### Toca hacer uno igual a este pero que este en el modulo
        rv=is_GetBusSpeed(self.cid)
        return rv

    def GetCameraType(self):
        '''Get camera type
        
        GetCameraType() returns the camera type.
        
        Syntax:
        =======
        
        rv=cam.GetCameraType ()
        
        Return Values:
        ==============
        
        rv:
            - CAMERA_TYPE_UEYE_USB_SE: USB uEye SE camera
            - CAMERA_TYPE_UEYE_USB_ME: USB uEye ME camera
            - CAMERA_TYPE_UEYE_USB_RE: USB uEye RE camera
            - CAMERA_TYPE_UEYE_USB_LE: USB uEye LE camera
            - CAMERA_TYPE_UEYE_ETH_HE: GigE uEye HE camera
            - CAMERA_TYPE_UEYE_ETH_SE: GigE uEye SE camera
            
        '''
        return is_GetCameraType(self.cid)
    
    def CheckNoSuccess(self,INT rv):
        '''Method that checks the return value of a is_XXXX function.
        
        If rv==NO_SUCCESS, the error mesage is printed and a exception is raised
        '''
        
        cdef char * ermsg
        if rv==IS_NO_SUCCESS:
            rv1=is_GetError (self.cid, &rv, &ermsg)
            if rv1==IS_NO_SUCCESS:
                raise Exception("Error getting error message")
            raise Exception(ermsg)
            
            
       
    
    def GetGlobalFlashDelays(self):
        '''Get global flash delay
        
        Rolling shutter cameras:

        Using GetGlobalFlashDelays(), you can determine the times required 
        to implement a global flash function for rolling shutter cameras. 
        This way, a rolling shutter camera can also be used as a global shutter 
        camera provided that no ambient light falls on the sensor outside 
        the flash period.
        If the exposure time is set too short so that no global flash operation 
        is possible, the function returns NO_SUCCESS.
        
        ** To use a rolling shutter camera with the Global Start function, 
        call the SetGlobalShutter() function before GetGlobalFlashDelays(). 
        Otherwise, incorrect values will be returned for Delay and Duration. **

        Global shutter cameras:

        In freerun mode, the exposure of global shutter cameras is delayed 
        if the exposure time is not set to the maximum value. GetGlobalFlashDelays() 
        determines the required delay in order to synchronise exposure and flash operation. 
        In triggered mode, the return values for delay and flash duration are 0, 
        since no delay is necessary before exposure starts.

        
        Syntax:
        =======
        
        delay,duration=cam.GetGlobalFlashDelays ()
        
        Return Values:
        ==============
        
        delay:
            flash delay in µs.
            
        duration: 
            flash duration in µs.
        
    
        '''
        
        cdef ULONG pulDelay
        cdef ULONG pulDuration
        rv=is_GetGlobalFlashDelays(self.cid,&pulDelay, &pulDuration)
        self.CheckNoSuccess(rv)
        return pulDelay,pulDuration
    
    def GetImageHistogram(self):
        """
        Syntax:
        INT GetImageHistogram (int nID, INT ColorMode, DWORD pHistoMem)
        
        Description:
        GetImageHistogram() computes the histogram of the submitted image. 
        The histogram always contains 256 values per channel. For colour modes 
        with a bit depth of more than 8 bits, the system evaluates the 8 most 
        significant bits (MSBs).

        Input Parameters:
        nID: Memory ID

        ColorMode: Colour mode of the image with the nID memory ID

        pHistoMem: Pointer to a DWORD array
        The array must be allocated in such a way that it can accommodate 3*256 
        values for colour formats and in raw Bayer mode. In monochrome mode, 
        the array must be able to accommodate 1*256 values.
        
        Return Values:
        SUCCESS: Function executed successfully

        NO_SUCCESS: General error message
        
        IS_NULL_POINTER: Invalid Array

        INVALID_COLOR_FORMAT: Unsupported colour format

        INVALID_PARAMETER: Unknown ColorModeparameter
                
        """
    
        pass
        
    
### Missing 
#AddToSequence
#AllocImageMem
#ClearSequence
#ConvertImage
#CopyImageMem
#CopyImageMemLines
#DirectRenderer---- windows
#DisableEvent
#EnableAutoExit
#EnableEvent
#EnableHdr
#EnableMessage ---- windows
#ExitCamera -> this is in the class destructor
#ExitEvent
#FreeImageMem-> this is in the class destructor
#GetActiveImageMem
#GetActSeqBuf
#GetAutoInfo
#GetCameraInfo -> This is called in the class constructor
#GetCameraList -> This is in the module, out of the class, but it is not working
#GetCameraLUT   ---- Giga
#GetCaptureErrorInfo
#GetColorConverter
#GetColorDepth ---windows
#GetComportNumber ---- Giga
#GetDuration --- GigE
#GetEthDeviceInfo --Giga
#GetError ------> CheckNoSuccess(self,INT rv):


#GetHdrKneepointInfo
#GetHdrKneepoints
#GetHdrMode

#GetImageInfo
#GetImageMem
#GetImageMemPitch
#GetNumberOfCameras-> This is out of the class, in the module
#GetOsVersion 
#GetSensorInfo -> This is in the init of the class
#GetSensorScalerInfo
#GetSupportedTestImages
#GetTestImageValueRange
#GetTimeout
#GetUsedBandwidth
#GetVsyncCount
#HasVideoStarted
#InitCamera -> This one is in the constructor of the class
#InitEvent
#InquireImageMem
#IsVideoFinish
#LoadImage
#LoadImageMem
#LoadParameters
#LockSeqBuf
#ReadEEPROM
#ReadI2C
#RenderBitmap ---- windows
#ResetCaptureErrorInfo
#SaveImage
#SaveImageEx
#SaveImageMem
#SaveImageMemEx
#SaveParameters
#SetAllocatedImageMem   
#SetAutoCfgIpSetup
#SetCameraID
#SetCameraLUT
#SetDisplayMode --- windows
#SetDisplayPos  ----windows
#SetErrorReport  
#SetFlashDelay
#SetFlashStrobe
#SetHdrKneepoints   
#SetImageMem
#SetImagePos
#SetIO
#SetIOMask
#SetLED
#SetOptimalCameraTiming
#SetPacketFilter
#SetPersistentIpCfg 
#SetSensorScaler ???????
#SetSensorTestImage
#SetStarterFirmware
#SetTriggerCounter
#SetTriggerDelay
#UnlockSeqBuf
#WaitEvent 
#WriteEEPROM
#WriteI2C
