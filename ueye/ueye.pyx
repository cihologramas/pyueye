#!python
#cython: embedsignature=True

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
from ueyeh cimport *
#~ from ueyeh import *
from sys import stderr
cimport cython



###Dictionary with the allowed AOI parameters for the different sensors
### Note may not be complete. It is used in the AOI setting to adjust the 
### parameters to the allowed values. The content of the dict are:
## MinWidth,MaxWidth,WidthStep,MinHeigth,MaxHeight,HeightStep,PosGridHoriz,PosGridVert

cdef dict AOIinfo={\
            #IS_SENSOR_UI141X_M          
            #IS_SENSOR_UI141X_C          
            #IS_SENSOR_UI144X_M          
            #IS_SENSOR_UI144X_C          

            c_IS_SENSOR_UI154X_M: (32, 1280,4,4,1024,2,4,2),          
            #IS_SENSOR_UI154X_C          
            #IS_SENSOR_UI145X_C          

            #IS_SENSOR_UI146X_C          
            #IS_SENSOR_UI148X_M          
            #IS_SENSOR_UI148X_C          

            #IS_SENSOR_UI121X_M          
            #IS_SENSOR_UI121X_C          
            #IS_SENSOR_UI122X_M          
            #IS_SENSOR_UI122X_C          

            #IS_SENSOR_UI164X_C          

            #IS_SENSOR_UI155X_C          

            #IS_SENSOR_UI1223_M          
            #IS_SENSOR_UI1223_C          

            #IS_SENSOR_UI149X_M          
            #IS_SENSOR_UI149X_C          

            #IS_SENSOR_UI1225_M          
            #IS_SENSOR_UI1225_C          

            c_IS_SENSOR_UI1645_C:  (32, 1280,4,4,1024,2,4,2),       
            #IS_SENSOR_UI1555_C          
            c_IS_SENSOR_UI1545_M:  (32, 1280,4,4,1024,2,4,2),       
            #IS_SENSOR_UI1545_C          
            #IS_SENSOR_UI1455_C          
            #IS_SENSOR_UI1465_C          
            #IS_SENSOR_UI1485_M          
            #IS_SENSOR_UI1485_C          
            #IS_SENSOR_UI1495_M          
            #IS_SENSOR_UI1495_C          

            #IS_SENSOR_UI112X_M          
            #IS_SENSOR_UI112X_C          

            #IS_SENSOR_UI1008_M          
            #IS_SENSOR_UI1008_C          

            #IS_SENSOR_UIF005_M           
            #IS_SENSOR_UIF005_C           

            #IS_SENSOR_UI1005_M           
            #IS_SENSOR_UI1005_C          

            #IS_SENSOR_UI1240_M          
            #IS_SENSOR_UI1240_C          
            #IS_SENSOR_UI1240_NIR        

            #IS_SENSOR_UI1240LE_M        
            #IS_SENSOR_UI1240LE_C        
            #IS_SENSOR_UI1240LE_NIR      

            #IS_SENSOR_UI1240ML_M        
            #IS_SENSOR_UI1240ML_C        
            #IS_SENSOR_UI1240ML_NIR      

            #IS_SENSOR_UI1243_M_SMI      
            #IS_SENSOR_UI1243_C_SMI      

            #IS_SENSOR_UI1543_M          
            #IS_SENSOR_UI1543_C          

            #IS_SENSOR_UI1544_M          
            #IS_SENSOR_UI1544_C          
            #IS_SENSOR_UI1543_M_WO       
            #IS_SENSOR_UI1543_C_WO       
            #IS_SENSOR_UI1453_C          
            #IS_SENSOR_UI1463_C          
            #IS_SENSOR_UI1483_M          
            #IS_SENSOR_UI1483_C          
            #IS_SENSOR_UI1493_M          
            #IS_SENSOR_UI1493_C          

            #IS_SENSOR_UI1463_M_WO       
            #IS_SENSOR_UI1463_C_WO       

            #IS_SENSOR_UI1553_C_WN       
            #IS_SENSOR_UI1483_M_WO       
            #IS_SENSOR_UI1483_C_WO       

            #IS_SENSOR_UI1580_M          
            #IS_SENSOR_UI1580_C          
            #IS_SENSOR_UI1580LE_M        
            #IS_SENSOR_UI1580LE_C        

            #IS_SENSOR_UI1360M           
            #IS_SENSOR_UI1360C           
            #IS_SENSOR_UI1360NIR         

            #IS_SENSOR_UI1370M           
            #IS_SENSOR_UI1370C           
            #IS_SENSOR_UI1370NIR         

            #IS_SENSOR_UI1250_M          
            #IS_SENSOR_UI1250_C          
            #IS_SENSOR_UI1250_NIR        

            #IS_SENSOR_UI1250LE_M        
            #IS_SENSOR_UI1250LE_C        
            #IS_SENSOR_UI1250LE_NIR      

            #IS_SENSOR_UI1250ML_M        
            #IS_SENSOR_UI1250ML_C        
            #IS_SENSOR_UI1250ML_NIR      

            #IS_SENSOR_XS                

            #IS_SENSOR_UI1493_M_AR       
            #IS_SENSOR_UI1493_C_AR       
            #IS_SENSOR_UI223X_M          
            #IS_SENSOR_UI223X_C          

            #IS_SENSOR_UI241X_M          
            #IS_SENSOR_UI241X_C          

            #IS_SENSOR_UI234X_M          
            #IS_SENSOR_UI234X_C          

            #IS_SENSOR_UI221X_M          
            #IS_SENSOR_UI221X_C          

            #IS_SENSOR_UI231X_M          
            #IS_SENSOR_UI231X_C          

            #IS_SENSOR_UI222X_M          
            #IS_SENSOR_UI222X_C          

            #IS_SENSOR_UI224X_M          
            #IS_SENSOR_UI224X_C          

            #IS_SENSOR_UI225X_M          
            #IS_SENSOR_UI225X_C          

            #IS_SENSOR_UI214X_M          
            #IS_SENSOR_UI214X_C          

            #IS_SENSOR_UI228X_M          
            #IS_SENSOR_UI228X_C          

            #IS_SENSOR_UI241X_M_R2       
            #IS_SENSOR_UI251X_M          
            #IS_SENSOR_UI241X_C_R2       
            #IS_SENSOR_UI251X_C          

            #IS_SENSOR_UI2130_M          
            #IS_SENSOR_UI2130_C 
            }         


cdef inline npy.ndarray zero_mat( int M, int N ):
    cdef npy.npy_intp length[2]
    length[0] = M; length[1] = N
    npy.Py_INCREF( npy.NPY_DOUBLE ) # This is apparently necessary
    return npy.PyArray_ZEROS( 2, length, npy.NPY_DOUBLE, 0 )

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
    if colormode==c_IS_CM_MONO8:
        return 8
    elif colormode==c_IS_CM_MONO12        or colormode==c_IS_CM_MONO16        \
      or colormode==c_IS_CM_BGR565_PACKED \
      or colormode==c_IS_CM_UYVY_PACKED   or colormode==c_IS_CM_CBYCRY_PACKED:
        return 16
    elif colormode==c_IS_CM_RGB8_PACKED or colormode==c_IS_CM_BGR8_PACKED:
        return 24
    elif colormode==c_IS_CM_RGBA8_PACKED or colormode==c_IS_CM_BGRA8_PACKED   \
      or colormode==c_IS_CM_RGBY8_PACKED or colormode==c_IS_CM_BGRY8_PACKED:
        return 32;
    else: return 8

def SetErrorReport(enable):
    '''Enable or disable API error reporting

    When enabled, verbose errors will be printed to stderr directly
    from the driver API calls.

    is_SetErrorReport() can be called before calling is_InitCamera().
    You only need to enable the is_SetErrorReport() function once for all 
    cameras in the application.

    TODO: Implement GetErrorReport?

    Sytnax:
    =======

    ueye.SetErrorReport(enable)

    Input Parameters:
    =================

    enable:  True = turn on error reporting. False = turn off (default state)
    '''

    if enable:
        mode = c_IS_ENABLE_ERR_REP
    else:
        mode = c_IS_DISABLE_ERR_REP
    rv= is_SetErrorReport (0, mode)
    if rv != c_IS_SUCCESS:
        raise Exception("Error setting ErrorReporting. API returned %d" % rv)
    
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
    cdef char **Imgs
    cdef int BufCount
    cdef int *BufIds
    cdef char *LastSeqBuf
    cdef char *LastSeqBuf1
    cdef char LastSeqBufLocked
    cdef public HIDS cid
    cdef public INT nMaxWidth,nMaxHeight,nColorMode, colormode
    cdef public object SerNo,ID,Version,Date,Select,SensorID,strSensorName, \
            bMasterGain, bRGain, bGGain, bBGain, \
            bGlobShutter, bitspixel
    cdef public INT LineInc, 
    cdef public int ImgMemId
    cdef public int LiveMode
    cdef INT AOIx0,AOIy0,AOIx1,AOIy1 ##Buffers to save the AOI to speed up the image grabbing
    
        
    def __init__(self, HIDS cid=0, int bufCount=3):
    
    
        rv=is_InitCamera(&cid, NULL)
        self.CheckNoSuccess(rv,"Error in Cam.__init__. Could not init the camera")

        if rv==c_IS_STARTER_FW_UPLOAD_NEEDED:
            raise Exception("The camera's starter firmware is not compatible with the driver and needs to be updated.")
        
        self.cid=cid
        
        ##If True, the camera is in livemode, this is any change in the image 
        ##is visible in the image buffer in realtime (no need to grab)
         
        self.LiveMode = False
        
        cdef CAMINFO cInfo
        rv =is_GetCameraInfo(self.cid, &cInfo)
        self.CheckNoSuccess(rv,"Error in Cam.__init__. Could not get camera info")
        
        self.SerNo=cInfo.SerNo
        self.ID=cInfo.ID
        self.Version=cInfo.Version
        self.Date=cInfo.Date
        self.Select=cInfo.Select
        #pInfo.Type,pInfo.Reserved
        
        cdef SENSORINFO sInfo
        rv=is_GetSensorInfo(self.cid, &sInfo)
        
        self.CheckNoSuccess(rv,"Error in Cam.__init__. Could not get sensor info")
        
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
        if self.nColorMode == c_IS_COLORMODE_BAYER:
            colormode= c_IS_CM_BGR8_PACKED
        elif self.nColorMode == c_IS_COLORMODE_MONOCHROME:
            colormode= c_IS_CM_MONO8
        else:
            raise Exception("Colormode not supported")
        
        
        #Set colormode and assign image memory. The image memory assigment is done 
        #in SetColorMode
    
        self.BufCount = bufCount
        self.BufIds = <int *>calloc(bufCount, sizeof(int))
        self.Imgs = <char **>calloc(bufCount, sizeof(char*))
        self.SetColorMode (colormode)    
    
        rv=is_SetExternalTrigger (self.cid, c_IS_SET_TRIGGER_OFF)
        self.CheckNoSuccess(rv, "Error in Cam.__init__. Could not set trigger")
        
        rv=is_FreezeVideo (self.cid, c_IS_WAIT)
        self.CheckNoSuccess(rv, "Error in Cam.__init__. Could not Freeze video")
        
        
        # Enable FRAME events to make video capturing easier:
        self.CheckNoSuccess(is_EnableEvent(self.cid, c_IS_SET_EVENT_FRAME),\
                            "Error in Cam.__init__. Could not enable frame event")
        
        #according to the ueye manual V 4.31 The init event is not used in linux
        #just in Windows
        
        #self.CheckNoSuccess(is_InitEvent(self.cid, NULL, c_IS_SET_EVENT_FRAME),\
        #                    "Error in Cam.__init__. Could not init frame event")
        
        
        #Start with auto BL_Compensation off by default if possible
        try:
            self.AutoBlackLevel=False
            self.BlackLevelOffset=0
        except AttributeError:
            pass
            
        self.LastSeqBuf=NULL
        self.LastSeqBufLocked=False
        
        ##Save the initial AOI this is used to speed up the GrabImage when using
        ##AOI=True
        self.AOIx0,self.AOIy0,w,h=self.AOI
        self.AOIx1=self.AOIx0+w
        self.AOIy1=self.AOIy0+h
        

    def __dealloc__(self):
        
        for i in range(self.BufCount):
            rv=is_FreeImageMem (self.cid, self.Imgs[i], self.BufIds[i])
            self.CheckNoSuccess(rv)

        if self.cid:
            rv=is_ExitCamera (self.cid)
            self.CheckNoSuccess(rv)            
    
    property AutoBlackLevel:
        
        def __get__(self):
            """Get current blacklevel mode
            """
            
            cdef INT rv, nMode
            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_GET_MODE, <void*>&nMode, sizeof(nMode))
            self.CheckNoSuccess(rv, "Couldn't get the AutoBlackLevel info")
            return nMode==c_IS_AUTO_BLACKLEVEL_ON
        
        def __set__(self, value):
            """Set the Auto Blacklevel parameter
            """
            assert value in (True,False), "AutoBlackLevel must be True or False"
        
            # Check if BL can be changed
            cdef INT nBlacklevelCaps, rv, nMode
            cdef BOOL bSetAutoBlacklevel, bSetBlacklevelOffset
            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_GET_CAPS,
                    <void*>&nBlacklevelCaps, sizeof(nBlacklevelCaps))

            self.CheckNoSuccess(rv, "Couldn't get the BlackLevel caps")
            
            #Check if the user can changed the state of the auto blacklevel
            bSetAutoBlacklevel = (nBlacklevelCaps & c_IS_BLACKLEVEL_CAP_SET_AUTO_BLACKLEVEL) != 0
                
            if not bSetAutoBlacklevel: raise AttributeError("The current camera does not support AutoBlackLevel settings")

            #Set new BL mode 
            nMode= c_IS_AUTO_BLACKLEVEL_ON if value else c_IS_AUTO_BLACKLEVEL_OFF

            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_SET_MODE, <void*>&nMode , sizeof(nMode ));
            self.CheckNoSuccess(rv, "Couldn't set the AutoBlackLevel")
    
    property BlackLevelOffset:
        
        def __get__(self):
        
            cdef INT rv, nMode
            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_GET_OFFSET, <void*>&nMode, sizeof(nMode))
            self.CheckNoSuccess(rv, "Couldn't get the AutoBlackLevel info")
            return nMode
        
        def __set__(self, INT value):
        
            # Check if BL can be changed
            cdef INT nBlacklevelCaps,rv
            cdef BOOL bSetAutoBlacklevel, bSetBlacklevelOffset
            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_GET_CAPS,
                    <void*>&nBlacklevelCaps, sizeof(nBlacklevelCaps))

            self.CheckNoSuccess(rv, "Couldn't get the BlackLevel caps")
                
            #The user can change the offset
            bSetBlacklevelOffset = (nBlacklevelCaps & c_IS_BLACKLEVEL_CAP_SET_OFFSET) != 0
            if not bSetBlacklevelOffset: raise AttributeError("The current camera does not support BlackLevelOffset settings")
            

            rv = is_Blacklevel(self.cid, c_IS_BLACKLEVEL_CMD_SET_OFFSET, <void*>&value, sizeof(value));
            self.CheckNoSuccess(rv, "Couldn't set BlackLevelOffset")
    
    property PixelClock:
        def __get__(self):
            cdef UINT value
            rv=is_PixelClock(self.cid, c_IS_PIXELCLOCK_CMD_GET, &value, sizeof(value))
            self.CheckNoSuccess(rv)
            return value
            
        def __set__(self, UINT value):
            rv=is_PixelClock(self.cid, c_IS_PIXELCLOCK_CMD_SET, &value, sizeof(value))
            self.CheckNoSuccess(rv)
            #return rv
    
    property PixelClockRange:
        def __get__(self):
            
            cdef UINT nRange[3]

            #ZeroMemory(nRange, sizeof(nRange));

            cdef rv = is_PixelClock(self.cid, c_IS_PIXELCLOCK_CMD_GET_RANGE, <void*>  nRange, sizeof(nRange))
            self.CheckNoSuccess(rv)
            return nRange[0],nRange[1],nRange[2]
    
    property ExposureTime:
        def __set__(self,double expo):
            rv= is_Exposure (self.cid, c_IS_EXPOSURE_CMD_SET_EXPOSURE, &expo, sizeof(double))
            self.CheckNoSuccess(rv)
        
        def __get__(self):
            cdef double nexpo
            rv= is_Exposure (self.cid, c_IS_EXPOSURE_CMD_GET_EXPOSURE, &nexpo, sizeof(double))
            self.CheckNoSuccess(rv)
            return nexpo
            
    property ExposureTimeRange:
        def __get__(self):
            cdef double dblRange[3]
            cdef INT rv
 
            rv = is_Exposure(self.cid, c_IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,<void*>dblRange, sizeof(dblRange))
            self.CheckNoSuccess(rv)
            
            return dblRange[0],dblRange[1],dblRange[2]

    
    
    property AOI:
        def __set__(self,value):
            ''' Set the area of interest
            XPos, YPos, Width, Height
            The AOI values are adjusted to fit the values allowed by the sensor
            and may not be equal to the given in 'value'
            '''
            
            
            cdef IS_RECT rectAOI
            rectAOI.s32X,rectAOI.s32Y,rectAOI.s32Width,rectAOI.s32Height = value
            
            ##Get the valid AOI params. If key error is raised, a valid entry in 
            ##the dict AOIinfo must be included
            try:
                MinW,MaxW,WStep,MinH,MaxH,HStep,PGHor,PGVer=AOIinfo[self.SensorID]
            except KeyError:
                #raise KeyError("The current camera is not defined in the AOIinfo dict")
                print "Pyueye Warning: The current camera is not defined in the AOIinfo dict."
                print "Using default AOI constraints. Hope they work."
                MinW,MaxW,WStep,MinH,MaxH,HStep,PGHor,PGVer = (32, 1280,4,4,1024,2,4,2)
            #Check values are on the correct grid positions
            rectAOI.s32X= <int>(rectAOI.s32X/PGHor)*PGHor
            rectAOI.s32Y= <int>(rectAOI.s32Y/PGVer)*PGVer
            
            #Check the correct values for width and height
            rectAOI.s32Width=<int>(rectAOI.s32Width/WStep)*WStep
            rectAOI.s32Height=<int>(rectAOI.s32Height/HStep)*HStep
            
            if rectAOI.s32Width < MinW:rectAOI.s32Width=MinW
            if rectAOI.s32Width > MaxW:rectAOI.s32Width=MaxW
            if rectAOI.s32Height < MinH:rectAOI.s32Height=MinH
            if rectAOI.s32Height > MaxH:rectAOI.s32Height=MaxH
            
            
            ##Config the AOI so it is shown in the absolute position
            #rectAOI.s32X|=c_IS_AOI_IMAGE_POS_ABSOLUTE
            #rectAOI.s32Y|=c_IS_AOI_IMAGE_POS_ABSOLUTE
            rv=is_AOI (self.cid, c_IS_AOI_IMAGE_SET_AOI, &rectAOI, sizeof(rectAOI))
            self.CheckNoSuccess(rv)
            ##Save the current AOI
            self.AOIx0,self.AOIy0,w,h=self.AOI
            self.AOIx1=self.AOIx0+w
            self.AOIy1=self.AOIy0+h
            
            if self.AOI != value: 
                print >> stderr, "The value passed to set the AOI is invalid for the current sensor."
                print >> stderr,"The current AOI value is {}".format(self.AOI)

        def __get__(self):
            cdef IS_RECT rectAOI           
            rv=is_AOI (self.cid, c_IS_AOI_IMAGE_GET_AOI, &rectAOI, sizeof(rectAOI))
            self.CheckNoSuccess(rv)
            return rectAOI.s32X,rectAOI.s32Y,rectAOI.s32Width,rectAOI.s32Height
            
    property FrameRate:
        def __set__(self,double value):
            
            cdef double nvalue
            rv=is_SetFrameRate(self.cid,value,&nvalue)
        
        
        def __get__(self):
            cdef double nvalue
            rv=is_SetFrameRate(self.cid,c_IS_GET_FRAMERATE,&nvalue)
            return nvalue 
            
    property FrameTimeRange:
        def __get__(self):
            '''Returns the frame rate settings
            
            Using FrameTimeRange(), you can read out the frame rate settings 
            which are available for the current pixel clock setting. 
            The returned values indicate the minimum and maximum frame duration 
            in seconds. You can set the frame duration between min and max 
            in increments defined by the intervall parameter.
        
            Syntax:
            =======
            
            min,max,intervall= cam.FrameTimeRange
               
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
    
    property Gamma:
        ''' Set gamma correction
        
        Gamma sets the value for digital gamma correction 
        (brighten dark image portions by applying a non-linear characteristic (LUT)). 
        Valid values are in the range between 0.01 and 10.
    
        
        GET_GAMMA: Returns the current setting.
        
        Return Value:
        =============
            SUCCESS: Function executed successfully
        
            Current setting when used together with GET_GAMMA        
        '''
        def __set__(self,double value):
 
            cdef int g=<int>(100*value)
            if g<0: g=1
            if g>1000: g=1000
            
            rv=is_SetGamma (self.cid, g)
            self.CheckNoSuccess(rv)
        
        def __get__(self):
            return is_SetGamma (self.cid, c_IS_GET_GAMMA)/100.
            
    def ReadEEPROM(self, raw=False):
        ''' Read the 64-byte user data EEPROM from the camera.

        Syntax:
        =======
        eeprom = cam.ReadEEPROM([raw])

        
        Input Parameters:
        =================
        raw:    True to return a 64-element list of the eeprom bytes.
                False (default) to return a friendly python 'bytes' string up to
                the first null-terminator.

        Return Value:
        =============
        eeprom: Depending on "raw" parameter, either a list or a 'bytes' string
                containing the user EEPROM.
        '''
        cdef char *c_eeprom = <char*>calloc( 64, sizeof(char) )
        cdef bytes py_eeprom
        rv = is_ReadEEPROM( self.cid, 0, c_eeprom, 64 )
        self.CheckNoSuccess(rv, "ReadEEPROM")
        if raw:
            raw_eeprom = [0]*64
            for i in range(64):
                raw_eeprom[i] = c_eeprom[i]
            return raw_eeprom
        else:
            py_eeprom = c_eeprom
            #stdlib.free(c_eeprom)
            return py_eeprom
    def WriteEEPROM(self, data, INT addr = 0, INT count = 0):
        ''' Write to the 64-byte user data EEPROM on the camera.

        Syntax:
        =======
        cam.WriteEEPROM(data, [addr])
        
        Input Parameters:
        =================
        data:   Either a 'bytes' string or a list of ints or chars.
        addr:   Start byte for write. 0 (default) to start at beginning.
        count:  Number of bytes to write. 0 (default) to write all of 'data'.
                If count > len(data), EEPROM is padded with zeroes. 
                Use 64 to ensure no remnants are left over.  

        Notes:
        =============
        If len(data) + addr is greater than 64, or if communication breaks down,
        or if data contains something other than chars and integers,
        an exception is thrown.
        If 'data' is a string, it will be zero-terminated in EEPROM.
        '''

        if count == 0:
            count = len(data)
            # Zero-terminate if it's a string:
            if type(data) == str:
                count += 1

        if count + addr > 64:
            raise Exception( "Attempted to write past end of 64-byte EEPROM." )

        cdef char *c_eeprom = <char*>calloc( 64, sizeof(char) )
        for i in range(count):
            if i < len(data):
                d = data[i]
                if type(d) == str:
                    d = ord(d)
            else:
                d = 0
            if type(d) != int:
                raise Exception( "Data must contain only chars or ints. data[%d] = %s" % (i, d) )
            c_eeprom[i] = d
        rv = is_WriteEEPROM( self.cid, addr, c_eeprom, count )
        self.CheckNoSuccess(rv, "writeEEPROM")
        
        
            
    def WaitEvent(self, INT which, INT timeout):
        ''' Wait for a uEye event.

        is_WaitEvent() allows waiting for uEye events. The function indicates 
        successful execution when the event has occurred within the specified 
        timeout.

        Note: Event must be enabled with EnableEvent first.

        Syntax:
        =======

        rv = cam.WaitEvent(which, timeout)

        Input Parameters:
        =================

        which:
            SET_EVENT_FRAME:  A new image is available.
            SET_EVENT_EXTTRIG:    An image which was captured following the 
                    arrival of a trigger has been transferred completely.
                    This is the earliest possible moment for a new capturing 
                    process. The image must then be post-processed by the 
                    driver and will be available after the IS_FRAME processing 
                    event.
            SET_EVENT_SEQ:  The sequence is completed.
            SET_EVENT_STEAL:  An image extracted from the overlay is available.
            SET_EVENT_CAPTURE_STATUS:  There is an information about image
                    capturing available. This information can be requested by
                    is_CaptureStatus().  Note that this event replaces the former
            SET_EVENT_TRANSFER_FAILED from previous versions.
            SET_EVENT_DEVICE_RECONNECTED:  A camera initialized with
                    is_InitCamera() and disconnected afterwards was reconnected.
            SET_EVENT_WB_FINISHED:  The automatic white balance control is
                    completed.
            SET_EVENT_AUTOBRIGHTNESS_FINISHED:  The automatic brightness
                    control in the run-once mode is completed.
            SET_EVENT_OVERLAY_DATA_LOST:  Direct3D/OpenGL mode: Because of a
                    re-programming the parameters of the overlay are invalid. The
                    overlay must be draw new.  
            SET_EVENT_REMOVE:  A camera initialized with is_InitCamera() was
                    disconnected.
            SET_EVENT_REMOVAL: A camera was removed.  This is independent of
                    the device handle (hCam is ignored).  
            SET_EVENT_NEW_DEVICE:  A new camera was connected.  This is
                    independent of the device handle (hCam is ignored).  
            SET_EVENT_STATUS_CHANGED:  The availability of a camera has
                    changed, e.g. an available camera was opened.

        Return Value:
        =============

        IS_SUCCESS: Function executed successfully
        IS_TIMED_OUT: Timeout occured before event arrived.

        An exception is thrown for the following internal return values:
            IS_NO_SUCCESS: General error message
        '''

        rv = is_WaitEvent(self.cid, which, timeout)
        if rv != c_IS_TIMED_OUT:
            self.CheckNoSuccess(rv, "WaitEvent")
        return rv

    def CaptureStatus(self, reset=False):
        '''Obtain or reset all uEye error counters

        The function returns information on errors that occurred during an
        image capture. All errors are listed that occurred since the last reset
        of the function.  

        Syntax:
        =======

        errorDict = cam.CaptureStatus([reset])

        
        Input Parameters:
        =================

        reset: If True, reset all counters to zero instead of returning them.


        Return Value:
        =============

        errorDict: Dictionary object with the following fields. Each field
                   represents a counter for that type of error.
                   See API documentation for possible causes/remedies.

            'API_NO_DEST_MEM'
                There is no destination memory for copying the finished image.
            'API_CONVERSION_FAILED'
                The current image could not be processed correctly.
            'API_IMAGE_LOCKED'
                The destination buffers are locked and could not be written to.
            'DRV_OUT_OF_BUFFERS'
                No free internal image memory is available to the driver. 
                The image was discarded.
            'DRV_DEVICE_NOT_READY'
                The camera is no longer available. It is not possible to access
                images that have already been transferred.
            'USB_TRANSFER_FAILED'
                The image was not transferred over the USB bus.
            'DEV_TIMEOUT'
                The maximum allowable time for image capturing in the camera 
                was exceeded.
                The selected timeout value is too low for image capture
            'ETH_BUFFER_OVERRUN'
                The sensor transfers more data than the internal camera memory 
                of the GigE uEye can accommodate.
            'ETH_MISSED_IMAGES'
                Freerun mode: The GigE uEye camera could neither process nor 
                output an image captured by the sensor.
                Hardware trigger mode: The GigE uEye camera received a hardware
                trigger signal which could not be processed because the sensor 
                was still busy.
        '''

        if reset:
            command = c_IS_CAPTURE_STATUS_INFO_CMD_RESET
        else:
            command = c_IS_CAPTURE_STATUS_INFO_CMD_GET

        cdef UEYE_CAPTURE_STATUS_INFO status
        rv = is_CaptureStatus(self.cid, command, &status, sizeof(status))
        self.CheckNoSuccess(rv)

        return {"Total": status.dwCapStatusCnt_Total,
            "API_NO_DEST_MEM": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_API_NO_DEST_MEM],
            "API_CONVERSION_FAILED": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_API_CONVERSION_FAILED],
            "API_IMAGE_LOCKED": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_API_IMAGE_LOCKED],
            "DRV_OUT_OF_BUFFERS": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_DRV_OUT_OF_BUFFERS],
            "DRV_DEVICE_NOT_READY": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_DRV_DEVICE_NOT_READY],
            "USB_TRANSFER_FAILED": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_USB_TRANSFER_FAILED],
            "DEV_TIMEOUT": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_DEV_TIMEOUT],
            "ETH_BUFFER_OVERRUN": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_ETH_BUFFER_OVERRUN],
            "ETH_MISSED_IMAGES": status.adwCapStatusCnt_Detail[c_IS_CAP_STATUS_ETH_MISSED_IMAGES],
            }

    cdef char * GetNextBuffer(self, timeout=1000):
        """ Get the next valid image buffer
        
            Returns NULL if timeout is reached.
        """
        
        cdef char * img
        cdef int rv
        cdef int tries=0
        
        # Wait for the event to come in:
        rv= is_WaitEvent(self.cid, c_IS_SET_EVENT_FRAME, timeout);
        if rv == c_IS_TIMED_OUT:
            #print "pyueye: GetNextBuffer's WaitEvent timed out."
            return NULL
        self.CheckNoSuccess(rv, "GetNextBuffer WaitEvent")

        # Grab the image:
        rv= is_GetImageMem(self.cid, <VOID**> &img)
        self.CheckNoSuccess(rv, "GetImageMem") 
        
        return img

    def GrabImage(self, BGR=False, UINT Timeout=500, LeaveLocked=False, char AOI=False):
        '''Grabs and reads an image from the camera and returns a numpy array

        By default, returns color images in RGB order for backwards compatibility.
        If working with OpenCV, you will most likely want to call with BGR=True.
        
        When using Live mode, it is highly recommended to use LeaveLocked=True,
        so that the frame you are using doesn't get overwritten by the driver.
        The frame will be unlocked on the next call to GrabImage, or by UnlockLastBuf.

        TODO: The AOI thing doesn't currently work because the cam.AOI property doesn't
              set absolute mode. Need to work out a way to do that.
        
        Syntax:
        =======
    
        im=cam.GrabImage([BGR,[Timeout,[LeaveLocked]]])
        
        Input Parameters:
        =================
        
        BGR:
            When True, color images will be retrieved in BGR order rather than RGB
            Default value is False (for backwards compatibility).

        Timeout:
            Max number of milliseconds to wait for a frame before throwing exception.
            Default value is 500.

        LeaveLocked:
            When True, the returned frame will be left locked so that the daemon
            won't overwrite the data in Live mode. It is recommended to use True
            if you are using Live mode. Default is False for back-compatibility.
            The buffer will be unlocked on the next call to GrabImage.
        AOI:
            When True, the returned data will have the AOI shape if not the whole
            image buffer will be returned. NOTE: Currently this does nothing.

        Return Value:
        =============
    
        im:
            Numpy array containing the image data. The data from the driver buffer
            to the numpy array. The returned numpy array is modified each time 
            the method is called. 
            If the request times out, it will return 0.
            
        Note: The default colormode for OpenCV is BGR, so it should be called with
              BGR=True. Default value is False only for backwards-compatibility.


        '''
        cdef npy.npy_intp dims3[3]
        
        # If we are supposed to be in Live Mode, make sure we still are:
        if (self.LiveMode and not self.IsLive()):
            print >> stderr, "Camera dropped out of Live mode. Re-starting..."
            self.CaptureVideo(c_IS_WAIT)

        # If we aren't in Live mode, kick off a single capture:
        if (not self.LiveMode):
            rv= is_FreezeVideo (self.cid, Timeout)
            self.CheckNoSuccess(rv)

        cdef char * img
        
        # Wait for new frame to come in:
        img=self.GetNextBuffer(Timeout)

        if img == NULL:
            # Must have timed out.
            return 0
        
        # Unlock previous buffer here, so there's no chance of overwrite.
        if self.LastSeqBufLocked:
            rv= is_UnlockSeqBuf(self.cid, c_IS_IGNORE_PARAMETER, self.LastSeqBuf)
            if rv != c_IS_SUCCESS:
                print >> stderr, "Buffer %d didn't unlock." % (<int>self.LastSeqBuf)
            self.LastSeqBufLocked=False

        # Create a numpy memory mapping to the frame:
        if self.colormode==c_IS_CM_RGB8_PACKED or self.colormode==c_IS_CM_BGR8_PACKED:
            dims3[0]=self.nMaxHeight
            dims3[1]=self.LineInc/3
            dims3[2]=3
            npy.Py_INCREF( npy.NPY_UINT8 )
            #data = npy.PyArray_SimpleNewFromData(3, dims3, npy.NPY_UINT8, self.Img)
            data = npy.PyArray_SimpleNewFromData(3, dims3, npy.NPY_UINT8, img)
            if BGR != (self.colormode == c_IS_CM_BGR8_PACKED):
                data=data[:,:,::-1]
        
        elif self.colormode==c_IS_CM_MONO8:
            dims3[0]=self.nMaxHeight
            dims3[1]=self.LineInc
            npy.Py_INCREF( npy.NPY_UINT8 )
            #data = npy.PyArray_SimpleNewFromData(2, dims3, npy.NPY_UINT8, self.Img)
            data = npy.PyArray_SimpleNewFromData(2, dims3, npy.NPY_UINT8, img)
            
        else:
            raise Exception("ColorFormat not suported")

        # Lock the buffer if requested:
        if LeaveLocked:
            rv= is_LockSeqBuf(self.cid, c_IS_IGNORE_PARAMETER, img)
            self.CheckNoSuccess(rv)
            self.LastSeqBufLocked=True
        
        self.LastSeqBuf = img
        
        #if AOI:
        #    return data[self.AOIy0:self.AOIy1,self.AOIx0:self.AOIx1]
        #else:
        return data

    
    def UnlockLastBuf(self):
        '''Unlocks the last-used buffer in the ring buffer

        This may be called when you are done working with the frame returned
        by GrabFrame(LeaveLocked=True), but it is not necessary since it will
        be automatically unlocked on the next call to GrabImage.

        Syntax:
        =======

        rv=cam.UnlockLastBuf()

        Return Value:
        =============

        SUCCESS (or an exception will be thrown)

        '''
        rv= is_UnlockSeqBuf(self.cid, c_IS_IGNORE_PARAMETER, self.LastSeqBuf)
        #if rv != IS_SUCCESS:
        #    print "Buffer %d didn't unlock." % (<int>self.LastSeqBuf)
        self.CheckNoSuccess(rv)

    
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
        
        rv=is_SetGlobalShutter (self.cid, mode)
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
            - Current setting when used together with GET_ROP_EFFECT
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
        if  rv!= c_IS_SUCCESS:
            raise Exception("Could not set timeout")
        return rv
            
    def IsLive (self):
        '''Determine whether camera is free-running or not

        Syntax:
        =======
        
        rv=cam.IsLive()
             

        Return Values:
        ==============
        rv:
            - TRUE if live capture is enabled

        '''
        return self.CaptureVideo(c_IS_GET_LIVE)

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
        if (Wait != c_IS_GET_LIVE):
            self.CheckNoSuccess(rv)
            self.LiveMode = True
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
        self.LiveMode = False
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

        The internal image ring buffer is also (re)initialized

        Syntax:
        =======
        
        rv=cam.SetColorMode (Mode)
               
        Input Parameters
        ================
        
        Mode: 
            Colour mode to be set
            - CM_MONO16           -       Greyscale (16)
            - CM_MONO12           -       Greyscale (12)
            - CM_MONO8            -       Greyscale (8)
            - CM_RGBA8_PACKED     -       RGB32 (8 8 8)
            - CM_RGBY8_PACKED     -       RGBY (8 8 8 8)
            - CM_RGB8_PACKED      -       RGB24 (8 8 8)
            - CM_BGRA8_PACKED     -       BGR32 (8 8 8)
            - CM_BGR8_PACKED      -       BGR24 (8 8 8)
            - CM_BGRY8_PACKED     -       BGRY (8 8 8)
            - CM_BGR565_PACKED    -       BGR16 (5 6 5)
            - CM_UYVY_PACKED      -       UYVY (8 8 8 8)
            - CM_UYVY_MONO_PACKED -       UYVY (8 8 8 8)
            - CM_UYVY_BAYER_PACKED-       UYVY (8 8 8 8)
            - CM_CBYCRY_PACKED    -       CbYCrY (8 8 8 8)
            - GET_COLOR_MODE      -       Returns the current setting.
            
        Return Values:
        ==============
        
        rv:
            if Mode != GET_COLORMODE returns Mode
            If Mode == GET_COLORMODE returns the current colormode
            
        Note:
        =====
        The self.Img memory block gets its size adjusted when the color mode is
        set.
        
        '''
        
        rv=is_SetColorMode (self.cid,  Mode)
        if (Mode==c_IS_GET_COLOR_MODE):
            return rv
        self.CheckNoSuccess(rv)

        # Save information relevant to the colormode
        self.colormode= is_SetColorMode(self.cid, c_IS_GET_COLOR_MODE)        
        self.bitspixel=bitspixel(self.colormode)
        
        if self.Imgs[0]!=NULL:
            for i in range(self.BufCount):
                rv=is_FreeImageMem (self.cid, self.Imgs[i], self.BufIds[i])
                self.CheckNoSuccess(rv)
                self.BufIds[i] = 0
        
        
        for i in range(self.BufCount):
            rv=is_AllocImageMem(self.cid, self.nMaxWidth, self.nMaxHeight, \
                                self.bitspixel, &self.Imgs[i], &self.BufIds[i])
            self.CheckNoSuccess(rv)
        
#~             rv=is_SetImageMem   (self.cid, self.Imgs[i], self.BufIds[i])
            rv=is_AddToSequence (self.cid, self.Imgs[i], self.BufIds[i])
            self.CheckNoSuccess(rv)
        
        # Initialize the queue so we can use the WaitForNextImage function
        #rv=is_InitImageQueue (self.cid, 0)
        #self.CheckNoSuccess(rv)
        
        # Get memory mapping information for later
        rv=is_GetImageMemPitch (self.cid, &self.LineInc)
        self.CheckNoSuccess(rv)
        
        return Mode
    
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
    
    cpdef CheckNoSuccess(self,INT rv, description=None):
        '''Method that checks the return value of a is_XXXX function.
        
        If rv != SUCCESS, the error mesage is printed and a exception is raised
        '''
        
        cdef char * ermsg
        #if rv==IS_NO_SUCCESS:
        if rv != c_IS_SUCCESS:
            if description:
                err = "Error in '%s' -- " % description
            else:
                err = "Error -- "

            err += "API call returned %d, " % rv
            rv1 = is_GetError (self.cid, &rv, &ermsg)
            if rv1 == 1:
                raise Exception(err + "Camera handle is not valid.")
            if rv1 != c_IS_SUCCESS:
                raise Exception(err + "but no error message available.")
            raise Exception(err + "'" + ermsg + "'")
            
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
        
    
###### Special methods
###### Methods that do not really belong in to the ueye API, but that are
###### Usefull.

    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def GrabImageA(self, unsigned char n, unsigned char cm=False,unsigned char threshold=0):
        """Grab an average of N images
         
            This is a modified copy of the GrabImage, that allows to get 
            an average of N frames.
            
            Input Parameters:
            
            n: 
                Number of images used in the averaging
            
            cm: (False/True) 
                If False, only the image is returned, if True, the image, plus
                the center of mass of the image are returned. Only the information
                contained in the AOI is used th calculate the center of mass.
            
            threshold:
                The pixels whos value is below the threshold are ignored in
                the average and in the center of mass calculations.
        """

        cdef int i,x,y,ya
        cdef double *avim, cmx=0,cmy=0,cmt=0
        cdef int w,h
        cdef double data
        w=self.AOIx1-self.AOIx0
        h=self.AOIy1-self.AOIy0
        #cdef npy.ndarray[npy.float_t, ndim = 2] av = npy.zeros((h,w), dtype = 'float',order='C')
        cdef npy.ndarray[npy.float_t, ndim = 2] av = zero_mat(h,w)
        
        cdef char * img
        
        if n==0: n=1;
        ##This seems to work fine, but when I try to exit video mode it frezes
        
#~          # If we are supposed to be in Live Mode, make sure we still are:
#~         if not self.IsLive():
#~             print >> stderr, "Camera dropped out of Live mode. Re-starting..."
#~             self.CaptureVideo(c_IS_WAIT)
        
        if self.colormode==c_IS_CM_MONO8:
            ##Grab first image
            rv= is_FreezeVideo (self.cid, c_IS_WAIT)
            self.CheckNoSuccess(rv)
            img=self.GetNextBuffer()
            
            ##The last grab is lost
            for i in range(n):
                ## Do the average while the next image is grabbed
                rv= is_FreezeVideo (self.cid, c_IS_WAIT)
                self.CheckNoSuccess(rv)
                for y in range(self.AOIy0,self.AOIy1):
                    ya=y*self.LineInc
                    for x in range(self.AOIx0,self.AOIx1):
                        data=(<unsigned char>img[x+ya])
                        if data<threshold: continue
                        av[y-self.AOIy0,x-self.AOIx0]+=data/n 
                        if cm:
                            cmt=cmt+data
                            cmx=cmx+data*x
                            cmy=cmy+data*y
                        
                img=self.GetNextBuffer()
            
        else: raise TypeError("Average not defined for this color mode")
        if cm:
            return av,(cmx/cmt, cmy/cmt)
        else:
            return av
            
    def AdjustExposure(self, limits=(200,240)):
        """Method to adjust the exposure time so that no point of the image
        is saturated. Only FrameRate is adjusted.
        
        It searches the exposure time so the maximum value is between the limits"""
        
        
        cdef double minexp,maxexp,step,expo
        cdef int maxdata,minlim,maxlim
        
        if self.colormode==c_IS_CM_MONO8:
            
            minlim,maxlim=limits
            minexp,maxexp,step=self.ExposureTimeRange
            
            li=minexp
            ls=maxexp
            lp=(li+ls)/2
            
            self.ExposureTime=li
            self.GrabImage(AOI=True)
            i=self.GrabImage(AOI=True).max()
            
            self.ExposureTime=ls
            self.GrabImage(AOI=True)
            s=self.GrabImage(AOI=True).max()
            
            self.ExposureTime=lp
            self.GrabImage(AOI=True)
            p=self.GrabImage(AOI=True).max()
            
            ## EXIT IF NOT POSSIBLE
            if i>maxlim: return False
            if s<minlim: return False 

            while True:
                if minlim<= p<= maxlim:return True
                if p< minlim:
                    li=lp
                if p> maxlim:
                    ls=lp
                lp=(li+ls)/2
                
                self.ExposureTime=lp
                self.GrabImage(AOI=True)
                p=self.GrabImage(AOI=True).max()
                if (ls-li)< 2*step: return True
                
        else:
            raise TypeError("This is defined only for monochrome cams")
            
