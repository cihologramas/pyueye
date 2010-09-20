# Copyright (c) 2010, Combustión Ingenieros Ltda.
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

# Author: Ricardo Amézquita Orozco <ramezquitao@cihologramas.com>

# This file was created initially using the information found in:
#*****************************************************************************
#*    file     uEye.h
#*    author   (c) 2004-2009 by Imaging Development Systems GmbH 
#*    homepage http://www.ids-imaging.com/
#*    date     2009/11/04 
#*    version  3.50.70 
#*
#*    brief   Library interface for IDS uEye - camera family.
#*            definition of exported API functions and constants
#***************************************************************************** 

# ----------------------------------------------------------------------------
# typedefs
# ----------------------------------------------------------------------------

cdef extern from "stdint.h":
    
        ctypedef signed int             BOOLEAN
        ctypedef signed int             BOOL
        ctypedef signed int             INT
        ctypedef unsigned int           UINT
        ctypedef signed int             LONG
        ctypedef void                   VOID
        ctypedef void*                  LPVOID
        ctypedef unsigned int           ULONG

        ctypedef unsigned long long     UINT64
        ctypedef signed long long       __int64
        ctypedef signed long long       LONGLONG
        ctypedef unsigned int           DWORD
        ctypedef unsigned short         WORD

        ctypedef unsigned char          BYTE
        ctypedef char                   CHAR
        ctypedef char                   TCHAR
        ctypedef unsigned char          UCHAR

        ctypedef signed char            LPTSTR
        #ctypedef const int8_t*     LPCTSTR
        #ctypedef const int8_t*     LPCSTR
        ctypedef unsigned int          WPARAM
        ctypedef unsigned int          LPARAM
        ctypedef unsigned int          LRESULT
        ctypedef unsigned int          HRESULT

        ctypedef void*             HWND
        ctypedef void*             HGLOBAL
        ctypedef void*             HINSTANCE
        ctypedef void*             HDC
        ctypedef void*             HMODULE
        ctypedef void*             HKEY
        ctypedef void*             HANDLE

        ctypedef BYTE*             LPBYTE
        ctypedef DWORD*            PDWORD
        ctypedef VOID*             PVOID
        ctypedef CHAR*             PCHAR

                

ctypedef    INT     IDSEXP
ctypedef    ULONG   IDSEXPUL 
ctypedef char IS_CHAR
ctypedef DWORD   HIDS
ctypedef DWORD   HCAM
ctypedef DWORD   HFALC





cdef extern from "uEye.h":

# ----------------------------------------------------------------------------
#Version Definition
# ----------------------------------------------------------------------------

        int UEYE_VERSION_CODE
        int IS_COLORMODE_INVALID

# ----------------------------------------------------------------------------
# Color modes
# ----------------------------------------------------------------------------
        int IS_COLORMODE_INVALID 
        int IS_COLORMODE_MONOCHROME             
        int IS_COLORMODE_BAYER                  

# ----------------------------------------------------------------------------
#  Sensor Types
# ----------------------------------------------------------------------------
        int IS_SENSOR_INVALID           

# CMOS Sensors
        int IS_SENSOR_UI141X_M                # VGA rolling shutter, monochrome
        int IS_SENSOR_UI141X_C                # VGA rolling shutter, color
        int IS_SENSOR_UI144X_M                # SXGA rolling shutter, monochrome
        int IS_SENSOR_UI144X_C                # SXGA rolling shutter, SXGA color

        int IS_SENSOR_UI154X_M                # SXGA rolling shutter, monochrome
        int IS_SENSOR_UI154X_C                # SXGA rolling shutter, color
        int IS_SENSOR_UI145X_C                # UXGA rolling shutter, color

        int IS_SENSOR_UI146X_C                # QXGA rolling shutter, color
        int IS_SENSOR_UI148X_M                # 5MP rolling shutter, mono
        int IS_SENSOR_UI148X_C                # 5MP rolling shutter, color

        int IS_SENSOR_UI121X_M                # VGA global shutter, monochrome
        int IS_SENSOR_UI121X_C                # VGA global shutter, VGA color
        int IS_SENSOR_UI122X_M                # WVGA global shutter, monochrome
        int IS_SENSOR_UI122X_C                # WVGA global shutter, color

        int IS_SENSOR_UI164X_C                # SXGA rolling shutter, color

        int IS_SENSOR_UI155X_C                # UXGA rolling shutter, color

        int IS_SENSOR_UI1223_M                # WVGA global shutter, monochrome
        int IS_SENSOR_UI1223_C                # WVGA global shutter, color

        int IS_SENSOR_UI149X_M                # 149x-M
        int IS_SENSOR_UI149X_C                # 149x-C

# LE models with xxx5
        int IS_SENSOR_UI1225_M                # WVGA global shutter, monochrome, LE model
        int IS_SENSOR_UI1225_C                # WVGA global shutter, color, LE model

        int IS_SENSOR_UI1645_C                # SXGA rolling shutter, color, LE model
        int IS_SENSOR_UI1555_C                # UXGA rolling shutter, color, LE model
        int IS_SENSOR_UI1545_M                # SXGA rolling shutter, monochrome, LE model
        int IS_SENSOR_UI1545_C                # SXGA rolling shutter, color, LE model
        int IS_SENSOR_UI1455_C                # UXGA rolling shutter, color, LE model
        int IS_SENSOR_UI1465_C                # QXGA rolling shutter, color, LE model
        int IS_SENSOR_UI1485_M                # 5MP rolling shutter, monochrome, LE model
        int IS_SENSOR_UI1485_C                # 5MP rolling shutter, color, LE model
        int IS_SENSOR_UI1495_M                # 149xLE-M
        int IS_SENSOR_UI1495_C                # 149xLE-C

# custom board level designs
        int IS_SENSOR_UI1543_M                # SXGA rolling shutter, monochrome, single board
        int IS_SENSOR_UI1543_C                # SXGA rolling shutter, color, single board

        int IS_SENSOR_UI1544_M                # SXGA rolling shutter, monochrome, single board
        int IS_SENSOR_UI1544_C                # SXGA rolling shutter, color, single board
        int IS_SENSOR_UI1543_M_WO             # SXGA rolling shutter, color, single board
        int IS_SENSOR_UI1543_C_WO             # SXGA rolling shutter, color, single board
        int IS_SENSOR_UI1453_C                # UXGA rolling shutter, color, single board
        int IS_SENSOR_UI1463_C                # QXGA rolling shutter, color, single board
        int IS_SENSOR_UI1483_C                # QSXGA rolling shutter, color, single board

        int IS_SENSOR_UI1463_M_WO             # QXGA rolling shutter, monochrome, single board
        int IS_SENSOR_UI1463_C_WO             # QXGA rolling shutter, color, single board

        int IS_SENSOR_UI1553_C_WN             # UXGA rolling shutter, color, single board
        int IS_SENSOR_UI1483_M_WO             # QSXGA rolling shutter, monochrome, single board
        int IS_SENSOR_UI1483_C_WO             # QSXGA rolling shutter, color, single board

# CCD Sensors
        int IS_SENSOR_UI223X_M                # Sony CCD sensor - XGA monochrome
        int IS_SENSOR_UI223X_C                # Sony CCD sensor - XGA color

        int IS_SENSOR_UI241X_M                # Sony CCD sensor - VGA monochrome
        int IS_SENSOR_UI241X_C                # Sony CCD sensor - VGA color

        int IS_SENSOR_UI234X_M                # Sony CCD sensor - SXGA monochrome
        int IS_SENSOR_UI234X_C                # Sony CCD sensor - SXGA color

        int IS_SENSOR_UI233X_M                # Kodak CCD sensor - 1MP mono
        int IS_SENSOR_UI233X_C                # Kodak CCD sensor - 1MP color

        int IS_SENSOR_UI221X_M                # Sony CCD sensor - VGA monochrome
        int IS_SENSOR_UI221X_C                # Sony CCD sensor - VGA color

        int IS_SENSOR_UI231X_M                # Sony CCD sensor - VGA monochrome
        int IS_SENSOR_UI231X_C                # Sony CCD sensor - VGA color

        int IS_SENSOR_UI222X_M                # Sony CCD sensor - CCIR / PAL monochrome
        int IS_SENSOR_UI222X_C                # Sony CCD sensor - CCIR / PAL color

        int IS_SENSOR_UI224X_M                # Sony CCD sensor - SXGA monochrome
        int IS_SENSOR_UI224X_C                # Sony CCD sensor - SXGA color

        int IS_SENSOR_UI225X_M                # Sony CCD sensor - UXGA monochrome
        int IS_SENSOR_UI225X_C                # Sony CCD sensor - UXGA color

        int IS_SENSOR_UI214X_M                # Sony CCD sensor - SXGA monochrome
        int IS_SENSOR_UI214X_C                # Sony CCD sensor - SXGA color

# ----------------------------------------------------------------------------
# error codes
# ----------------------------------------------------------------------------
        int IS_NO_SUCCESS                           # function call failed
        int IS_SUCCESS                              # function call succeeded
        int IS_INVALID_CAMERA_HANDLE                # camera handle is not valid or zero
        int IS_INVALID_HANDLE                       # a handle other than the camera handle is invalid

        int IS_IO_REQUEST_FAILED                     # an io request to the driver failed
        int IS_CANT_OPEN_DEVICE                      # returned by is_InitCamera
        int IS_CANT_CLOSE_DEVICE                  
        int IS_CANT_SETUP_MEMORY                  
        int IS_NO_HWND_FOR_ERROR_REPORT           
        int IS_ERROR_MESSAGE_NOT_CREATED          
        int IS_ERROR_STRING_NOT_FOUND             
        int IS_HOOK_NOT_CREATED                   
        int IS_TIMER_NOT_CREATED                 
        int IS_CANT_OPEN_REGISTRY                
        int IS_CANT_READ_REGISTRY                
        int IS_CANT_VALIDATE_BOARD               
        int IS_CANT_GIVE_BOARD_ACCESS            
        int IS_NO_IMAGE_MEM_ALLOCATED            
        int IS_CANT_CLEANUP_MEMORY               
        int IS_CANT_COMMUNICATE_WITH_DRIVER      
        int IS_FUNCTION_NOT_SUPPORTED_YET        
        int IS_OPERATING_SYSTEM_NOT_SUPPORTED    

        int IS_INVALID_VIDEO_IN                  
        int IS_INVALID_IMG_SIZE                  
        int IS_INVALID_ADDRESS                   
        int IS_INVALID_VIDEO_MODE                
        int IS_INVALID_AGC_MODE                  
        int IS_INVALID_GAMMA_MODE                
        int IS_INVALID_SYNC_LEVEL                
        int IS_INVALID_CBARS_MODE                
        int IS_INVALID_COLOR_MODE                
        int IS_INVALID_SCALE_FACTOR              
        int IS_INVALID_IMAGE_SIZE                
        int IS_INVALID_IMAGE_POS                 
        int IS_INVALID_CAPTURE_MODE              
        int IS_INVALID_RISC_PROGRAM              
        int IS_INVALID_BRIGHTNESS                
        int IS_INVALID_CONTRAST                  
        int IS_INVALID_SATURATION_U              
        int IS_INVALID_SATURATION_V              
        int IS_INVALID_HUE                       
        int IS_INVALID_HOR_FILTER_STEP           
        int IS_INVALID_VERT_FILTER_STEP          
        int IS_INVALID_EEPROM_READ_ADDRESS       
        int IS_INVALID_EEPROM_WRITE_ADDRESS      
        int IS_INVALID_EEPROM_READ_LENGTH        
        int IS_INVALID_EEPROM_WRITE_LENGTH       
        int IS_INVALID_BOARD_INFO_POINTER        
        int IS_INVALID_DISPLAY_MODE              
        int IS_INVALID_ERR_REP_MODE              
        int IS_INVALID_BITS_PIXEL                
        int IS_INVALID_MEMORY_POINTER            

        int IS_FILE_WRITE_OPEN_ERROR             
        int IS_FILE_READ_OPEN_ERROR              
        int IS_FILE_READ_INVALID_BMP_ID          
        int IS_FILE_READ_INVALID_BMP_SIZE        
        int IS_FILE_READ_INVALID_BIT_COUNT       
        int IS_WRONG_KERNEL_VERSION              

        int IS_RISC_INVALID_XLENGTH              
        int IS_RISC_INVALID_YLENGTH              
        int IS_RISC_EXCEED_IMG_SIZE              

# DirectDraw Mode errors
        int IS_DD_MAIN_FAILED                    
        int IS_DD_PRIMSURFACE_FAILED             
        int IS_DD_SCRN_SIZE_NOT_SUPPORTED        
        int IS_DD_CLIPPER_FAILED                 
        int IS_DD_CLIPPER_HWND_FAILED            
        int IS_DD_CLIPPER_CONNECT_FAILED         
        int IS_DD_BACKSURFACE_FAILED             
        int IS_DD_BACKSURFACE_IN_SYSMEM          
        int IS_DD_MDL_MALLOC_ERR                 
        int IS_DD_MDL_SIZE_ERR                   
        int IS_DD_CLIP_NO_CHANGE                 
        int IS_DD_PRIMMEM_NULL                   
        int IS_DD_BACKMEM_NULL                   
        int IS_DD_BACKOVLMEM_NULL                
        int IS_DD_OVERLAYSURFACE_FAILED          
        int IS_DD_OVERLAYSURFACE_IN_SYSMEM       
        int IS_DD_OVERLAY_NOT_ALLOWED            
        int IS_DD_OVERLAY_COLKEY_ERR             
        int IS_DD_OVERLAY_NOT_ENABLED            
        int IS_DD_GET_DC_ERROR                   
        int IS_DD_DDRAW_DLL_NOT_LOADED           
        int IS_DD_THREAD_NOT_CREATED             
        int IS_DD_CANT_GET_CAPS                  
        int IS_DD_NO_OVERLAYSURFACE              
        int IS_DD_NO_OVERLAYSTRETCH              
        int IS_DD_CANT_CREATE_OVERLAYSURFACE     
        int IS_DD_CANT_UPDATE_OVERLAYSURFACE     
        int IS_DD_INVALID_STRETCH                

        int IS_EV_INVALID_EVENT_NUMBER          
        int IS_INVALID_MODE                     
        int IS_CANT_FIND_FALCHOOK               
        int IS_CANT_FIND_HOOK                   
        int IS_CANT_GET_HOOK_PROC_ADDR          
        int IS_CANT_CHAIN_HOOK_PROC             
        int IS_CANT_SETUP_WND_PROC              
        int IS_HWND_NULL                        
        int IS_INVALID_UPDATE_MODE              
        int IS_NO_ACTIVE_IMG_MEM                
        int IS_CANT_INIT_EVENT                  
        int IS_FUNC_NOT_AVAIL_IN_OS             
        int IS_CAMERA_NOT_CONNECTED             
        int IS_SEQUENCE_LIST_EMPTY              
        int IS_CANT_ADD_TO_SEQUENCE             
        int IS_LOW_OF_SEQUENCE_RISC_MEM         
        int IS_IMGMEM2FREE_USED_IN_SEQ          
        int IS_IMGMEM_NOT_IN_SEQUENCE_LIST      
        int IS_SEQUENCE_BUF_ALREADY_LOCKED      
        int IS_INVALID_DEVICE_ID                
        int IS_INVALID_BOARD_ID                 
        int IS_ALL_DEVICES_BUSY                 
        int IS_HOOK_BUSY                        
        int IS_TIMED_OUT                        
        int IS_NULL_POINTER                     
        int IS_WRONG_HOOK_VERSION               
        int IS_INVALID_PARAMETER                   # a parameter specified was invalid
        int IS_NOT_ALLOWED                      
        int IS_OUT_OF_MEMORY                    
        int IS_INVALID_WHILE_LIVE               
        int IS_ACCESS_VIOLATION                    # an internal exception occurred
        int IS_UNKNOWN_ROP_EFFECT               
        int IS_INVALID_RENDER_MODE              
        int IS_INVALID_THREAD_CONTEXT           
        int IS_NO_HARDWARE_INSTALLED            
        int IS_INVALID_WATCHDOG_TIME            
        int IS_INVALID_WATCHDOG_MODE            
        int IS_INVALID_PASSTHROUGH_IN           
        int IS_ERROR_SETTING_PASSTHROUGH_IN     
        int IS_FAILURE_ON_SETTING_WATCHDOG      
        int IS_NO_USB20                            # the usb port doesnt support usb 2.0
        int IS_CAPTURE_RUNNING                     # there is already a capture running

        int IS_MEMORY_BOARD_ACTIVATED              # operation could not execute while mboard is enabled
        int IS_MEMORY_BOARD_DEACTIVATED            # operation could not execute while mboard is disabled
        int IS_NO_MEMORY_BOARD_CONNECTED           # no memory board connected
        int IS_TOO_LESS_MEMORY                     # image size is above memory capacity
        int IS_IMAGE_NOT_PRESENT                   # requested image is no longer present in the camera
        int IS_MEMORY_MODE_RUNNING              
        int IS_MEMORYBOARD_DISABLED             

        int IS_TRIGGER_ACTIVATED                   # operation could not execute while trigger is enabled
        int IS_WRONG_KEY                        
        int IS_CRC_ERROR                        
        int IS_NOT_YET_RELEASED                    # this feature is not available yet
        int IS_NOT_CALIBRATED                      # the camera is not calibrated
        int IS_WAITING_FOR_KERNEL                  # a request to the kernel exceeded
        int IS_NOT_SUPPORTED                       # operation mode is not supported
        int IS_TRIGGER_NOT_ACTIVATED               # operation could not execute while trigger is disabled
        int IS_OPERATION_ABORTED                
        int IS_BAD_STRUCTURE_SIZE               
        int IS_INVALID_BUFFER_SIZE              
        int IS_INVALID_PIXEL_CLOCK              
        int IS_INVALID_EXPOSURE_TIME            
        int IS_AUTO_EXPOSURE_RUNNING            
        int IS_CANNOT_CREATE_BB_SURF               # error creating backbuffer surface  
        int IS_CANNOT_CREATE_BB_MIX                # backbuffer mixer surfaces can not be created
        int IS_BB_OVLMEM_NULL                      # backbuffer overlay mem could not be locked  
        int IS_CANNOT_CREATE_BB_OVL                # backbuffer overlay mem could not be created  
        int IS_NOT_SUPP_IN_OVL_SURF_MODE           # function not supported in overlay surface mode  
        int IS_INVALID_SURFACE                     # surface invalid
        int IS_SURFACE_LOST                        # surface has been lost  
        int IS_RELEASE_BB_OVL_DC                   # error releasing backbuffer overlay DC  
        int IS_BB_TIMER_NOT_CREATED                # backbuffer timer could not be created  
        int IS_BB_OVL_NOT_EN                       # backbuffer overlay has not been enabled  
        int IS_ONLY_IN_BB_MODE                     # only possible in backbuffer mode 
        int IS_INVALID_COLOR_FORMAT                # invalid color format
        int IS_INVALID_WB_BINNING_MODE             # invalid binning mode for AWB 
        int IS_INVALID_I2C_DEVICE_ADDRESS          # invalid I2C device address
        int IS_COULD_NOT_CONVERT                   # current image couldn't be converted
        int IS_TRANSFER_ERROR                      # transfer failed
        int IS_PARAMETER_SET_NOT_PRESENT           # the parameter set is not present
        int IS_INVALID_CAMERA_TYPE                 # the camera type in the ini file doesn't match
        int IS_INVALID_HOST_IP_HIBYTE              # HIBYTE of host address is invalid
        int IS_CM_NOT_SUPP_IN_CURR_DISPLAYMODE     # color mode is not supported in the current display mode
        int IS_NO_IR_FILTER                     
        int IS_STARTER_FW_UPLOAD_NEEDED            # device starter firmware is not compatible    

        int IS_DR_LIBRARY_NOT_FOUND                # the DirectRender library could not be found
        int IS_DR_DEVICE_OUT_OF_MEMORY             # insufficient graphics adapter video memory
        int IS_DR_CANNOT_CREATE_SURFACE            # the image or overlay surface could not be created
        int IS_DR_CANNOT_CREATE_VERTEX_BUFFER      # the vertex buffer could not be created
        int IS_DR_CANNOT_CREATE_TEXTURE            # the texture could not be created  
        int IS_DR_CANNOT_LOCK_OVERLAY_SURFACE      # the overlay surface could not be locked
        int IS_DR_CANNOT_UNLOCK_OVERLAY_SURFACE    # the overlay surface could not be unlocked
        int IS_DR_CANNOT_GET_OVERLAY_DC            # cannot get the overlay surface DC 
        int IS_DR_CANNOT_RELEASE_OVERLAY_DC        # cannot release the overlay surface DC
        int IS_DR_DEVICE_CAPS_INSUFFICIENT         # insufficient graphics adapter capabilities

# ----------------------------------------------------------------------------
# common definitions
# ----------------------------------------------------------------------------
        int IS_OFF                              
        int IS_ON                               
        int IS_IGNORE_PARAMETER                 


# ----------------------------------------------------------------------------
#  device enumeration
# ----------------------------------------------------------------------------
        int IS_USE_DEVICE_ID                    
        int IS_ALLOW_STARTER_FW_UPLOAD          

# ----------------------------------------------------------------------------
# AutoExit enable/disable
# ----------------------------------------------------------------------------
        int IS_GET_AUTO_EXIT_ENABLED            
        int IS_DISABLE_AUTO_EXIT                
        int IS_ENABLE_AUTO_EXIT                 


# ----------------------------------------------------------------------------
# live/freeze parameters
# ----------------------------------------------------------------------------
        int IS_GET_LIVE                         

        int IS_WAIT                             
        int IS_DONT_WAIT                        
        int IS_FORCE_VIDEO_STOP                 
        int IS_FORCE_VIDEO_START                
        int IS_USE_NEXT_MEM                     


# ----------------------------------------------------------------------------
# video finish constants
# ----------------------------------------------------------------------------
        int IS_VIDEO_NOT_FINISH                 
        int IS_VIDEO_FINISH                     


# ----------------------------------------------------------------------------
# bitmap render modes
# ----------------------------------------------------------------------------
        int IS_GET_RENDER_MODE                  

        int IS_RENDER_DISABLED                  
        int IS_RENDER_NORMAL                    
        int IS_RENDER_FIT_TO_WINDOW             
        int IS_RENDER_DOWNSCALE_1_2             
        int IS_RENDER_MIRROR_UPDOWN             
        int IS_RENDER_DOUBLE_HEIGHT             
        int IS_RENDER_HALF_HEIGHT               


# ----------------------------------------------------------------------------
# external trigger modes
# ----------------------------------------------------------------------------
        int IS_GET_EXTERNALTRIGGER              
        int IS_GET_TRIGGER_STATUS               
        int IS_GET_TRIGGER_MASK                 
        int IS_GET_TRIGGER_INPUTS               
        int IS_GET_SUPPORTED_TRIGGER_MODE       
        int IS_GET_TRIGGER_COUNTER              

# old defines for compatibility 
        int IS_SET_TRIG_OFF                     
        int IS_SET_TRIG_HI_LO                   
        int IS_SET_TRIG_LO_HI                   
        int IS_SET_TRIG_SOFTWARE                
        int IS_SET_TRIG_HI_LO_SYNC              
        int IS_SET_TRIG_LO_HI_SYNC              

        int IS_SET_TRIG_MASK                    

# New defines
        int IS_SET_TRIGGER_CONTINUOUS           
        int IS_SET_TRIGGER_OFF                  
        int IS_SET_TRIGGER_HI_LO                 
        int IS_SET_TRIGGER_LO_HI                 
        int IS_SET_TRIGGER_SOFTWARE              
        int IS_SET_TRIGGER_HI_LO_SYNC           
        int IS_SET_TRIGGER_LO_HI_SYNC           


        int IS_GET_TRIGGER_DELAY                
        int IS_GET_MIN_TRIGGER_DELAY            
        int IS_GET_MAX_TRIGGER_DELAY            
        int IS_GET_TRIGGER_DELAY_GRANULARITY    


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
# pixelclock
        int IS_GET_PIXEL_CLOCK                  
        int IS_GET_DEFAULT_PIXEL_CLK            
# frame rate
        int IS_GET_FRAMERATE                    
        int IS_GET_DEFAULT_FRAMERATE            
# exposure
        int IS_GET_EXPOSURE_TIME                
        int IS_GET_DEFAULT_EXPOSURE             

# ----------------------------------------------------------------------------
# Gain definitions
# ----------------------------------------------------------------------------
        int IS_GET_MASTER_GAIN                  
        int IS_GET_RED_GAIN                     
        int IS_GET_GREEN_GAIN                   
        int IS_GET_BLUE_GAIN                    
        int IS_GET_DEFAULT_MASTER               
        int IS_GET_DEFAULT_RED                  
        int IS_GET_DEFAULT_GREEN                
        int IS_GET_DEFAULT_BLUE                 
        int IS_GET_GAINBOOST                    
        int IS_SET_GAINBOOST_ON                 
        int IS_SET_GAINBOOST_OFF                
        int IS_GET_SUPPORTED_GAINBOOST          
        int IS_MIN_GAIN                         
        int IS_MAX_GAIN                         


# ----------------------------------------------------------------------------
# Gain factor definitions
# ----------------------------------------------------------------------------
        int IS_GET_MASTER_GAIN_FACTOR           
        int IS_GET_RED_GAIN_FACTOR              
        int IS_GET_GREEN_GAIN_FACTOR            
        int IS_GET_BLUE_GAIN_FACTOR             
        int IS_SET_MASTER_GAIN_FACTOR           
        int IS_SET_RED_GAIN_FACTOR              
        int IS_SET_GREEN_GAIN_FACTOR            
        int IS_SET_BLUE_GAIN_FACTOR             
        int IS_GET_DEFAULT_MASTER_GAIN_FACTOR   
        int IS_GET_DEFAULT_RED_GAIN_FACTOR      
        int IS_GET_DEFAULT_GREEN_GAIN_FACTOR    
        int IS_GET_DEFAULT_BLUE_GAIN_FACTOR     
        int IS_INQUIRE_MASTER_GAIN_FACTOR       
        int IS_INQUIRE_RED_GAIN_FACTOR          
        int IS_INQUIRE_GREEN_GAIN_FACTOR        
        int IS_INQUIRE_BLUE_GAIN_FACTOR         


# ----------------------------------------------------------------------------
# Global Shutter definitions
# ----------------------------------------------------------------------------
        int IS_SET_GLOBAL_SHUTTER_ON            
        int IS_SET_GLOBAL_SHUTTER_OFF           
        int IS_GET_GLOBAL_SHUTTER               
        int IS_GET_SUPPORTED_GLOBAL_SHUTTER     


# ----------------------------------------------------------------------------
# Black level definitions
# ----------------------------------------------------------------------------
        int IS_GET_BL_COMPENSATION              
        int IS_GET_BL_OFFSET                    
        int IS_GET_BL_DEFAULT_MODE              
        int IS_GET_BL_DEFAULT_OFFSET            
        int IS_GET_BL_SUPPORTED_MODE            

        int IS_BL_COMPENSATION_DISABLE          
        int IS_BL_COMPENSATION_ENABLE           
        int IS_BL_COMPENSATION_OFFSET           

        int IS_MIN_BL_OFFSET                    
        int IS_MAX_BL_OFFSET                    

# ----------------------------------------------------------------------------
# hardware gamma definitions
# ----------------------------------------------------------------------------
        int IS_GET_HW_GAMMA                     
        int IS_GET_HW_SUPPORTED_GAMMA           

        int IS_SET_HW_GAMMA_OFF                 
        int IS_SET_HW_GAMMA_ON                  

# ----------------------------------------------------------------------------
# camera LUT
# ----------------------------------------------------------------------------
        int IS_ENABLE_CAMERA_LUT                
        int IS_SET_CAMERA_LUT_VALUES            
        int IS_ENABLE_RGB_GRAYSCALE             
        int IS_GET_CAMERA_LUT_USER              
        int IS_GET_CAMERA_LUT_COMPLETE          

# ----------------------------------------------------------------------------
# camera LUT presets
# ----------------------------------------------------------------------------
        int IS_CAMERA_LUT_IDENTITY              
        int IS_CAMERA_LUT_NEGATIV               
        int IS_CAMERA_LUT_GLOW1                 
        int IS_CAMERA_LUT_GLOW2                 
        int IS_CAMERA_LUT_ASTRO1                
        int IS_CAMERA_LUT_RAINBOW1              
        int IS_CAMERA_LUT_MAP1                  
        int IS_CAMERA_LUT_COLD_HOT              
        int IS_CAMERA_LUT_SEPIC                 
        int IS_CAMERA_LUT_ONLY_RED              
        int IS_CAMERA_LUT_ONLY_GREEN            
        int IS_CAMERA_LUT_ONLY_BLUE             

        int IS_CAMERA_LUT_64                    
        int IS_CAMERA_LUT_128                   


# ----------------------------------------------------------------------------
# image parameters
# ----------------------------------------------------------------------------
# brightness
        int IS_GET_BRIGHTNESS                   
        int IS_MIN_BRIGHTNESS                   
        int IS_MAX_BRIGHTNESS                   
        int IS_DEFAULT_BRIGHTNESS               
# contrast
        int IS_GET_CONTRAST                     
        int IS_MIN_CONTRAST                     
        int IS_MAX_CONTRAST                     
        int IS_DEFAULT_CONTRAST                 
# gamma
        int IS_GET_GAMMA                        
        int IS_MIN_GAMMA                        
        int IS_MAX_GAMMA                        
        int IS_DEFAULT_GAMMA                    
# saturation   (Falcon)
        int IS_GET_SATURATION_U                 
        int IS_MIN_SATURATION_U                 
        int IS_MAX_SATURATION_U                 
        int IS_DEFAULT_SATURATION_U             
        int IS_GET_SATURATION_V                 
        int IS_MIN_SATURATION_V                 
        int IS_MAX_SATURATION_V                 
        int IS_DEFAULT_SATURATION_V             
# hue  (Falcon)
        int IS_GET_HUE                          
        int IS_MIN_HUE                          
        int IS_MAX_HUE                          
        int IS_DEFAULT_HUE                      


# ----------------------------------------------------------------------------
# Image position and size
# ----------------------------------------------------------------------------
        int IS_GET_IMAGE_SIZE_X                 
        int IS_GET_IMAGE_SIZE_Y                 
        int IS_GET_IMAGE_SIZE_X_INC             
        int IS_GET_IMAGE_SIZE_Y_INC             
        int IS_GET_IMAGE_SIZE_X_MIN             
        int IS_GET_IMAGE_SIZE_Y_MIN             
        int IS_GET_IMAGE_SIZE_X_MAX             
        int IS_GET_IMAGE_SIZE_Y_MAX             

        int IS_GET_IMAGE_POS_X                  
        int IS_GET_IMAGE_POS_Y                  
        int IS_GET_IMAGE_POS_X_ABS              
        int IS_GET_IMAGE_POS_Y_ABS              
        int IS_GET_IMAGE_POS_X_INC              
        int IS_GET_IMAGE_POS_Y_INC              
        int IS_GET_IMAGE_POS_X_MIN              
        int IS_GET_IMAGE_POS_Y_MIN              
        int IS_GET_IMAGE_POS_X_MAX              
        int IS_GET_IMAGE_POS_Y_MAX              

        int IS_SET_IMAGE_POS_X_ABS              
        int IS_SET_IMAGE_POS_Y_ABS              

# Compatibility
        int IS_SET_IMAGEPOS_X_ABS               
        int IS_SET_IMAGEPOS_Y_ABS               


# ----------------------------------------------------------------------------
# ROP effect constants
# ----------------------------------------------------------------------------
        int IS_GET_ROP_EFFECT                   
        int IS_GET_SUPPORTED_ROP_EFFECT         

        int IS_SET_ROP_NONE                     
        int IS_SET_ROP_MIRROR_UPDOWN            
        int IS_SET_ROP_MIRROR_UPDOWN_ODD        
        int IS_SET_ROP_MIRROR_UPDOWN_EVEN       
        int IS_SET_ROP_MIRROR_LEFTRIGHT         


# ----------------------------------------------------------------------------
# Subsampling
# ----------------------------------------------------------------------------
        int IS_GET_SUBSAMPLING                      
        int IS_GET_SUPPORTED_SUBSAMPLING            
        int IS_GET_SUBSAMPLING_TYPE                 
        int IS_GET_SUBSAMPLING_FACTOR_HORIZONTAL    
        int IS_GET_SUBSAMPLING_FACTOR_VERTICAL      

        int IS_SUBSAMPLING_DISABLE                  

        int IS_SUBSAMPLING_2X_VERTICAL              
        int IS_SUBSAMPLING_2X_HORIZONTAL            
        int IS_SUBSAMPLING_4X_VERTICAL              
        int IS_SUBSAMPLING_4X_HORIZONTAL            
        int IS_SUBSAMPLING_3X_VERTICAL              
        int IS_SUBSAMPLING_3X_HORIZONTAL            
        int IS_SUBSAMPLING_5X_VERTICAL              
        int IS_SUBSAMPLING_5X_HORIZONTAL            
        int IS_SUBSAMPLING_6X_VERTICAL              
        int IS_SUBSAMPLING_6X_HORIZONTAL            
        int IS_SUBSAMPLING_8X_VERTICAL              
        int IS_SUBSAMPLING_8X_HORIZONTAL            
        int IS_SUBSAMPLING_16X_VERTICAL             
        int IS_SUBSAMPLING_16X_HORIZONTAL           

        int IS_SUBSAMPLING_COLOR                    
        int IS_SUBSAMPLING_MONO                     

        int IS_SUBSAMPLING_MASK_VERTICAL            
        int IS_SUBSAMPLING_MASK_HORIZONTAL          

# Compatibility
        int IS_SUBSAMPLING_VERT                     
        int IS_SUBSAMPLING_HOR                      


# ----------------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------------
        int IS_GET_BINNING                      
        int IS_GET_SUPPORTED_BINNING            
        int IS_GET_BINNING_TYPE                 
        int IS_GET_BINNING_FACTOR_HORIZONTAL    
        int IS_GET_BINNING_FACTOR_VERTICAL      

        int IS_BINNING_DISABLE                  

        int IS_BINNING_2X_VERTICAL              
        int IS_BINNING_2X_HORIZONTAL            
        int IS_BINNING_4X_VERTICAL              
        int IS_BINNING_4X_HORIZONTAL            
        int IS_BINNING_3X_VERTICAL              
        int IS_BINNING_3X_HORIZONTAL            
        int IS_BINNING_5X_VERTICAL              
        int IS_BINNING_5X_HORIZONTAL            
        int IS_BINNING_6X_VERTICAL              
        int IS_BINNING_6X_HORIZONTAL            
        int IS_BINNING_8X_VERTICAL              
        int IS_BINNING_8X_HORIZONTAL            
        int IS_BINNING_16X_VERTICAL             
        int IS_BINNING_16X_HORIZONTAL           

        int IS_BINNING_COLOR                    
        int IS_BINNING_MONO                     

        int IS_BINNING_MASK_VERTICAL            
        int IS_BINNING_MASK_HORIZONTAL          

# Compatibility
        int IS_BINNING_VERT                     
        int IS_BINNING_HOR                      

# ----------------------------------------------------------------------------
# Auto Control Parameter
# ----------------------------------------------------------------------------
        int IS_SET_ENABLE_AUTO_GAIN             
        int IS_GET_ENABLE_AUTO_GAIN             
        int IS_SET_ENABLE_AUTO_SHUTTER          
        int IS_GET_ENABLE_AUTO_SHUTTER          
        int IS_SET_ENABLE_AUTO_WHITEBALANCE     
        int IS_GET_ENABLE_AUTO_WHITEBALANCE     
        int IS_SET_ENABLE_AUTO_FRAMERATE        
        int IS_GET_ENABLE_AUTO_FRAMERATE        
        int IS_SET_ENABLE_AUTO_SENSOR_GAIN      
        int IS_GET_ENABLE_AUTO_SENSOR_GAIN      
        int IS_SET_ENABLE_AUTO_SENSOR_SHUTTER   
        int IS_GET_ENABLE_AUTO_SENSOR_SHUTTER   
        int IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER  
        int IS_GET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER  
        int IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE     
        int IS_GET_ENABLE_AUTO_SENSOR_FRAMERATE     

        int IS_SET_AUTO_REFERENCE               
        int IS_GET_AUTO_REFERENCE               
        int IS_SET_AUTO_GAIN_MAX                
        int IS_GET_AUTO_GAIN_MAX                
        int IS_SET_AUTO_SHUTTER_MAX             
        int IS_GET_AUTO_SHUTTER_MAX             
        int IS_SET_AUTO_SPEED                   
        int IS_GET_AUTO_SPEED                   
        int IS_SET_AUTO_WB_OFFSET               
        int IS_GET_AUTO_WB_OFFSET               
        int IS_SET_AUTO_WB_GAIN_RANGE           
        int IS_GET_AUTO_WB_GAIN_RANGE           
        int IS_SET_AUTO_WB_SPEED                
        int IS_GET_AUTO_WB_SPEED                
        int IS_SET_AUTO_WB_ONCE                 
        int IS_GET_AUTO_WB_ONCE                 
        int IS_SET_AUTO_BRIGHTNESS_ONCE         
        int IS_GET_AUTO_BRIGHTNESS_ONCE         
        int IS_SET_AUTO_HYSTERESIS              
        int IS_GET_AUTO_HYSTERESIS              
        int IS_GET_AUTO_HYSTERESIS_RANGE        
        int IS_SET_AUTO_WB_HYSTERESIS           
        int IS_GET_AUTO_WB_HYSTERESIS           
        int IS_GET_AUTO_WB_HYSTERESIS_RANGE     
        int IS_SET_AUTO_SKIPFRAMES              
        int IS_GET_AUTO_SKIPFRAMES              
        int IS_GET_AUTO_SKIPFRAMES_RANGE        
        int IS_SET_AUTO_WB_SKIPFRAMES           
        int IS_GET_AUTO_WB_SKIPFRAMES           
        int IS_GET_AUTO_WB_SKIPFRAMES_RANGE     

# ----------------------------------------------------------------------------
# Auto Control definitions
# ----------------------------------------------------------------------------
        int IS_MIN_AUTO_BRIGHT_REFERENCE        
        int IS_MAX_AUTO_BRIGHT_REFERENCE        
        int IS_DEFAULT_AUTO_BRIGHT_REFERENCE    
        int IS_MIN_AUTO_SPEED                   
        int IS_MAX_AUTO_SPEED                   
        int IS_DEFAULT_AUTO_SPEED               

        int IS_DEFAULT_AUTO_WB_OFFSET           
        int IS_MIN_AUTO_WB_OFFSET               
        int IS_MAX_AUTO_WB_OFFSET               
        int IS_DEFAULT_AUTO_WB_SPEED            
        int IS_MIN_AUTO_WB_SPEED                
        int IS_MAX_AUTO_WB_SPEED                
        int IS_MIN_AUTO_WB_REFERENCE            
        int IS_MAX_AUTO_WB_REFERENCE            


# ----------------------------------------------------------------------------
# AOI types to set/get
# ----------------------------------------------------------------------------
        int IS_SET_AUTO_BRIGHT_AOI              
        int IS_GET_AUTO_BRIGHT_AOI              
        int IS_SET_IMAGE_AOI                    
        int IS_GET_IMAGE_AOI                    
        int IS_SET_AUTO_WB_AOI                  
        int IS_GET_AUTO_WB_AOI                  


# ----------------------------------------------------------------------------
# color modes
# ----------------------------------------------------------------------------
        int IS_GET_COLOR_MODE                   

        int IS_SET_CM_RGB32                     
        int IS_SET_CM_RGB24                     
        int IS_SET_CM_RGB16                     
        int IS_SET_CM_RGB15                     
        int IS_SET_CM_Y8                        
        int IS_SET_CM_RGB8                      
        int IS_SET_CM_BAYER                     
        int IS_SET_CM_UYVY                      
        int IS_SET_CM_UYVY_MONO                 
        int IS_SET_CM_UYVY_BAYER                
        int IS_SET_CM_CBYCRY                    

        int IS_SET_CM_RGBY                      
        int IS_SET_CM_RGB30                     
        int IS_SET_CM_Y12                       
        int IS_SET_CM_BAYER12                   
        int IS_SET_CM_Y16                       
        int IS_SET_CM_BAYER16                   

        int IS_CM_MODE_MASK                     

# planar vs packed format
        int IS_CM_FORMAT_PACKED                 
        int IS_CM_FORMAT_PLANAR                 
        int IS_CM_FORMAT_MASK                   

# BGR vs. RGB order
        int IS_CM_ORDER_BGR                     
        int IS_CM_ORDER_RGB                     
        int IS_CM_ORDER_MASK                     


# define compliant color format names
        int IS_CM_MONO8                 # occupies 8 Bit
        int IS_CM_MONO12                # occupies 16 Bit
        int IS_CM_MONO16                # occupies 16 Bit

        int IS_CM_BAYER_RG8             # occupies 8 Bit
        int IS_CM_BAYER_RG12            # occupies 16 Bit
        int IS_CM_BAYER_RG16            # occupies 16 Bit

        int IS_CM_BGR555_PACKED         # occupies 16 Bit
        int IS_CM_BGR565_PACKED         # occupies 16 Bit 

        int IS_CM_RGB8_PACKED           # occupies 24 Bit
        int IS_CM_BGR8_PACKED           # occupies 24 Bit  
        int IS_CM_RGBA8_PACKED          # occupies 32 Bit
        int IS_CM_BGRA8_PACKED          # occupies 32 Bit
        int IS_CM_RGBY8_PACKED          # occupies 32 Bit
        int IS_CM_BGRY8_PACKED          # occupies 32 Bit
        int IS_CM_RGB10V2_PACKED        # occupies 32 Bit
        int IS_CM_BGR10V2_PACKED        # occupies 32 Bit

        int IS_CM_YUV422_PACKED         # no compliant version      
        int IS_CM_UYVY_PACKED           # occupies 16 Bit
        int IS_CM_UYVY_MONO_PACKED      
        int IS_CM_UYVY_BAYER_PACKED     
        int IS_CM_CBYCRY_PACKED         # occupies 16 Bit

        int IS_CM_RGB8_PLANAR           
        int IS_CM_RGB12_PLANAR          
        int IS_CM_RGB16_PLANAR          


        int IS_CM_ALL_POSSIBLE                  

# ----------------------------------------------------------------------------
# Hotpixel correction
# ----------------------------------------------------------------------------
        int IS_GET_BPC_MODE                      
        int IS_GET_BPC_THRESHOLD                 
        int IS_GET_BPC_SUPPORTED_MODE            

        int IS_BPC_DISABLE                       
        int IS_BPC_ENABLE_LEVEL_1                
        int IS_BPC_ENABLE_LEVEL_2                
        int IS_BPC_ENABLE_USER                   
        int IS_BPC_ENABLE_SOFTWARE          
        int IS_BPC_ENABLE_HARDWARE          

        int IS_SET_BADPIXEL_LIST                 
        int IS_GET_BADPIXEL_LIST                 
        int IS_GET_LIST_SIZE                     


# ----------------------------------------------------------------------------
# color correction definitions
# ----------------------------------------------------------------------------
        int IS_GET_CCOR_MODE                    
        int IS_GET_SUPPORTED_CCOR_MODE          
        int IS_GET_DEFAULT_CCOR_MODE            
        int IS_GET_CCOR_FACTOR                  
        int IS_GET_CCOR_FACTOR_MIN              
        int IS_GET_CCOR_FACTOR_MAX              
        int IS_GET_CCOR_FACTOR_DEFAULT          

        int IS_CCOR_DISABLE                     
        int IS_CCOR_ENABLE                      
        int IS_CCOR_ENABLE_NORMAL           
        int IS_CCOR_ENABLE_BG40_ENHANCED        
        int IS_CCOR_ENABLE_HQ_ENHANCED          
        int IS_CCOR_SET_IR_AUTOMATIC            
        int IS_CCOR_FACTOR                      

        int IS_CCOR_ENABLE_MASK             


# ----------------------------------------------------------------------------
# bayer algorithm modes
# ----------------------------------------------------------------------------
        int IS_GET_BAYER_CV_MODE                

        int IS_SET_BAYER_CV_NORMAL              
        int IS_SET_BAYER_CV_BETTER              
        int IS_SET_BAYER_CV_BEST                


# ----------------------------------------------------------------------------
# color converter modes
# ----------------------------------------------------------------------------
        int IS_CONV_MODE_NONE                   
        int IS_CONV_MODE_SOFTWARE               
        int IS_CONV_MODE_SOFTWARE_3X3           
        int IS_CONV_MODE_SOFTWARE_5X5           
        int IS_CONV_MODE_HARDWARE_3X3           


# ----------------------------------------------------------------------------
# Edge enhancement
# ----------------------------------------------------------------------------
        int IS_GET_EDGE_ENHANCEMENT             

        int IS_EDGE_EN_DISABLE                  
        int IS_EDGE_EN_STRONG                   
        int IS_EDGE_EN_WEAK                     


# ----------------------------------------------------------------------------
# white balance modes
# ----------------------------------------------------------------------------
        int IS_GET_WB_MODE                      

        int IS_SET_WB_DISABLE                   
        int IS_SET_WB_USER                      
        int IS_SET_WB_AUTO_ENABLE               
        int IS_SET_WB_AUTO_ENABLE_ONCE          

        int IS_SET_WB_DAYLIGHT_65               
        int IS_SET_WB_COOL_WHITE                
        int IS_SET_WB_U30                       
        int IS_SET_WB_ILLUMINANT_A              
        int IS_SET_WB_HORIZON                   


# ----------------------------------------------------------------------------
# flash strobe constants
# ----------------------------------------------------------------------------
        int IS_GET_FLASHSTROBE_MODE             
        int IS_GET_FLASHSTROBE_LINE             
        int IS_GET_SUPPORTED_FLASH_IO_PORTS     

        int IS_SET_FLASH_OFF                    
        int IS_SET_FLASH_ON                     
        int IS_SET_FLASH_LO_ACTIVE          
        int IS_SET_FLASH_HI_ACTIVE              
        int IS_SET_FLASH_HIGH                   
        int IS_SET_FLASH_LOW                    
        int IS_SET_FLASH_LO_ACTIVE_FREERUN      
        int IS_SET_FLASH_HI_ACTIVE_FREERUN      
        int IS_SET_FLASH_IO_1                   
        int IS_SET_FLASH_IO_2                   
        int IS_SET_FLASH_IO_3                   
        int IS_SET_FLASH_IO_4                   
        int IS_FLASH_IO_PORT_MASK             

        int IS_GET_FLASH_DELAY                  
        int IS_GET_FLASH_DURATION               
        int IS_GET_MAX_FLASH_DELAY              
        int IS_GET_MAX_FLASH_DURATION           
        int IS_GET_MIN_FLASH_DELAY              
        int IS_GET_MIN_FLASH_DURATION           
        int IS_GET_FLASH_DELAY_GRANULARITY      
        int IS_GET_FLASH_DURATION_GRANULARITY   

# ----------------------------------------------------------------------------
# Digital IO constants
# ----------------------------------------------------------------------------
        int IS_GET_IO                           
        int IS_GET_IO_MASK                      
        int IS_GET_INPUT_MASK                   
        int IS_GET_OUTPUT_MASK                  
        int IS_GET_SUPPORTED_IO_PORTS           


# ----------------------------------------------------------------------------
# EEPROM defines
# ----------------------------------------------------------------------------
        int IS_EEPROM_MIN_USER_ADDRESS          
        int IS_EEPROM_MAX_USER_ADDRESS          
        int IS_EEPROM_MAX_USER_SPACE            


# ----------------------------------------------------------------------------
# error report modes
# ----------------------------------------------------------------------------
        int IS_GET_ERR_REP_MODE                 
        int IS_ENABLE_ERR_REP                   
        int IS_DISABLE_ERR_REP                  


# ----------------------------------------------------------------------------
# display mode selectors
# ----------------------------------------------------------------------------
        int IS_GET_DISPLAY_MODE                 
        int IS_GET_DISPLAY_SIZE_X               
        int IS_GET_DISPLAY_SIZE_Y               
        int IS_GET_DISPLAY_POS_X                
        int IS_GET_DISPLAY_POS_Y                

        int IS_SET_DM_DIB                       
        int IS_SET_DM_DIRECTDRAW                
        int IS_SET_DM_DIRECT3D                  
        int IS_SET_DM_ALLOW_SYSMEM              
        int IS_SET_DM_ALLOW_PRIMARY             

# -- overlay display mode ---
        int IS_GET_DD_OVERLAY_SCALE             

        int IS_SET_DM_ALLOW_OVERLAY             
        int IS_SET_DM_ALLOW_SCALING             
        int IS_SET_DM_ALLOW_FIELDSKIP           
        int IS_SET_DM_MONO                      
        int IS_SET_DM_BAYER                     
        int IS_SET_DM_YCBCR                     

# -- backbuffer display mode ---
        int IS_SET_DM_BACKBUFFER                


# ----------------------------------------------------------------------------
# DirectRenderer commands
# ----------------------------------------------------------------------------
        int DR_GET_OVERLAY_DC                       
        int DR_GET_MAX_OVERLAY_SIZE                 
        int DR_GET_OVERLAY_KEY_COLOR                
        int DR_RELEASE_OVERLAY_DC                   
        int DR_SHOW_OVERLAY                                  
        int DR_HIDE_OVERLAY                                        
        int DR_SET_OVERLAY_SIZE                                            
        int DR_SET_OVERLAY_POSITION                     
        int DR_SET_OVERLAY_KEY_COLOR                 
        int DR_SET_HWND                              
        int DR_ENABLE_SCALING                       
        int DR_DISABLE_SCALING                      
        int DR_CLEAR_OVERLAY                        
        int DR_ENABLE_SEMI_TRANSPARENT_OVERLAY      
        int DR_DISABLE_SEMI_TRANSPARENT_OVERLAY     
        int DR_CHECK_COMPATIBILITY                  
        int DR_SET_VSYNC_OFF                        
        int DR_SET_VSYNC_AUTO                       
        int DR_SET_USER_SYNC                        
        int DR_GET_USER_SYNC_POSITION_RANGE         
        int DR_LOAD_OVERLAY_FROM_FILE               
        int DR_STEAL_NEXT_FRAME                     
        int DR_SET_STEAL_FORMAT                     
        int DR_GET_STEAL_FORMAT                     
        int DR_ENABLE_IMAGE_SCALING                 
        int DR_GET_OVERLAY_SIZE                     

# ----------------------------------------------------------------------------
# DirectDraw keying color constants
# ----------------------------------------------------------------------------
        int IS_GET_KC_RED                       
        int IS_GET_KC_GREEN                     
        int IS_GET_KC_BLUE                      
        int IS_GET_KC_RGB                       
        int IS_GET_KC_INDEX                     
        int IS_GET_KEYOFFSET_X                  
        int IS_GET_KEYOFFSET_Y                  

# RGB-triple for default key-color in 15,16,24,32 bit mode
        int IS_SET_KC_DEFAULT                      # 0xbbggrr
# color index for default key-color in 8bit palette mode
        int IS_SET_KC_DEFAULT_8                 


# ----------------------------------------------------------------------------
# Memoryboard
# ----------------------------------------------------------------------------
        int IS_MEMORY_GET_COUNT                 
        int IS_MEMORY_GET_DELAY                 
        int IS_MEMORY_MODE_DISABLE              
        int IS_MEMORY_USE_TRIGGER               


# ----------------------------------------------------------------------------
# Test image modes
# ----------------------------------------------------------------------------
        int IS_GET_TEST_IMAGE                   

        int IS_SET_TEST_IMAGE_DISABLED          
        int IS_SET_TEST_IMAGE_MEMORY_1          
        int IS_SET_TEST_IMAGE_MEMORY_2          
        int IS_SET_TEST_IMAGE_MEMORY_3          


# ----------------------------------------------------------------------------
# Led settings
# ----------------------------------------------------------------------------
        int IS_SET_LED_OFF                      
        int IS_SET_LED_ON                       
        int IS_SET_LED_TOGGLE                   
        int IS_GET_LED                          


# ----------------------------------------------------------------------------
# save options
# ----------------------------------------------------------------------------
        int IS_SAVE_USE_ACTUAL_IMAGE_SIZE       

# ----------------------------------------------------------------------------
# renumeration modes
# ----------------------------------------------------------------------------
        int IS_RENUM_BY_CAMERA                  
        int IS_RENUM_BY_HOST                    

# ----------------------------------------------------------------------------
# event constants
# ----------------------------------------------------------------------------
        int IS_SET_EVENT_ODD                    
        int IS_SET_EVENT_EVEN                   
        int IS_SET_EVENT_FRAME                  
        int IS_SET_EVENT_EXTTRIG                
        int IS_SET_EVENT_VSYNC                  
        int IS_SET_EVENT_SEQ                    
        int IS_SET_EVENT_STEAL                  
        int IS_SET_EVENT_VPRES                  
        int IS_SET_EVENT_TRANSFER_FAILED        
        int IS_SET_EVENT_DEVICE_RECONNECTED     
        int IS_SET_EVENT_MEMORY_MODE_FINISH     
        int IS_SET_EVENT_FRAME_RECEIVED         
        int IS_SET_EVENT_WB_FINISHED            
        int IS_SET_EVENT_AUTOBRIGHTNESS_FINISHED 

        int IS_SET_EVENT_REMOVE                 
        int IS_SET_EVENT_REMOVAL                
        int IS_SET_EVENT_NEW_DEVICE             
        int IS_SET_EVENT_STATUS_CHANGED         


# ----------------------------------------------------------------------------
# Window message defines
# ----------------------------------------------------------------------------
        int IS_UEYE_MESSAGE                      
        int IS_FRAME                          
        int IS_SEQUENCE                       
        int IS_TRIGGER                        
        int IS_TRANSFER_FAILED                
        int IS_DEVICE_RECONNECTED             
        int IS_MEMORY_MODE_FINISH             
        int IS_FRAME_RECEIVED                 
        int IS_GENERIC_ERROR                  
        int IS_STEAL_VIDEO                    
        int IS_WB_FINISHED                    
        int IS_AUTOBRIGHTNESS_FINISHED        

        int IS_DEVICE_REMOVED                 
        int IS_DEVICE_REMOVAL                 
        int IS_NEW_DEVICE                     
        int IS_DEVICE_STATUS_CHANGED          


# ----------------------------------------------------------------------------
# camera id constants
# ----------------------------------------------------------------------------
        int IS_GET_CAMERA_ID                    


# ----------------------------------------------------------------------------
# camera info constants
# ----------------------------------------------------------------------------
        int IS_GET_STATUS                       

        int IS_EXT_TRIGGER_EVENT_CNT            
        int IS_FIFO_OVR_CNT                     
        int IS_SEQUENCE_CNT                     
        int IS_LAST_FRAME_FIFO_OVR              
        int IS_SEQUENCE_SIZE                    
        int IS_VIDEO_PRESENT                    
        int IS_STEAL_FINISHED                   
        int IS_STORE_FILE_PATH                  
        int IS_LUMA_BANDWIDTH_FILTER            
        int IS_BOARD_REVISION                   
        int IS_MIRROR_BITMAP_UPDOWN             
        int IS_BUS_OVR_CNT                      
        int IS_STEAL_ERROR_CNT                  
        int IS_LOW_COLOR_REMOVAL                
        int IS_CHROMA_COMB_FILTER               
        int IS_CHROMA_AGC                       
        int IS_WATCHDOG_ON_BOARD                
        int IS_PASSTHROUGH_ON_BOARD             
        int IS_EXTERNAL_VREF_MODE               
        int IS_WAIT_TIMEOUT                     
        int IS_TRIGGER_MISSED                   
        int IS_LAST_CAPTURE_ERROR               
        int IS_PARAMETER_SET_1                  
        int IS_PARAMETER_SET_2                  
        int IS_STANDBY                          
        int IS_STANDBY_SUPPORTED                
        int IS_QUEUED_IMAGE_EVENT_CNT           

# ----------------------------------------------------------------------------
# interface type defines
# ----------------------------------------------------------------------------
        int IS_INTERFACE_TYPE_USB               
        int IS_INTERFACE_TYPE_ETH               

# ----------------------------------------------------------------------------
# board type defines
# ----------------------------------------------------------------------------
        int IS_BOARD_TYPE_FALCON                
        int IS_BOARD_TYPE_EAGLE                 
        int IS_BOARD_TYPE_FALCON2               
        int IS_BOARD_TYPE_FALCON_PLUS           
        int IS_BOARD_TYPE_FALCON_QUATTRO        
        int IS_BOARD_TYPE_FALCON_DUO            
        int IS_BOARD_TYPE_EAGLE_QUATTRO         
        int IS_BOARD_TYPE_EAGLE_DUO             
        int IS_BOARD_TYPE_UEYE_USB                  # 0x40
        int IS_BOARD_TYPE_UEYE_USB_SE               # 0x40
        int IS_BOARD_TYPE_UEYE_USB_RE               # 0x40
        int IS_BOARD_TYPE_UEYE_USB_ME               # 0x41
        int IS_BOARD_TYPE_UEYE_USB_LE               # 0x42
        int IS_BOARD_TYPE_UEYE_ETH                  # 0x80
        int IS_BOARD_TYPE_UEYE_ETH_HE               # 0x80
        int IS_BOARD_TYPE_UEYE_ETH_SE               # 0x81
        int IS_BOARD_TYPE_UEYE_ETH_RE               # 0x81

# ----------------------------------------------------------------------------
# camera type defines
# ----------------------------------------------------------------------------
        int IS_CAMERA_TYPE_UEYE_USB         
        int IS_CAMERA_TYPE_UEYE_USB_SE      
        int IS_CAMERA_TYPE_UEYE_USB_RE      
        int IS_CAMERA_TYPE_UEYE_USB_ME      
        int IS_CAMERA_TYPE_UEYE_USB_LE      
        int IS_CAMERA_TYPE_UEYE_ETH         
        int IS_CAMERA_TYPE_UEYE_ETH_HE      
        int IS_CAMERA_TYPE_UEYE_ETH_SE      
        int IS_CAMERA_TYPE_UEYE_ETH_RE      

# ----------------------------------------------------------------------------
# readable operation system defines
# ----------------------------------------------------------------------------
        int IS_OS_UNDETERMINED                  
        int IS_OS_WIN95                         
        int IS_OS_WINNT40                       
        int IS_OS_WIN98                         
        int IS_OS_WIN2000                       
        int IS_OS_WINXP                         
        int IS_OS_WINME                         
        int IS_OS_WINNET                        
        int IS_OS_WINSERVER2003                 
        int IS_OS_WINVISTA                      
        int IS_OS_LINUX24                       
        int IS_OS_LINUX26                       
        int IS_OS_WIN7                          


# ----------------------------------------------------------------------------
# Bus speed
# ----------------------------------------------------------------------------
        int IS_USB_10                            #  1,5 Mb/s
        int IS_USB_11                            #   12 Mb/s
        int IS_USB_20                            #  480 Mb/s
        int IS_USB_30                            # 5000 Mb/s
        int IS_ETHERNET_10                       #   10 Mb/s
        int IS_ETHERNET_100                      #  100 Mb/s
        int IS_ETHERNET_1000                     # 1000 Mb/s
        int IS_ETHERNET_10000                    #10000 Mb/s

        int IS_USB_LOW_SPEED                    
        int IS_USB_FULL_SPEED                   
        int IS_USB_HIGH_SPEED                   
        int IS_USB_SUPER_SPEED                  
        int IS_ETHERNET_10Base                  
        int IS_ETHERNET_100Base                 
        int IS_ETHERNET_1000Base                
        int IS_ETHERNET_10GBase                 

# ----------------------------------------------------------------------------
# HDR
# ----------------------------------------------------------------------------
        int IS_HDR_NOT_SUPPORTED                
        int IS_HDR_KNEEPOINTS                   
        int IS_DISABLE_HDR                      
        int IS_ENABLE_HDR                       


# ----------------------------------------------------------------------------
# Test images
# ----------------------------------------------------------------------------
        int IS_TEST_IMAGE_NONE                          
        int IS_TEST_IMAGE_WHITE                         
        int IS_TEST_IMAGE_BLACK                         
        int IS_TEST_IMAGE_HORIZONTAL_GREYSCALE          
        int IS_TEST_IMAGE_VERTICAL_GREYSCALE            
        int IS_TEST_IMAGE_DIAGONAL_GREYSCALE            
        int IS_TEST_IMAGE_WEDGE_GRAY                    
        int IS_TEST_IMAGE_WEDGE_COLOR                   
        int IS_TEST_IMAGE_ANIMATED_WEDGE_GRAY           

        int IS_TEST_IMAGE_ANIMATED_WEDGE_COLOR          
        int IS_TEST_IMAGE_MONO_BARS                     
        int IS_TEST_IMAGE_COLOR_BARS1                   
        int IS_TEST_IMAGE_COLOR_BARS2                   
        int IS_TEST_IMAGE_GREYSCALE1                    
        int IS_TEST_IMAGE_GREY_AND_COLOR_BARS           
        int IS_TEST_IMAGE_MOVING_GREY_AND_COLOR_BARS    
        int IS_TEST_IMAGE_ANIMATED_LINE                 

        int IS_TEST_IMAGE_ALTERNATE_PATTERN             
        int IS_TEST_IMAGE_VARIABLE_GREY                 
        int IS_TEST_IMAGE_MONOCHROME_HORIZONTAL_BARS    
        int IS_TEST_IMAGE_MONOCHROME_VERTICAL_BARS      
        int IS_TEST_IMAGE_CURSOR_H                      
        int IS_TEST_IMAGE_CURSOR_V                      
        int IS_TEST_IMAGE_COLDPIXEL_GRID                
        int IS_TEST_IMAGE_HOTPIXEL_GRID                 

        int IS_TEST_IMAGE_VARIABLE_RED_PART             
        int IS_TEST_IMAGE_VARIABLE_GREEN_PART           
        int IS_TEST_IMAGE_VARIABLE_BLUE_PART            
        int IS_TEST_IMAGE_SHADING_IMAGE                 
#                                                  0x10000000
#                                                  0x20000000
#                                                  0x40000000
#                                                  0x80000000


# ----------------------------------------------------------------------------
# Sensor scaler
# ----------------------------------------------------------------------------
        int IS_ENABLE_SENSOR_SCALER             
        int IS_ENABLE_ANTI_ALIASING             


# ----------------------------------------------------------------------------
# Timeouts
# ----------------------------------------------------------------------------
        int IS_TRIGGER_TIMEOUT                  


# ----------------------------------------------------------------------------
# Auto pixel clock modes
# ----------------------------------------------------------------------------
        int IS_BEST_PCLK_RUN_ONCE               

# ----------------------------------------------------------------------------
# sequence flags
# ----------------------------------------------------------------------------
        int IS_LOCK_LAST_BUFFER                 

# ----------------------------------------------------------------------------
# Image files types
# ----------------------------------------------------------------------------
        int IS_IMG_BMP                          
        int IS_IMG_JPG                          
        int IS_IMG_PNG                          
        int IS_IMG_RAW                          
        int IS_IMG_TIF                          

# ----------------------------------------------------------------------------
# I2C defines
# nRegisterAddr | IS_I2C_16_BIT_REGISTER
# ----------------------------------------------------------------------------
        int IS_I2C_16_BIT_REGISTER              

# ----------------------------------------------------------------------------
# DirectDraw steal video constants   (Falcon)
# ----------------------------------------------------------------------------
        int IS_INIT_STEAL_VIDEO                 
        int IS_EXIT_STEAL_VIDEO                 
        int IS_INIT_STEAL_VIDEO_MANUAL          
        int IS_INIT_STEAL_VIDEO_AUTO            
        int IS_SET_STEAL_RATIO                  
        int IS_USE_MEM_IMAGE_SIZE               
        int IS_STEAL_MODES_MASK                 
        int IS_SET_STEAL_COPY                   
        int IS_SET_STEAL_NORMAL                 

# ----------------------------------------------------------------------------
# AGC modes   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_AGC_MODE                     
        int IS_SET_AGC_OFF                      
        int IS_SET_AGC_ON                       


# ----------------------------------------------------------------------------
# Gamma modes   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_GAMMA_MODE                   
        int IS_SET_GAMMA_OFF                    
        int IS_SET_GAMMA_ON                     


# ----------------------------------------------------------------------------
# sync levels   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_SYNC_LEVEL                   
        int IS_SET_SYNC_75                      
        int IS_SET_SYNC_125                     


# ----------------------------------------------------------------------------
# color bar modes   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_CBARS_MODE                   
        int IS_SET_CBARS_OFF                    
        int IS_SET_CBARS_ON                     


# ----------------------------------------------------------------------------
# horizontal filter defines   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_HOR_FILTER_MODE              
        int IS_GET_HOR_FILTER_STEP              

        int IS_DISABLE_HOR_FILTER               
        int IS_ENABLE_HOR_FILTER                
        #int IS_HOR_FILTER_STEP(_s_)         ((_s_ + 1) << 1)
        int IS_HOR_FILTER_STEP1                 
        int IS_HOR_FILTER_STEP2                 
        int IS_HOR_FILTER_STEP3                 


# ----------------------------------------------------------------------------
# vertical filter defines   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_VERT_FILTER_MODE             
        int IS_GET_VERT_FILTER_STEP             

        int IS_DISABLE_VERT_FILTER              
        int IS_ENABLE_VERT_FILTER               
        #int IS_VERT_FILTER_STEP(_s_)        ((_s_ + 1) << 1)
        int IS_VERT_FILTER_STEP1                
        int IS_VERT_FILTER_STEP2                
        int IS_VERT_FILTER_STEP3                


# ----------------------------------------------------------------------------
# scaler modes   (Falcon)
# ----------------------------------------------------------------------------
        float IS_GET_SCALER_MODE          
        float IS_SET_SCALER_OFF           
        float IS_SET_SCALER_ON            

        float IS_MIN_SCALE_X              
        float IS_MAX_SCALE_X              
        float IS_MIN_SCALE_Y              
        float IS_MAX_SCALE_Y              


# ----------------------------------------------------------------------------
# video source selectors   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_VIDEO_IN                     
        int IS_GET_VIDEO_PASSTHROUGH            
        int IS_GET_VIDEO_IN_TOGGLE              
        int IS_GET_TOGGLE_INPUT_1               
        int IS_GET_TOGGLE_INPUT_2               
        int IS_GET_TOGGLE_INPUT_3               
        int IS_GET_TOGGLE_INPUT_4               

        int IS_SET_VIDEO_IN_1                   
        int IS_SET_VIDEO_IN_2                   
        int IS_SET_VIDEO_IN_S                   
        int IS_SET_VIDEO_IN_3                   
        int IS_SET_VIDEO_IN_4                   
        int IS_SET_VIDEO_IN_1S                  
        int IS_SET_VIDEO_IN_2S                  
        int IS_SET_VIDEO_IN_3S                  
        int IS_SET_VIDEO_IN_4S                  
        int IS_SET_VIDEO_IN_EXT                 
        int IS_SET_TOGGLE_OFF                   
        int IS_SET_VIDEO_IN_SYNC                


# ----------------------------------------------------------------------------
# video crossbar selectors   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_CROSSBAR                     

        int IS_CROSSBAR_1                       
        int IS_CROSSBAR_2                       
        int IS_CROSSBAR_3                       
        int IS_CROSSBAR_4                       
        int IS_CROSSBAR_5                       
        int IS_CROSSBAR_6                       
        int IS_CROSSBAR_7                       
        int IS_CROSSBAR_8                       
        int IS_CROSSBAR_9                       
        int IS_CROSSBAR_10                      
        int IS_CROSSBAR_11                      
        int IS_CROSSBAR_12                      
        int IS_CROSSBAR_13                      
        int IS_CROSSBAR_14                      
        int IS_CROSSBAR_15                      
        int IS_CROSSBAR_16                      
        int IS_SELECT_AS_INPUT                  


# ----------------------------------------------------------------------------
# video format selectors   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_VIDEO_MODE                   

        int IS_SET_VM_PAL                       
        int IS_SET_VM_NTSC                      
        int IS_SET_VM_SECAM                     
        int IS_SET_VM_AUTO                      


# ----------------------------------------------------------------------------
# capture modes   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_CAPTURE_MODE                 

        int IS_SET_CM_ODD                       
        int IS_SET_CM_EVEN                      
        int IS_SET_CM_FRAME                     
        int IS_SET_CM_NONINTERLACED             
        int IS_SET_CM_NEXT_FRAME                
        int IS_SET_CM_NEXT_FIELD                
        int IS_SET_CM_BOTHFIELDS            
        int IS_SET_CM_FRAME_STEREO              


# ----------------------------------------------------------------------------
# display update mode constants   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_UPDATE_MODE                  
        int IS_SET_UPDATE_TIMER                 
        int IS_SET_UPDATE_EVENT                 


# ----------------------------------------------------------------------------
# sync generator mode constants   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_SYNC_GEN                     
        int IS_SET_SYNC_GEN_OFF                 
        int IS_SET_SYNC_GEN_ON                  


# ----------------------------------------------------------------------------
# decimation modes   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_DECIMATION_MODE              
        int IS_GET_DECIMATION_NUMBER            

        int IS_DECIMATION_OFF                   
        int IS_DECIMATION_CONSECUTIVE           
        int IS_DECIMATION_DISTRIBUTED           
        

# ----------------------------------------------------------------------------
# hardware watchdog defines   (Falcon)
# ----------------------------------------------------------------------------
        int IS_GET_WATCHDOG_TIME                
        int IS_GET_WATCHDOG_RESOLUTION          
        int IS_GET_WATCHDOG_ENABLE              

        int IS_WATCHDOG_MINUTES                 
        int IS_WATCHDOG_SECONDS                 
        int IS_DISABLE_WATCHDOG                 
        int IS_ENABLE_WATCHDOG                  
        int IS_RETRIGGER_WATCHDOG               
        int IS_ENABLE_AUTO_DEACTIVATION         
        int IS_DISABLE_AUTO_DEACTIVATION        
        int IS_WATCHDOG_RESERVED                

       

# ----------------------------------------------------------------------------
# invalid values for device handles
# ----------------------------------------------------------------------------
#~ #define IS_INVALID_HIDS (HIDS)0
#~ #define IS_INVALID_HCAM (HIDS)0
#~ #define IS_INVALID_HFALC (HIDS)0
#~ 
#~ 
# ----------------------------------------------------------------------------
# info struct
# ----------------------------------------------------------------------------
#~ #define FALCINFO   BOARDINFO
#~ #define PFALCINFO  PBOARDINFO
#~ #define CAMINFO    BOARDINFO
#~ #define PCAMINFO   PBOARDINFO
#~ 
        struct _BOARDINFO:
            char          SerNo[12]          # e.g. "1234512345"  (11 char)
            char          ID[20]             # e.g. "IDS GmbH"
            char          Version[10]        # e.g. "V2.10"  (9 char)
            char          Date[12]           # e.g. "24.01.2006" (11 char)
            unsigned char Select             # contains board select number for multi board support
            unsigned char Type               # e.g. IS_BOARD_TYPE_UEYE_USB
            char          Reserved[8]        # (7 char)
            
        ctypedef _BOARDINFO BOARDINFO
        ctypedef BOARDINFO * PBOARDINFO
        ctypedef _BOARDINFO FALCINFO
        ctypedef FALCINFO * PFALCINFO
        ctypedef _BOARDINFO CAMINFO
        ctypedef CAMINFO * PCAMINFO
 
 
        struct _SENSORINFO:
            WORD          SensorID           # e.g. IS_SENSOR_UI224X_C
            IS_CHAR       strSensorName[32]  # e.g. "UI-224X-C"
            char          nColorMode         # e.g. IS_COLORMODE_BAYER
            DWORD         nMaxWidth          # e.g. 1280
            DWORD         nMaxHeight         # e.g. 1024
            BOOL          bMasterGain        # e.g. TRUE
            BOOL          bRGain             # e.g. TRUE
            BOOL          bGGain             # e.g. TRUE
            BOOL          bBGain             # e.g. TRUE
            BOOL          bGlobShutter       # e.g. TRUE
        ctypedef _SENSORINFO SENSORINFO
        ctypedef SENSORINFO * PSENSORINFO
            
        struct _REVISIONINFO:
            WORD  size                     # 2
            WORD  Sensor                   # 2
            WORD  Cypress                  # 2
            DWORD Blackfin                 # 4
            WORD  DspFirmware              # 2
                                        # --12
            WORD  USB_Board                # 2
            WORD  Sensor_Board             # 2
            WORD  Processing_Board         # 2
            WORD  Memory_Board             # 2
            WORD  Housing                  # 2
            WORD  Filter                   # 2
            WORD  Timing_Board             # 2
            WORD  Product                  # 2
            WORD  Power_Board              # 2
            WORD  Logic_Board              # 2
                                        # --28
            BYTE reserved[96]          # --128
        
        ctypedef _REVISIONINFO REVISIONINFO
        ctypedef  REVISIONINFO * pREVISIONINFO

# ----------------------------------------------------------------------------
# Capture errors
# ----------------------------------------------------------------------------

#~ typedef enum _UEYE_CAPTURE_ERROR
#~ {
    #~ IS_CAPERR_API_NO_DEST_MEM=              0xa2,
    #~ IS_CAPERR_API_CONVERSION_FAILED=        0xa3,
    #~ IS_CAPERR_API_IMAGE_LOCKED=             0xa5,
#~ 
    #~ IS_CAPERR_DRV_OUT_OF_BUFFERS=           0xb2,
    #~ IS_CAPERR_DRV_DEVICE_NOT_READY=         0xb4,
#~ 
    #~ IS_CAPERR_USB_TRANSFER_FAILED=          0xc7,
#~ 
    #~ IS_CAPERR_DEV_TIMEOUT=                  0xd6,
#~ 
    #~ IS_CAPERR_ETH_BUFFER_OVERRUN=           0xe4,
    #~ IS_CAPERR_ETH_MISSED_IMAGES=            0xe5
#~ 
#~ } UEYE_CAPTURE_ERROR
#~ 
#~ typedef struct _UEYE_CAPTURE_ERROR_INFO
#~ {
    #~ DWORD dwCapErrCnt_Total
#~ 
    #~ BYTE  reserved[60]
#~ 
    #~ DWORD adwCapErrCnt_Detail[256] # access via UEYE_CAPTURE_ERROR
#~ 
#~ } UEYE_CAPTURE_ERROR_INFO
#~ 
#~ 
#~ 
#~ #ifndef UEYE_CAMERA_INFO_STRUCT
#~ #define UEYE_CAMERA_INFO_STRUCT
        struct _UEYE_CAMERA_INFO:
            DWORD     dwCameraID   # this is the user definable camera ID
            DWORD     dwDeviceID   # this is the systems enumeration ID
            DWORD     dwSensorID   # this is the sensor ID e.g. IS_SENSOR_UI141X_M
            DWORD     dwInUse      # flag, whether the camera is in use or not
            IS_CHAR   SerNo[16]    # serial number of the camera
            IS_CHAR   Model[16]    # model name of the camera
            DWORD     dwStatus     # various flags with camera status
            DWORD     dwReserved[15]
        
        ctypedef _UEYE_CAMERA_INFO UEYE_CAMERA_INFO
        ctypedef UEYE_CAMERA_INFO * PUEYE_CAMERA_INFO



# usage of the list:
# 1. call the DLL with .dwCount = 0
# 2. DLL returns .dwCount = N  (N = number of available cameras)
# 3. call DLL with .dwCount = N and a pointer to UEYE_CAMERA_LIST with
#    and array of UEYE_CAMERA_INFO[N]
# 4. DLL will fill in the array with the camera infos and
#    will update the .dwCount member with the actual number of cameras
#    because there may be a change in number of cameras between step 2 and 3
# 5. check if there's a difference in actual .dwCount and formerly
#    reported value of N and call DLL again with an updated array size
        struct UEYE_CAMERA_LIST:
            ULONG   dwCount
            UEYE_CAMERA_INFO uci[1]
        #struct UEYE_CAMERA_LIST
        ctypedef UEYE_CAMERA_LIST* PUEYE_CAMERA_LIST
             
 
#~ # ----------------------------------------------------------------------------
#~ # the  following defines are the status bits of the dwStatus member of
#~ # the UEYE_CAMERA_INFO structure
#~ #define FIRMWARE_DOWNLOAD_NOT_SUPPORTED                 0x00000001
#~ #define INTERFACE_SPEED_NOT_SUPPORTED                   0x00000002
#~ #define INVALID_SENSOR_DETECTED                         0x00000004
#~ #define AUTHORIZATION_FAILED                            0x00000008
#~ #define DEVSTS_INCLUDED_STARTER_FIRMWARE_INCOMPATIBLE   0x00000010
#~ 
#~ # the following macro determines the availability of the camera based
#~ # on the status flags
#~ #define IS_CAMERA_AVAILABLE(_s_)     ( (((_s_) & FIRMWARE_DOWNLOAD_NOT_SUPPORTED) == 0) && \
                                       #~ (((_s_) & INTERFACE_SPEED_NOT_SUPPORTED)   == 0) && \
                                       #~ (((_s_) & INVALID_SENSOR_DETECTED)         == 0) && \
                                       #~ (((_s_) & AUTHORIZATION_FAILED)            == 0) )
#~ 
#~ # ----------------------------------------------------------------------------
#~ # auto feature structs and definitions
#~ # ----------------------------------------------------------------------------
#~ #define AC_SHUTTER              0x00000001
#~ #define AC_GAIN                 0x00000002
#~ #define AC_WHITEBAL             0x00000004
#~ #define AC_WB_RED_CHANNEL       0x00000008
#~ #define AC_WB_GREEN_CHANNEL     0x00000010
#~ #define AC_WB_BLUE_CHANNEL      0x00000020
#~ #define AC_FRAMERATE            0x00000040
#~ #define AC_SENSOR_SHUTTER       0x00000080
#~ #define AC_SENSOR_GAIN          0x00000100
#~ #define AC_SENSOR_GAIN_SHUTTER  0x00000200
#~ #define AC_SENSOR_FRAMERATE     0x00000400
#~ 
#~ #define ACS_ADJUSTING 0x00000001
#~ #define ACS_FINISHED  0x00000002
#~ #define ACS_DISABLED  0x00000004
#~ 
#~ typedef struct _AUTO_BRIGHT_STATUS
#~ {
    #~ DWORD curValue             # current average greylevel
    #~ long curError              # current auto brightness error
    #~ DWORD curController        # current active brightness controller -> AC_x
    #~ DWORD curCtrlStatus        # current control status -> ACS_x
#~ } AUTO_BRIGHT_STATUS, *PAUTO_BRIGHT_STATUS
#~ 
#~ 
#~ 
#~ typedef struct _AUTO_WB_CHANNNEL_STATUS
#~ {
    #~ DWORD curValue             # current average greylevel
    #~ long curError              # current auto wb error
    #~ DWORD curCtrlStatus        # current control status -> ACS_x
#~ } AUTO_WB_CHANNNEL_STATUS, *PAUTO_WB_CHANNNEL_STATUS
#~ 
#~ typedef struct _AUTO_WB_STATUS
#~ {
    #~ AUTO_WB_CHANNNEL_STATUS RedChannel
    #~ AUTO_WB_CHANNNEL_STATUS GreenChannel
    #~ AUTO_WB_CHANNNEL_STATUS BlueChannel
    #~ DWORD curController        # current active wb controller -> AC_x
#~ } AUTO_WB_STATUS, *PAUTO_WB_STATUS
#~ 
#~ typedef struct _UEYE_AUTO_INFO
#~ {
    #~ DWORD               AutoAbility        # auto control ability
    #~ AUTO_BRIGHT_STATUS  sBrightCtrlStatus  # brightness auto control status
    #~ AUTO_WB_STATUS      sWBCtrlStatus      # white balance auto control status
    #~ DWORD               reserved[12]
#~ } UEYE_AUTO_INFO, *PUEYE_AUTO_INFO
#~ 
#~ 
#~ # ----------------------------------------------------------------------------
#~ # function exports
#~ # ----------------------------------------------------------------------------
#~ #ifdef __LINUX__
    #~ IDSEXP is_WaitEvent             (HIDS hf, INT which, INT nTimeout)
#~ #endif
#~ 
#~ # ----------------------------------------------------------------------------
#~ # functions only effective on Falcon boards
#~ # ----------------------------------------------------------------------------
  #~ IDSEXP   is_SetVideoInput          (HIDS hf, INT Source)
        IDSEXP   is_SetSaturation          (HIDS hf, INT ChromU, INT ChromV)
  #~ IDSEXP   is_SetHue                 (HIDS hf, INT Hue)
  #~ IDSEXP   is_SetVideoMode           (HIDS hf, INT Mode)
  #~ IDSEXP   is_SetAGC                 (HIDS hf, INT Mode)
  #~ IDSEXP   is_SetSyncLevel           (HIDS hf, INT Level)
  #~ IDSEXP   is_ShowColorBars          (HIDS hf, INT Mode)
  #~ IDSEXP   is_SetScaler              (HIDS hf, float Scalex, float Scaley)
  #~ IDSEXP   is_SetCaptureMode         (HIDS hf, INT Mode)
  #~ IDSEXP   is_SetHorFilter           (HIDS hf, INT Mode)
  #~ IDSEXP   is_SetVertFilter          (HIDS hf, INT Mode)
  #~ IDSEXP   is_ScaleDDOverlay         (HIDS hf, BOOL boScale)
  #~ IDSEXP   is_GetCurrentField        (HIDS hf, int* pField)
  #~ IDSEXP   is_SetVideoSize           (HIDS hf, INT xpos, INT ypos, INT xsize, INT ysize)
  #~ IDSEXP   is_SetKeyOffset           (HIDS hf, INT nOffsetX, INT nOffsetY)
  #~ IDSEXP   is_PrepareStealVideo      (HIDS hf, int Mode, ULONG StealColorMode)
  #~ IDSEXP   is_SetParentHwnd          (HIDS hf, HWND hwnd)
  #~ IDSEXP   is_SetUpdateMode          (HIDS hf, INT mode)
  #~ IDSEXP   is_OvlSurfaceOffWhileMove (HIDS hf, BOOL boMode)
  #~ IDSEXP   is_GetPciSlot             (HIDS hf, INT* pnSlot)
  #~ IDSEXP   is_GetIRQ                 (HIDS hf, INT* pnIRQ)
  #~ IDSEXP   is_SetToggleMode          (HIDS hf, int nInput1, int nInput2, int nInput3, int nInput4)
  #~ IDSEXP   is_SetDecimationMode      (HIDS hf, INT nMode, INT nDecimate)
  #~ IDSEXP   is_SetSync                (HIDS hf, INT nSync)
  #~ # only FALCON duo/quattro
  #~ IDSEXP   is_SetVideoCrossbar       (HIDS hf, INT In, INT Out)
  #~ # watchdog functions
  #~ IDSEXP   is_WatchdogTime           (HIDS hf, long lTime)
  #~ IDSEXP   is_Watchdog               (HIDS hf, long lMode)
  #~ # video out functions
  #~ IDSEXP   is_SetPassthrough         (HIDS hf, INT Source)
#~ 
#~ # ----------------------------------------------------------------------------
#~ # alias functions for compatibility
#~ # ----------------------------------------------------------------------------
  #~ IDSEXP   is_InitBoard              (HIDS* phf, HWND hWnd)
  #~ IDSEXP   is_ExitBoard              (HIDS hf)
  #~ IDSEXP   is_InitFalcon             (HIDS* phf, HWND hWnd)
  #~ IDSEXP   is_ExitFalcon             (HIDS hf)
#~ 
  #~ IDSEXP   is_GetBoardType           (HIDS hf)
  #~ IDSEXP   is_GetBoardInfo           (HIDS hf, PBOARDINFO pInfo)
  #~ IDSEXPUL is_BoardStatus            (HIDS hf, INT nInfo, ULONG ulValue)
  #~ IDSEXP   is_GetNumberOfDevices     (void)
  #~ IDSEXP   is_GetNumberOfBoards      (INT* pnNumBoards)
#~ 
# ----------------------------------------------------------------------------
# common function
# ----------------------------------------------------------------------------
        IDSEXP   is_StopLiveVideo          (HIDS hf, INT Wait)
        IDSEXP   is_FreezeVideo            (HIDS hf, INT Wait)
        IDSEXP   is_CaptureVideo           (HIDS hf, INT Wait)
        IDSEXP   is_IsVideoFinish          (HIDS hf, BOOL* pbo)
        IDSEXP   is_HasVideoStarted        (HIDS hf, BOOL* pbo)

        IDSEXP   is_SetBrightness          (HIDS hf, INT Bright)
        IDSEXP   is_SetContrast            (HIDS hf, INT Cont)
        IDSEXP   is_SetGamma               (HIDS hf, INT nGamma)

        IDSEXP   is_AllocImageMem          (HIDS hf, INT width, INT height, INT bitspixel, char** ppcImgMem, int* pid)
        IDSEXP   is_SetImageMem            (HIDS hf, char* pcMem, int id)
        IDSEXP   is_FreeImageMem           (HIDS hf, char* pcMem, int id)
        IDSEXP   is_GetImageMem            (HIDS hf, VOID** pMem)
        IDSEXP   is_GetActiveImageMem      (HIDS hf, char** ppcMem, int* pnID)
        IDSEXP   is_InquireImageMem        (HIDS hf, char* pcMem, int nID, int* pnX, int* pnY, int* pnBits, int* pnPitch)
        IDSEXP   is_GetImageMemPitch       (HIDS hf, INT* pPitch)

        IDSEXP   is_SetAllocatedImageMem   (HIDS hf, INT width, INT height, INT bitspixel, char* pcImgMem, int* pid)
        IDSEXP   is_SaveImageMem           (HIDS hf, IS_CHAR* File, char* pcMem, int nID)
        IDSEXP   is_CopyImageMem           (HIDS hf, char* pcSource, int nID, char* pcDest)
        IDSEXP   is_CopyImageMemLines      (HIDS hf, char* pcSource, int nID, int nLines, char* pcDest)

        IDSEXP   is_AddToSequence          (HIDS hf, char* pcMem, INT nID)
        IDSEXP   is_ClearSequence          (HIDS hf)
        IDSEXP   is_GetActSeqBuf           (HIDS hf, INT* pnNum, char** ppcMem, char** ppcMemLast)
        IDSEXP   is_LockSeqBuf             (HIDS hf, INT nNum, char* pcMem)
        IDSEXP   is_UnlockSeqBuf           (HIDS hf, INT nNum, char* pcMem)

        IDSEXP   is_SetImageSize           (HIDS hf, INT x, INT y)
        IDSEXP   is_SetImagePos            (HIDS hf, INT x, INT y)

        IDSEXP   is_GetError               (HIDS hf, INT* pErr, IS_CHAR** ppcErr)
        IDSEXP   is_SetErrorReport         (HIDS hf, INT Mode)

        IDSEXP   is_ReadEEPROM             (HIDS hf, INT Adr, char* pcString, INT Count)
        IDSEXP   is_WriteEEPROM            (HIDS hf, INT Adr, char* pcString, INT Count)
        IDSEXP   is_SaveImage              (HIDS hf, IS_CHAR* File)

        IDSEXP   is_SetColorMode           (HIDS hf, INT Mode)
        IDSEXP   is_GetColorDepth          (HIDS hf, INT* pnCol, INT* pnColMode)
        # bitmap display function
        IDSEXP   is_RenderBitmap           (HIDS hf, INT nMemID, HWND hwnd, INT nMode)

        IDSEXP   is_SetDisplayMode         (HIDS hf, INT Mode)
        IDSEXP   is_GetDC                  (HIDS hf, HDC* phDC)
        IDSEXP   is_ReleaseDC              (HIDS hf, HDC hDC)
        IDSEXP   is_UpdateDisplay          (HIDS hf)
        IDSEXP   is_SetDisplaySize         (HIDS hf, INT x, INT y)
        IDSEXP   is_SetDisplayPos          (HIDS hf, INT x, INT y)

        IDSEXP   is_LockDDOverlayMem       (HIDS hf, VOID** ppMem, INT* pPitch)
        IDSEXP   is_UnlockDDOverlayMem     (HIDS hf)
        IDSEXP   is_LockDDMem              (HIDS hf, VOID** ppMem, INT* pPitch)
        IDSEXP   is_UnlockDDMem            (HIDS hf)
        IDSEXP   is_GetDDOvlSurface        (HIDS hf, void** ppDDSurf)
        IDSEXP   is_SetKeyColor            (HIDS hf, INT r, INT g, INT b)
        IDSEXP   is_StealVideo             (HIDS hf, int Wait)
        IDSEXP   is_SetHwnd                (HIDS hf, HWND hwnd)


        IDSEXP   is_SetDDUpdateTime        (HIDS hf, INT ms)
        IDSEXP   is_EnableDDOverlay        (HIDS hf)
        IDSEXP   is_DisableDDOverlay       (HIDS hf)
        IDSEXP   is_ShowDDOverlay          (HIDS hf)
        IDSEXP   is_HideDDOverlay          (HIDS hf)


        IDSEXP   is_GetVsyncCount          (HIDS hf, long* pIntr, long* pActIntr)
        IDSEXP   is_GetOsVersion           ()
        # version information
        IDSEXP   is_GetDLLVersion          ()

        IDSEXP   is_InitEvent              (HIDS hf, HANDLE hEv, INT which)
        IDSEXP   is_ExitEvent              (HIDS hf, INT which)
        IDSEXP   is_EnableEvent            (HIDS hf, INT which)
        IDSEXP   is_DisableEvent           (HIDS hf, INT which)

        IDSEXP   is_SetIO                  (HIDS hf, INT nIO)
        IDSEXP   is_SetFlashStrobe         (HIDS hf, INT nMode, INT nLine)
        IDSEXP   is_SetExternalTrigger     (HIDS hf, INT nTriggerMode)
        IDSEXP   is_SetTriggerCounter      (HIDS hf, INT nValue)
        IDSEXP   is_SetRopEffect           (HIDS hf, INT effect, INT param, INT reserved)

# ----------------------------------------------------------------------------
# new functions only valid for uEye camera family
# ----------------------------------------------------------------------------
# Camera functions

        IDSEXP is_InitCamera                  (HIDS* phf, HWND hWnd)
        IDSEXP is_ExitCamera                  (HIDS hf)
        IDSEXP is_GetCameraInfo               (HIDS hf, PCAMINFO pInfo)
        IDSEXPUL is_CameraStatus              (HIDS hf, INT nInfo, ULONG ulValue)
        IDSEXP is_GetCameraType               (HIDS hf)
 
        IDSEXP is_GetNumberOfCameras          (INT* pnNumCams)
 

        # PixelClock
        IDSEXP is_GetPixelClockRange          (HIDS hf, INT *pnMin, INT *pnMax)
        IDSEXP is_SetPixelClock               (HIDS hf, INT Clock)
        IDSEXP is_GetUsedBandwidth            (HIDS hf)
        # Set/Get Frame rate
        IDSEXP is_GetFrameTimeRange           (HIDS hf, double *min, double *max, double *intervall)
        IDSEXP is_SetFrameRate                (HIDS hf, double FPS, double* newFPS)
        # Set/Get Exposure
        IDSEXP is_GetExposureRange            (HIDS hf, double *min, double *max, double *intervall)
        IDSEXP is_SetExposureTime             (HIDS hf, double EXP, double* newEXP)
        # get frames per second
        IDSEXP is_GetFramesPerSecond          (HIDS hf, double *dblFPS)

        # is_SetIOMask ( only uEye USB )
        IDSEXP is_SetIOMask                   (HIDS hf, INT nMask)

        # Get Sensor info
        IDSEXP is_GetSensorInfo               (HIDS hf, PSENSORINFO pInfo)
        # Get RevisionInfo
  #~ IDSEXP is_GetRevisionInfo             (HIDS hf, PREVISIONINFO prevInfo)
 
        # enable/disable AutoExit after device remove
        IDSEXP is_EnableAutoExit              (HIDS hf, INT nMode)
        # Message
  #~ IDSEXP is_EnableMessage               (HIDS hf, INT which, HWND hWnd)
 
        # hardware gain settings
        IDSEXP is_SetHardwareGain             (HIDS hf, INT nMaster, INT nRed, INT nGreen, INT nBlue)

        # set render mode
        IDSEXP is_SetRenderMode               (HIDS hf, INT Mode)
 
        # enable/disable WhiteBalance
        IDSEXP is_SetWhiteBalance             (HIDS hf, INT nMode)
        IDSEXP is_SetWhiteBalanceMultipliers  (HIDS hf, double dblRed, double dblGreen, double dblBlue)
        IDSEXP is_GetWhiteBalanceMultipliers  (HIDS hf, double *pdblRed, double *pdblGreen, double *pdblBlue)

        # Edge enhancement
        IDSEXP is_SetEdgeEnhancement          (HIDS hf, INT nEnable)
        
        # Sensor features
        IDSEXP is_SetColorCorrection          (HIDS hf, INT nEnable, double *factors)
        IDSEXP is_SetBlCompensation           (HIDS hf, INT nEnable, INT offset, INT reserved)

        # Hotpixel
        IDSEXP is_SetBadPixelCorrection       (HIDS hf, INT nEnable, INT threshold)
        IDSEXP is_LoadBadPixelCorrectionTable (HIDS hf, IS_CHAR *File)
        IDSEXP is_SaveBadPixelCorrectionTable (HIDS hf, IS_CHAR *File)
        IDSEXP is_SetBadPixelCorrectionTable  (HIDS hf, INT nMode, WORD *pList)

        # Memoryboard
        IDSEXP is_SetMemoryMode               (HIDS hf, INT nCount, INT nDelay)
        IDSEXP is_TransferImage               (HIDS hf, INT nMemID, INT seqID, INT imageNr, INT reserved)
        IDSEXP is_TransferMemorySequence      (HIDS hf, INT seqID, INT StartNr, INT nCount, INT nSeqPos)
        IDSEXP is_MemoryFreezeVideo           (HIDS hf, INT nMemID, INT Wait)
        IDSEXP is_GetLastMemorySequence       (HIDS hf, INT *pID)
        IDSEXP is_GetNumberOfMemoryImages     (HIDS hf, INT nID, INT *pnCount)
        IDSEXP is_GetMemorySequenceWindow     (HIDS hf, INT nID, INT *left, INT *top, INT *right, INT *bottom)
        IDSEXP is_IsMemoryBoardConnected      (HIDS hf, BOOL *pConnected)
        IDSEXP is_ResetMemory                 (HIDS hf, INT nReserved)
 
        IDSEXP is_SetSubSampling              (HIDS hf, INT mode)
        IDSEXP is_ForceTrigger                (HIDS hf)
 
        # new with driver version 1.12.0006
        IDSEXP is_GetBusSpeed                 (HIDS hf)
        
        # new with driver version 1.12.0015
        IDSEXP is_SetBinning                  (HIDS hf, INT mode)
 
        # new with driver version 1.12.0017
        IDSEXP is_ResetToDefault              (HIDS hf)
        IDSEXP is_LoadParameters              (HIDS hf, IS_CHAR* pFilename)
        IDSEXP is_SaveParameters              (HIDS hf, IS_CHAR* pFilename)
 
        # new with driver version 1.14.0001
        IDSEXP is_GetGlobalFlashDelays        (HIDS hf, ULONG *pulDelay, ULONG *pulDuration)
        IDSEXP is_SetFlashDelay               (HIDS hf, ULONG ulDelay, ULONG ulDuration)
        # new with driver version 1.14.0002
        IDSEXP is_LoadImage                   (HIDS hf, IS_CHAR* File)
 
        # new with driver version 1.14.0008
        IDSEXP is_SetImageAOI                 (HIDS hf, INT xPos, INT yPos, INT width, INT height)
        IDSEXP is_SetCameraID                 (HIDS hf, INT nID)
        IDSEXP is_SetBayerConversion          (HIDS hf, INT nMode)
        IDSEXP is_SetTestImage                (HIDS hf, INT nMode)
        # new with driver version 1.14.0009
        IDSEXP is_SetHardwareGamma            (HIDS hf, INT nMode)

        # new with driver version 2.00.0001
        IDSEXP is_GetCameraList               (PUEYE_CAMERA_LIST pucl) nogil
        #IDSEXP is_GetCameraList               (UEYE_CAMERA_LIST *pucl)
 
        # new with driver version 2.00.0011
        IDSEXP is_SetAOI                      (HIDS hf, INT type, INT *pXPos, INT *pYPos, INT *pWidth, INT *pHeight)
        IDSEXP is_SetAutoParameter            (HIDS hf, INT param, double *pval1, double *pval2)
        #IDSEXP is_GetAutoInfo                 (HIDS hf, UEYE_AUTO_INFO *pInfo)
   
        # new with driver version 2.20.0001
        IDSEXP is_ConvertImage                (HIDS hf, char* pcSource, int nIDSource, char** pcDest, INT *nIDDest, INT *reserved)
        IDSEXP is_SetConvertParam             (HIDS hf, BOOL ColorCorrection, INT BayerConversionMode, INT ColorMode, INT Gamma, double* WhiteBalanceMultipliers)
        
        IDSEXP is_SaveImageEx                 (HIDS hf, IS_CHAR* File, INT fileFormat, INT Param)
        IDSEXP is_SaveImageMemEx              (HIDS hf, IS_CHAR* File, char* pcMem, INT nID, INT FileFormat, INT Param)
        IDSEXP is_LoadImageMem                (HIDS hf, IS_CHAR* File, char** ppcImgMem, INT* pid)
   
        IDSEXP is_GetImageHistogram           (HIDS hf, int nID, INT ColorMode, DWORD* pHistoMem)
        IDSEXP is_SetTriggerDelay             (HIDS hf, INT nTriggerDelay)
 
        # new with driver version 2.21.0000
        IDSEXP is_SetGainBoost                (HIDS hf, INT mode)
        IDSEXP is_SetLED                      (HIDS hf, INT nValue)
 
        IDSEXP is_SetGlobalShutter            (HIDS hf, INT mode)
        IDSEXP is_SetExtendedRegister         (HIDS hf, INT index,WORD value)
        IDSEXP is_GetExtendedRegister         (HIDS hf, INT index, WORD *pwValue)
 
        # new with driver version 2.22.0002
        IDSEXP is_SetHWGainFactor             (HIDS hf, INT nMode, INT nFactor)
    
        # camera renumeration
        IDSEXP is_Renumerate                  (HIDS hf, INT nMode)
    
        # Read / Write I2C
        IDSEXP is_WriteI2C                    (HIDS hf, INT nDeviceAddr, INT nRegisterAddr, BYTE* pbData, INT nLen)
        IDSEXP is_ReadI2C                     (HIDS hf, INT nDeviceAddr, INT nRegisterAddr, BYTE* pbData, INT nLen)
 
 
        # new with driver version 3.10.0000
  #~ typedef struct _KNEEPOINT
  #~ {
      #~ double x
      #~ double y
  #~ } KNEEPOINT, *PKNEEPOINT
#~ 
#~ 
  #~ typedef struct _KNEEPOINTARRAY
  #~ {
      #~ INT NumberOfUsedKneepoints
      #~ KNEEPOINT Kneepoint[10]
  #~ } KNEEPOINTARRAY, *PKNEEPOINTARRAY
#~ 
#~ 
  #~ typedef struct _KNEEPOINTINFO
  #~ {
      #~ INT NumberOfSupportedKneepoints
      #~ INT NumberOfUsedKneepoints
      #~ double MinValueX
      #~ double MaxValueX
      #~ double MinValueY
      #~ double MaxValueY
      #~ KNEEPOINT DefaultKneepoint[10]
      #~ INT Reserved[10]
  #~ } KNEEPOINTINFO, *PKNEEPOINTINFO
#~ 
#~ 
        # HDR functions
        IDSEXP is_GetHdrMode                  (HIDS hf, INT *Mode)
        IDSEXP is_EnableHdr                   (HIDS hf, INT Enable)
        #~ IDSEXP is_SetHdrKneepoints            (HIDS hf, KNEEPOINTARRAY *KneepointArray, INT KneepointArraySize)
        #~ IDSEXP is_GetHdrKneepoints            (HIDS hf, KNEEPOINTARRAY *KneepointArray, INT KneepointArraySize)
        #~ IDSEXP is_GetHdrKneepointInfo         (HIDS hf, KNEEPOINTINFO *KneepointInfo, INT KneepointInfoSize)
 
        IDSEXP is_SetOptimalCameraTiming      (HIDS hf, INT Mode, INT Timeout, INT *pMaxPxlClk, double *pMaxFrameRate)
 
        IDSEXP is_GetSupportedTestImages      (HIDS hf, INT *SupportedTestImages)
        IDSEXP is_GetTestImageValueRange      (HIDS hf, INT TestImage, INT *TestImageValueMin, INT *TestImageValueMax)
        IDSEXP is_SetSensorTestImage          (HIDS hf, INT Param1, INT Param2)
 
        IDSEXP is_SetCameraLUT                (HIDS hf, UINT Mode, UINT NumberOfEntries, double *pRed_Grey, double *pGreen, double *pBlue)
        IDSEXP is_GetCameraLUT                (HIDS hf, UINT Mode, UINT NumberOfEntries, double *pRed_Grey, double *pGreen, double *pBlue)
 
        IDSEXP is_GetColorConverter           (HIDS hf, INT ColorMode, INT *pCurrentConvertMode, INT *pDefaultConvertMode, INT *pSupportedConvertModes)
        IDSEXP is_SetColorConverter           (HIDS hf, INT ColorMode, INT ConvertMode)
 
        #IDSEXP is_GetCaptureErrorInfo         (HIDS hf, UEYE_CAPTURE_ERROR_INFO *pCaptureErrorInfo, UINT SizeCaptureErrorInfo)
        IDSEXP is_ResetCaptureErrorInfo       (HIDS hf )
 
        IDSEXP is_WaitForNextImage            (HIDS hf, UINT timeout, char **ppcMem, INT *imageID)
        IDSEXP is_InitImageQueue              (HIDS hf, INT nMode)
        IDSEXP is_ExitImageQueue              (HIDS hf)
 
        IDSEXP is_SetTimeout                  (HIDS hf, UINT nMode, UINT Timeout)
        IDSEXP is_GetTimeout                  (HIDS hf, UINT nMode, UINT *pTimeout)
 
 
        #typedef enum  eUEYE_GET_ESTIMATED_TIME_MODE
  #~ {
      #~ IS_SE_STARTER_FW_UPLOAD =   0x00000001, /*!< get estimated duration of GigE SE starter firmware upload in milliseconds */
#~ 
  #~ } UEYE_GET_ESTIMATED_TIME_MODE    
  #~ 
  #~ 
        IDSEXP is_GetDuration                 (HIDS hf, UINT nMode, INT* pnTime)
#~ 
#~ 
        # new with driver version 3.40.0000
  #~ typedef struct _SENSORSCALERINFO
  #~ {
      #~ INT       nCurrMode
      #~ INT       nNumberOfSteps
      #~ double    dblFactorIncrement
      #~ double    dblMinFactor
      #~ double    dblMaxFactor
      #~ double    dblCurrFactor
      #~ BYTE      bReserved[88]
  #~ } SENSORSCALERINFO
#~ 
#~ 
  #~ IDSEXP is_GetSensorScalerInfo (HIDS hf, SENSORSCALERINFO *pSensorScalerInfo, INT nSensorScalerInfoSize)
        IDSEXP is_SetSensorScaler      (HIDS hf, UINT nMode, double dblFactor) 

  #~ typedef struct _UEYETIME
  #~ {
      #~ WORD      wYear
      #~ WORD      wMonth
      #~ WORD      wDay
      #~ WORD      wHour
      #~ WORD      wMinute
      #~ WORD      wSecond
      #~ WORD      wMilliseconds
      #~ BYTE      byReserved[10]
  #~ } UEYETIME
#~ 
#~ 
  #~ typedef struct _UEYEIMAGEINFO
  #~ {
      #~ DWORD                 dwFlags
      #~ BYTE                  byReserved1[4]
      #~ UINT64                u64TimestampDevice
      #~ UEYETIME              TimestampSystem
      #~ DWORD                 dwIoStatus
      #~ BYTE                  byReserved2[4]
      #~ UINT64                u64FrameNumber
      #~ DWORD                 dwImageBuffers
      #~ DWORD                 dwImageBuffersInUse
      #~ DWORD                 dwReserved3
      #~ DWORD                 dwImageHeight
      #~ DWORD                 dwImageWidth
  #~ } UEYEIMAGEINFO
#~ 
#~ 
  #~ IDSEXP is_GetImageInfo (HIDS hf, INT nImageBufferID, UEYEIMAGEINFO *pImageInfo, INT nImageInfoSize)
#~ 
        # ----------------------------------------------------------------------------
        # new functions and datatypes only valid for uEye ETH
        # ----------------------------------------------------------------------------
    
#~ #pragma pack( push, 1)
#~ 
  #~ # IP V4 address
  #~ typedef union _UEYE_ETH_ADDR_IPV4
  #~ {
      #~ struct
      #~ {
          #~ BYTE  by1
          #~ BYTE  by2
          #~ BYTE  by3
          #~ BYTE  by4
      #~ } by
#~ 
      #~ DWORD dwAddr
#~ 
  #~ } UEYE_ETH_ADDR_IPV4, *PUEYE_ETH_ADDR_IPV4
#~ 
  #~ # Ethernet address
  #~ typedef struct _UEYE_ETH_ADDR_MAC
  #~ {
      #~ BYTE abyOctet[6]
#~ 
  #~ } UEYE_ETH_ADDR_MAC, *PUEYE_ETH_ADDR_MAC
#~ 
  #~ # IP configuration
  #~ typedef struct _UEYE_ETH_IP_CONFIGURATION
  #~ {
      #~ UEYE_ETH_ADDR_IPV4    ipAddress      /*!< IP address */
      #~ UEYE_ETH_ADDR_IPV4    ipSubnetmask   /*!< IP subnetmask */
#~ 
      #~ BYTE                  reserved[4]    /*!< reserved */
#~ 
  #~ } UEYE_ETH_IP_CONFIGURATION, *PUEYE_ETH_IP_CONFIGURATION
#~ 
  #~ # values for UEYE_ETH_DEVICE_INFO_HEARTBEAT::dwDeviceStatusFlags
  #~ typedef enum _UEYE_ETH_DEVICESTATUS
  #~ {
      #~ IS_ETH_DEVSTATUS_READY_TO_OPERATE=            0x00000001, /*!< device is ready to operate */
      #~ IS_ETH_DEVSTATUS_TESTING_IP_CURRENT=          0x00000002, /*!< device is (arp-)probing its current ip */
      #~ IS_ETH_DEVSTATUS_TESTING_IP_PERSISTENT=       0x00000004, /*!< device is (arp-)probing its persistent ip */
      #~ IS_ETH_DEVSTATUS_TESTING_IP_RANGE=            0x00000008, /*!< device is (arp-)probing the autocfg ip range */
#~ 
      #~ IS_ETH_DEVSTATUS_INAPPLICABLE_IP_CURRENT=     0x00000010, /*!< current ip is inapplicable */
      #~ IS_ETH_DEVSTATUS_INAPPLICABLE_IP_PERSISTENT=  0x00000020, /*!< persistent ip is inapplicable */
      #~ IS_ETH_DEVSTATUS_INAPPLICABLE_IP_RANGE=       0x00000040, /*!< autocfg ip range is inapplicable */
#~ 
      #~ IS_ETH_DEVSTATUS_UNPAIRED=                    0x00000100, /*!< device is unpaired */
      #~ IS_ETH_DEVSTATUS_PAIRING_IN_PROGRESS=         0x00000200, /*!< device is being paired */
      #~ IS_ETH_DEVSTATUS_PAIRED=                      0x00000400, /*!< device is paired */
#~ 
      #~ IS_ETH_DEVSTATUS_FORCE_100MBPS=               0x00001000, /*!< device phy is configured to 100 Mbps */
      #~ IS_ETH_DEVSTATUS_NO_COMPORT=                  0x00002000, /*!< device does not support ueye eth comport */
#~ 
      #~ IS_ETH_DEVSTATUS_RECEIVING_FW_STARTER=        0x00010000, /*!< device is receiving the starter firmware */
      #~ IS_ETH_DEVSTATUS_RECEIVING_FW_RUNTIME=        0x00020000, /*!< device is receiving the runtime firmware */
      #~ IS_ETH_DEVSTATUS_INAPPLICABLE_FW_RUNTIME=     0x00040000, /*!< runtime firmware is inapplicable */
      #~ IS_ETH_DEVSTATUS_INAPPLICABLE_FW_STARTER=     0x00080000, /*!< starter firmware is inapplicable */
#~ 
      #~ IS_ETH_DEVSTATUS_REBOOTING_FW_RUNTIME=        0x00100000, /*!< device is rebooting to runtime firmware */
      #~ IS_ETH_DEVSTATUS_REBOOTING_FW_STARTER=        0x00200000, /*!< device is rebooting to starter firmware */
      #~ IS_ETH_DEVSTATUS_REBOOTING_FW_FAILSAFE=       0x00400000, /*!< device is rebooting to failsafe firmware */
#~ 
      #~ IS_ETH_DEVSTATUS_RUNTIME_FW_ERR0=             0x80000000, /*!< checksum error runtime firmware */
#~ 
  #~ } UEYE_ETH_DEVICESTATUS
#~ 
  #~ # heartbeat info transmitted periodically by a device
  #~ # contained in UEYE_ETH_DEVICE_INFO
  #~ typedef struct _UEYE_ETH_DEVICE_INFO_HEARTBEAT
  #~ {
      #~ BYTE                  abySerialNumber[1r the actual size of a :ctype:`word ` is (provided the header file defines it correctly). Conversion to and from Python types, if any, will also be used for this new type.2]        /*!< camera's serial number (string) */
#~ 
      #~ BYTE                  byDeviceType               /*!< device type / board type, 0x80 for ETH */
      #~ 
      #~ BYTE                  byCameraID                 /*!< camera id */
#~ 
      #~ WORD                  wSensorID                  /*!< camera's sensor's id */
#~ 
      #~ WORD                  wSizeImgMem_MB             /*!< size of camera's image memory in MB */
#~ 
      #~ BYTE                  reserved_1[2]              /*!< reserved */
#~ 
      #~ DWORD                 dwVerStarterFirmware       /*!< starter firmware version */
#~ 
      #~ DWORD                 dwVerRuntimeFirmware       /*!< runtime firmware version */
#~ 
      #~ DWORD                 dwStatus                   /*!< camera status flags */
#~ 
      #~ BYTE                  reserved_2[4]              /*!< reserved */
      #~ 
      #~ WORD                  wTemperature               /*!< camera temperature */
      #~ 
      #~ WORD                  wLinkSpeed_Mb              /*!< link speed in Mb */
#~ 
      #~ UEYE_ETH_ADDR_MAC     macDevice                  /*!< camera's MAC address */
      #~ 
      #~ WORD                  wComportOffset             /*!< comport offset from 100, valid range -99 to +156 */
#~ 
      #~ UEYE_ETH_IP_CONFIGURATION ipcfgPersistentIpCfg   /*!< persistent IP configuration */
#~ 
      #~ UEYE_ETH_IP_CONFIGURATION ipcfgCurrentIpCfg      /*!< current IP configuration */
#~ 
      #~ UEYE_ETH_ADDR_MAC     macPairedHost              /*!< paired host's MAC address */
#~ 
      #~ BYTE                  reserved_4[2]              /*!< reserved */
#~ 
      #~ UEYE_ETH_ADDR_IPV4    ipPairedHostIp             /*!< paired host's IP address */
#~ 
      #~ UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeBegin      /*!< begin of IP address range */
#~ 
      #~ UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeEnd        /*!< end of IP address range */
#~ 
      #~ BYTE                  abyUserSpace[8]            /*!< user space data (first 8 bytes) */
#~ 
      #~ BYTE                  reserved_5[84]             /*!< reserved */
#~ 
      #~ BYTE                  reserved_6[64]             /*!< reserved */
#~ 
  #~ } UEYE_ETH_DEVICE_INFO_HEARTBEAT, *PUEYE_ETH_DEVICE_INFO_HEARTBEAT
#~ 
  #~ # values for UEYE_ETH_DEVICE_INFO_CONTROL::dwControlStatus
  #~ typedef enum _UEYE_ETH_CONTROLSTATUS
  #~ {
      #~ IS_ETH_CTRLSTATUS_AVAILABLE=              0x00000001, /*!< device is available TO US */
      #~ IS_ETH_CTRLSTATUS_ACCESSIBLE1=            0x00000002, /*!< device is accessible BY US, i.e. directly 'unicastable' */
      #~ IS_ETH_CTRLSTATUS_ACCESSIBLE2=            0x00000004, /*!< device is accessible BY US, i.e. not on persistent ip and adapters ip autocfg range is valid */
#~ 
      #~ IS_ETH_CTRLSTATUS_PERSISTENT_IP_USED=     0x00000010, /*!< device is running on persistent ip configuration */
      #~ IS_ETH_CTRLSTATUS_COMPATIBLE=             0x00000020, /*!< device is compatible TO US */
      #~ IS_ETH_CTRLSTATUS_ADAPTER_ON_DHCP=        0x00000040, /*!< adapter is configured to use dhcp */
#~ 
      #~ IS_ETH_CTRLSTATUS_UNPAIRING_IN_PROGRESS=  0x00000100, /*!< device is being unpaired FROM US */
      #~ IS_ETH_CTRLSTATUS_PAIRING_IN_PROGRESS=    0x00000200, /*!< device is being paired TO US */
#~ 
      #~ IS_ETH_CTRLSTATUS_PAIRED=                 0x00001000, /*!< device is paired TO US */
#~ 
      #~ IS_ETH_CTRLSTATUS_FW_UPLOAD_STARTER=      0x00010000, /*!< device is receiving the starter firmware */
      #~ IS_ETH_CTRLSTATUS_FW_UPLOAD_RUNTIME=      0x00020000, /*!< device is receiving the runtime firmware */
#~ 
      #~ IS_ETH_CTRLSTATUS_REBOOTING=              0x00100000, /*!< device is rebooting */
#~ 
      #~ IS_ETH_CTRLSTATUS_INITIALIZED=            0x08000000, /*!< device object is initialized */
#~ 
      #~ IS_ETH_CTRLSTATUS_TO_BE_DELETED=          0x40000000, /*!< device object is being deleted */
      #~ IS_ETH_CTRLSTATUS_TO_BE_REMOVED=          0x80000000, /*!< device object is being removed */
#~ 
  #~ } UEYE_ETH_CONTROLSTATUS
#~ 
  #~ # control info for a listed device
  #~ # contained in UEYE_ETH_DEVICE_INFO
  #~ typedef struct _UEYE_ETH_DEVICE_INFO_CONTROL
  #~ {
      #~ DWORD     dwDeviceID         /*!< device's unique id */
#~ 
      #~ DWORD     dwControlStatus    /*!< device control status */
#~ 
      #~ BYTE      reserved_1[80]     /*!< reserved */
#~ 
      #~ BYTE      reserved_2[64]     /*!< reserved */
#~ 
  #~ } UEYE_ETH_DEVICE_INFO_CONTROL, *PUEYE_ETH_DEVICE_INFO_CONTROL
#~ 
  #~ # Ethernet configuration
  #~ typedef struct _UEYE_ETH_ETHERNET_CONFIGURATION
  #~ {
      #~ UEYE_ETH_IP_CONFIGURATION ipcfg
      #~ UEYE_ETH_ADDR_MAC         mac
#~ 
  #~ } UEYE_ETH_ETHERNET_CONFIGURATION, *PUEYE_ETH_ETHERNET_CONFIGURATION
#~ 
  #~ # autocfg ip setup
  #~ typedef struct _UEYE_ETH_AUTOCFG_IP_SETUP
  #~ {
      #~ UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeBegin      /*!< begin of ip address range for devices */
      #~ UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeEnd        /*!< end of ip address range for devices */
#~ 
      #~ BYTE                  reserved[4]    /*!< reserved */
#~ 
  #~ } UEYE_ETH_AUTOCFG_IP_SETUP, *PUEYE_ETH_AUTOCFG_IP_SETUP
#~ 
  #~ # values for incoming packets filter setup
  #~ typedef enum _UEYE_ETH_PACKETFILTER_SETUP
  #~ {
      #~ # notice: arp and icmp (ping) packets are always passed!
#~ 
      #~ IS_ETH_PCKTFLT_PASSALL=       0,  /*!< pass all packets to OS */
      #~ IS_ETH_PCKTFLT_BLOCKUEGET=    1,  /*!< block UEGET packets to the OS */
      #~ IS_ETH_PCKTFLT_BLOCKALL=      2   /*!< block all packets to the OS */
#~ 
  #~ } UEYE_ETH_PACKETFILTER_SETUP
#~ 
  #~ # values for link speed setup
  #~ typedef enum _UEYE_ETH_LINKSPEED_SETUP
  #~ {
      #~ IS_ETH_LINKSPEED_100MB=       100,    /*!< 100 MBits */
      #~ IS_ETH_LINKSPEED_1000MB=      1000    /*!< 1000 MBits */
#~ 
  #~ } UEYE_ETH_LINKSPEED_SETUP
#~ 
#~ 
  #~ # control info for a device's network adapter
  #~ # contained in UEYE_ETH_DEVICE_INFO
  #~ typedef struct _UEYE_ETH_ADAPTER_INFO
  #~ {
      #~ DWORD                             dwAdapterID        /*!< adapter's unique id */
#~ 
      #~ DWORD                             dwDeviceLinkspeed  /*!< device's linked to this adapter are forced to use this link speed */
#~ 
      #~ UEYE_ETH_ETHERNET_CONFIGURATION   ethcfg         /*!< adapter's eth configuration */
      #~ BYTE                              reserved_2[2]  /*!< reserved */
      #~ BOOL                              bIsEnabledDHCP /*!< adapter's dhcp enabled flag */
#~ 
      #~ UEYE_ETH_AUTOCFG_IP_SETUP         autoCfgIp                  /*!< the setup for the ip auto configuration */
      #~ BOOL                              bIsValidAutoCfgIpRange     /*!<    the given range is valid when: 
                                                                            #~ - begin and end are valid ip addresses
                                                                            #~ - begin and end are in the subnet of the adapter
                                                                            #~ - */
#~ 
      #~ DWORD                             dwCntDevicesKnown      /*!< count of listed Known devices */
      #~ DWORD                             dwCntDevicesPaired     /*!< count of listed Paired devices */
#~ 
      #~ WORD                              wPacketFilter          /*!< Setting for the Incoming Packets Filter. see UEYE_ETH_PACKETFILTER_SETUP enum above. */
#~ 
      #~ BYTE                              reserved_3[38]         /*!< reserved */
      #~ BYTE                              reserved_4[64]         /*!< reserved */
#~ 
  #~ } UEYE_ETH_ADAPTER_INFO, *PUEYE_ETH_ADAPTER_INFO
#~ 
  #~ # driver info
  #~ # contained in UEYE_ETH_DEVICE_INFO
  #~ typedef struct _UEYE_ETH_DRIVER_INFO
  #~ {
      #~ DWORD     dwMinVerStarterFirmware    /*!< minimum version compatible starter firmware */
      #~ DWORD     dwMaxVerStarterFirmware    /*!< maximum version compatible starter firmware */
#~ 
      #~ BYTE      reserved_1[8]              /*!< reserved */
      #~ BYTE      reserved_2[64]             /*!< reserved */
#~ 
  #~ } UEYE_ETH_DRIVER_INFO, *PUEYE_ETH_DRIVER_INFO
#~ 
#~ 
#~ 
  #~ # use is_GetEthDeviceInfo() to obtain this data.
  #~ typedef struct _UEYE_ETH_DEVICE_INFO
  #~ {
      #~ UEYE_ETH_DEVICE_INFO_HEARTBEAT    infoDevHeartbeat
#~ 
      #~ UEYE_ETH_DEVICE_INFO_CONTROL      infoDevControl
#~ 
      #~ UEYE_ETH_ADAPTER_INFO             infoAdapter
#~ 
      #~ UEYE_ETH_DRIVER_INFO              infoDriver
#~ 
  #~ } UEYE_ETH_DEVICE_INFO, *PUEYE_ETH_DEVICE_INFO
#~ 
#~ 
  #~ typedef struct _UEYE_COMPORT_CONFIGURATION
  #~ {
      #~ WORD wComportNumber
#~ 
  #~ } UEYE_COMPORT_CONFIGURATION, *PUEYE_COMPORT_CONFIGURATION
#~ 
#~ 
#~ #pragma pack(pop)
#~ 
  #~ IDSEXP is_GetEthDeviceInfo    (HIDS hf, UEYE_ETH_DEVICE_INFO* pDeviceInfo, UINT uStructSize)
  #~ IDSEXP is_SetPersistentIpCfg  (HIDS hf, UEYE_ETH_IP_CONFIGURATION* pIpCfg, UINT uStructSize)
  #~ IDSEXP is_SetStarterFirmware  (HIDS hf, const CHAR* pcFilepath, UINT uFilepathLen)
#~ 
  #~ IDSEXP is_SetAutoCfgIpSetup   (INT iAdapterID, const UEYE_ETH_AUTOCFG_IP_SETUP* pSetup, UINT uStructSize)
  #~ IDSEXP is_SetPacketFilter     (INT iAdapterID, UINT uFilterSetting)
  #~ 
  #~ IDSEXP is_GetComportNumber    (HIDS hf, UINT *pComportNumber)
#~ 
  #~ IDSEXP is_DirectRenderer      (HIDS hf, UINT nMode, void *pParam, UINT SizeOfParam)
#~ 
#~ #ifdef __cplusplus
#~ }
#~ #endif  /* __cplusplus */
#~ 
#~ #pragma pack(pop)
#~ 
#~ #endif  # #ifndef __IDS_HEADER__
