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

from ueye cimport *

VERSION_CODE                    =   UEYE_VERSION_CODE
            
#----------------------------------------------------------------------------
#Color modes
#----------------------------------------------------------------------------
COLORMODE_INVALID               =   IS_COLORMODE_INVALID
COLORMODE_MONOCHROME            =   IS_COLORMODE_MONOCHROME
COLORMODE_BAYER                 =   IS_COLORMODE_BAYER     

# ----------------------------------------------------------------------------
#  Sensor Types
# ----------------------------------------------------------------------------
SENSOR_INVALID                  =   IS_SENSOR_INVALID           

# CMOS Sensors
SENSOR_UI141X_M                 =   IS_SENSOR_UI141X_M                # VGA rolling shutter, monochrome
SENSOR_UI141X_C                 =   IS_SENSOR_UI141X_C                # VGA rolling shutter, color
SENSOR_UI144X_M                 =   IS_SENSOR_UI144X_M                # SXGA rolling shutter, monochrome
SENSOR_UI144X_C                 =   IS_SENSOR_UI144X_C                # SXGA rolling shutter, SXGA color

SENSOR_UI154X_M                 =   IS_SENSOR_UI154X_M                # SXGA rolling shutter, monochrome
SENSOR_UI154X_C                 =   IS_SENSOR_UI154X_C                # SXGA rolling shutter, color
SENSOR_UI145X_C                 =   IS_SENSOR_UI145X_C                # UXGA rolling shutter, color

SENSOR_UI146X_C                 =   IS_SENSOR_UI146X_C                # QXGA rolling shutter, color
SENSOR_UI148X_M                 =   IS_SENSOR_UI148X_M                # 5MP rolling shutter, mono
SENSOR_UI148X_C                 =   IS_SENSOR_UI148X_C                # 5MP rolling shutter, color

SENSOR_UI121X_M                 =   IS_SENSOR_UI121X_M                # VGA global shutter, monochrome
SENSOR_UI121X_C                 =   IS_SENSOR_UI121X_C                # VGA global shutter, VGA color
SENSOR_UI122X_M                 =   IS_SENSOR_UI122X_M                # WVGA global shutter, monochrome
SENSOR_UI122X_C                 =   IS_SENSOR_UI122X_C                # WVGA global shutter, color

SENSOR_UI164X_C                 =   IS_SENSOR_UI164X_C                # SXGA rolling shutter, color

SENSOR_UI155X_C                 =   IS_SENSOR_UI155X_C                # UXGA rolling shutter, color

SENSOR_UI1223_M                 =   IS_SENSOR_UI1223_M                # WVGA global shutter, monochrome
SENSOR_UI1223_C                 =   IS_SENSOR_UI1223_C                # WVGA global shutter, color

SENSOR_UI149X_M                 =   IS_SENSOR_UI149X_M                # 149x-M
SENSOR_UI149X_C                 =   IS_SENSOR_UI149X_C                # 149x-C

# LE models with xxx5
SENSOR_UI1225_M                 =   IS_SENSOR_UI1225_M                # WVGA global shutter, monochrome, LE model
SENSOR_UI1225_C                 =   IS_SENSOR_UI1225_C                # WVGA global shutter, color, LE model

SENSOR_UI1645_C                 =   IS_SENSOR_UI1645_C                # SXGA rolling shutter, color, LE model
SENSOR_UI1555_C                 =   IS_SENSOR_UI1555_C                # UXGA rolling shutter, color, LE model
SENSOR_UI1545_M                 =   IS_SENSOR_UI1545_M                # SXGA rolling shutter, monochrome, LE model
SENSOR_UI1545_C                 =   IS_SENSOR_UI1545_C                # SXGA rolling shutter, color, LE model
SENSOR_UI1455_C                 =   IS_SENSOR_UI1455_C                # UXGA rolling shutter, color, LE model
SENSOR_UI1465_C                 =   IS_SENSOR_UI1465_C                # QXGA rolling shutter, color, LE model
SENSOR_UI1485_M                 =   IS_SENSOR_UI1485_M                # 5MP rolling shutter, monochrome, LE model
SENSOR_UI1485_C                 =   IS_SENSOR_UI1485_C                # 5MP rolling shutter, color, LE model
SENSOR_UI1495_M                 =   IS_SENSOR_UI1495_M                # 149xLE-M
SENSOR_UI1495_C                 =   IS_SENSOR_UI1495_C                # 149xLE-C

# custom board level designs
SENSOR_UI1543_M                 =   IS_SENSOR_UI1543_M                # SXGA rolling shutter, monochrome, single board
SENSOR_UI1543_C                 =   IS_SENSOR_UI1543_C                # SXGA rolling shutter, color, single board

SENSOR_UI1544_M                 =   IS_SENSOR_UI1544_M                # SXGA rolling shutter, monochrome, single board
SENSOR_UI1544_C                 =   IS_SENSOR_UI1544_C                # SXGA rolling shutter, color, single board
SENSOR_UI1543_M_WO              =   IS_SENSOR_UI1543_M_WO             # SXGA rolling shutter, color, single board
SENSOR_UI1543_C_WO              =   IS_SENSOR_UI1543_C_WO             # SXGA rolling shutter, color, single board
SENSOR_UI1453_C                 =   IS_SENSOR_UI1453_C                # UXGA rolling shutter, color, single board
SENSOR_UI1463_C                 =   IS_SENSOR_UI1463_C                # QXGA rolling shutter, color, single board
SENSOR_UI1483_C                 =   IS_SENSOR_UI1483_C                # QSXGA rolling shutter, color, single board

SENSOR_UI1463_M_WO              =   IS_SENSOR_UI1463_M_WO             # QXGA rolling shutter, monochrome, single board
SENSOR_UI1463_C_WO              =   IS_SENSOR_UI1463_C_WO             # QXGA rolling shutter, color, single board

SENSOR_UI1553_C_WN              =   IS_SENSOR_UI1553_C_WN             # UXGA rolling shutter, color, single board
SENSOR_UI1483_M_WO              =   IS_SENSOR_UI1483_M_WO             # QSXGA rolling shutter, monochrome, single board
SENSOR_UI1483_C_WO              =   IS_SENSOR_UI1483_C_WO             # QSXGA rolling shutter, color, single board

# CCD Sensors
SENSOR_UI223X_M                 =   IS_SENSOR_UI223X_M                # Sony CCD sensor - XGA monochrome
SENSOR_UI223X_C                 =   IS_SENSOR_UI223X_C                # Sony CCD sensor - XGA color

SENSOR_UI241X_M                 =   IS_SENSOR_UI241X_M                # Sony CCD sensor - VGA monochrome
SENSOR_UI241X_C                 =   IS_SENSOR_UI241X_C                # Sony CCD sensor - VGA color

SENSOR_UI234X_M                 =   IS_SENSOR_UI234X_M                # Sony CCD sensor - SXGA monochrome
SENSOR_UI234X_C                 =   IS_SENSOR_UI234X_C                # Sony CCD sensor - SXGA color

#Not defines in 3.90
#SENSOR_UI233X_M                 =   IS_SENSOR_UI233X_M                # Kodak CCD sensor - 1MP mono
#SENSOR_UI233X_C                 =   IS_SENSOR_UI233X_C                # Kodak CCD sensor - 1MP color

SENSOR_UI221X_M                 =   IS_SENSOR_UI221X_M                # Sony CCD sensor - VGA monochrome
SENSOR_UI221X_C                 =   IS_SENSOR_UI221X_C                # Sony CCD sensor - VGA color

SENSOR_UI231X_M                 =   IS_SENSOR_UI231X_M                # Sony CCD sensor - VGA monochrome
SENSOR_UI231X_C                 =   IS_SENSOR_UI231X_C                # Sony CCD sensor - VGA color

SENSOR_UI222X_M                 =   IS_SENSOR_UI222X_M                # Sony CCD sensor - CCIR / PAL monochrome
SENSOR_UI222X_C                 =   IS_SENSOR_UI222X_C                # Sony CCD sensor - CCIR / PAL color

SENSOR_UI224X_M                 =   IS_SENSOR_UI224X_M                # Sony CCD sensor - SXGA monochrome
SENSOR_UI224X_C                 =   IS_SENSOR_UI224X_C                # Sony CCD sensor - SXGA color

SENSOR_UI225X_M                 =   IS_SENSOR_UI225X_M                # Sony CCD sensor - UXGA monochrome
SENSOR_UI225X_C                 =   IS_SENSOR_UI225X_C                # Sony CCD sensor - UXGA color

SENSOR_UI214X_M                 =   IS_SENSOR_UI214X_M                # Sony CCD sensor - SXGA monochrome
SENSOR_UI214X_C                 =   IS_SENSOR_UI214X_C                # Sony CCD sensor - SXGA color

# ----------------------------------------------------------------------------
# error codes
# ----------------------------------------------------------------------------
NO_SUCCESS                      =   IS_NO_SUCCESS                           # function call failed
SUCCESS                         =   IS_SUCCESS                              # function call succeeded
INVALID_CAMERA_HANDLE           =   IS_INVALID_CAMERA_HANDLE                # camera handle is not valid or zero
INVALID_HANDLE                  =   IS_INVALID_HANDLE                       # a handle other than the camera handle is invalid

IO_REQUEST_FAILED               =   IS_IO_REQUEST_FAILED                     # an io request to the driver failed
CANT_OPEN_DEVICE                =   IS_CANT_OPEN_DEVICE                      # returned by is_InitCamera
CANT_CLOSE_DEVICE               =   IS_CANT_CLOSE_DEVICE                  
CANT_SETUP_MEMORY               =   IS_CANT_SETUP_MEMORY                  
NO_HWND_FOR_ERROR_REPORT        =   IS_NO_HWND_FOR_ERROR_REPORT           
ERROR_MESSAGE_NOT_CREATED       =   IS_ERROR_MESSAGE_NOT_CREATED          
ERROR_STRING_NOT_FOUND          =   IS_ERROR_STRING_NOT_FOUND             
HOOK_NOT_CREATED                =   IS_HOOK_NOT_CREATED                   
TIMER_NOT_CREATED               =   IS_TIMER_NOT_CREATED                 
CANT_OPEN_REGISTRY              =   IS_CANT_OPEN_REGISTRY                
CANT_READ_REGISTRY              =   IS_CANT_READ_REGISTRY                
CANT_VALIDATE_BOARD             =   IS_CANT_VALIDATE_BOARD               
CANT_GIVE_BOARD_ACCESS          =   IS_CANT_GIVE_BOARD_ACCESS            
NO_IMAGE_MEM_ALLOCATED          =   IS_NO_IMAGE_MEM_ALLOCATED            
CANT_CLEANUP_MEMORY             =   IS_CANT_CLEANUP_MEMORY               
CANT_COMMUNICATE_WITH_DRIVER    =   IS_CANT_COMMUNICATE_WITH_DRIVER      
FUNCTION_NOT_SUPPORTED_YET      =   IS_FUNCTION_NOT_SUPPORTED_YET        
OPERATING_SYSTEM_NOT_SUPPORTED  =   IS_OPERATING_SYSTEM_NOT_SUPPORTED    

INVALID_VIDEO_IN                =   IS_INVALID_VIDEO_IN                  
INVALID_IMG_SIZE                =   IS_INVALID_IMG_SIZE                  
INVALID_ADDRESS                 =   IS_INVALID_ADDRESS                   
INVALID_VIDEO_MODE              =   IS_INVALID_VIDEO_MODE                
INVALID_AGC_MODE                =   IS_INVALID_AGC_MODE                  
INVALID_GAMMA_MODE              =   IS_INVALID_GAMMA_MODE                
INVALID_SYNC_LEVEL              =   IS_INVALID_SYNC_LEVEL                
INVALID_CBARS_MODE              =   IS_INVALID_CBARS_MODE                
INVALID_COLOR_MODE              =   IS_INVALID_COLOR_MODE                
INVALID_SCALE_FACTOR            =   IS_INVALID_SCALE_FACTOR              
INVALID_IMAGE_SIZE              =   IS_INVALID_IMAGE_SIZE                
INVALID_IMAGE_POS               =   IS_INVALID_IMAGE_POS                 
INVALID_CAPTURE_MODE            =   IS_INVALID_CAPTURE_MODE              
INVALID_RISC_PROGRAM            =   IS_INVALID_RISC_PROGRAM              
INVALID_BRIGHTNESS              =   IS_INVALID_BRIGHTNESS                
INVALID_CONTRAST                =   IS_INVALID_CONTRAST                  
INVALID_SATURATION_U            =   IS_INVALID_SATURATION_U              
INVALID_SATURATION_V            =   IS_INVALID_SATURATION_V              
INVALID_HUE                     =   IS_INVALID_HUE                       
INVALID_HOR_FILTER_STEP         =   IS_INVALID_HOR_FILTER_STEP           
INVALID_VERT_FILTER_STEP        =   IS_INVALID_VERT_FILTER_STEP          
INVALID_EEPROM_READ_ADDRESS     =   IS_INVALID_EEPROM_READ_ADDRESS       
INVALID_EEPROM_WRITE_ADDRESS    =   IS_INVALID_EEPROM_WRITE_ADDRESS      
INVALID_EEPROM_READ_LENGTH      =   IS_INVALID_EEPROM_READ_LENGTH        
INVALID_EEPROM_WRITE_LENGTH     =   IS_INVALID_EEPROM_WRITE_LENGTH       
INVALID_BOARD_INFO_POINTER      =   IS_INVALID_BOARD_INFO_POINTER        
INVALID_DISPLAY_MODE            =   IS_INVALID_DISPLAY_MODE              
INVALID_ERR_REP_MODE            =   IS_INVALID_ERR_REP_MODE              
INVALID_BITS_PIXEL              =   IS_INVALID_BITS_PIXEL                
INVALID_MEMORY_POINTER          =   IS_INVALID_MEMORY_POINTER            

FILE_WRITE_OPEN_ERROR           =   IS_FILE_WRITE_OPEN_ERROR             
FILE_READ_OPEN_ERROR            =   IS_FILE_READ_OPEN_ERROR              
FILE_READ_INVALID_BMP_ID        =   IS_FILE_READ_INVALID_BMP_ID          
FILE_READ_INVALID_BMP_SIZE      =   IS_FILE_READ_INVALID_BMP_SIZE        
FILE_READ_INVALID_BIT_COUNT     =   IS_FILE_READ_INVALID_BIT_COUNT       
WRONG_KERNEL_VERSION            =   IS_WRONG_KERNEL_VERSION              

RISC_INVALID_XLENGTH            =   IS_RISC_INVALID_XLENGTH              
RISC_INVALID_YLENGTH            =   IS_RISC_INVALID_YLENGTH              
RISC_EXCEED_IMG_SIZE            =   IS_RISC_EXCEED_IMG_SIZE              

# DirectDraw Mode errors
DD_MAIN_FAILED                  =   IS_DD_MAIN_FAILED                    
DD_PRIMSURFACE_FAILED           =   IS_DD_PRIMSURFACE_FAILED             
DD_SCRN_SIZE_NOT_SUPPORTED      =   IS_DD_SCRN_SIZE_NOT_SUPPORTED        
DD_CLIPPER_FAILED               =   IS_DD_CLIPPER_FAILED                 
DD_CLIPPER_HWND_FAILED          =   IS_DD_CLIPPER_HWND_FAILED            
DD_CLIPPER_CONNECT_FAILED       =   IS_DD_CLIPPER_CONNECT_FAILED         
DD_BACKSURFACE_FAILED           =   IS_DD_BACKSURFACE_FAILED             
DD_BACKSURFACE_IN_SYSMEM        =   IS_DD_BACKSURFACE_IN_SYSMEM          
DD_MDL_MALLOC_ERR               =   IS_DD_MDL_MALLOC_ERR                 
DD_MDL_SIZE_ERR                 =   IS_DD_MDL_SIZE_ERR                   
DD_CLIP_NO_CHANGE               =   IS_DD_CLIP_NO_CHANGE                 
DD_PRIMMEM_NULL                 =   IS_DD_PRIMMEM_NULL                   
DD_BACKMEM_NULL                 =   IS_DD_BACKMEM_NULL                   
DD_BACKOVLMEM_NULL              =   IS_DD_BACKOVLMEM_NULL                
DD_OVERLAYSURFACE_FAILED        =   IS_DD_OVERLAYSURFACE_FAILED          
DD_OVERLAYSURFACE_IN_SYSMEM     =   IS_DD_OVERLAYSURFACE_IN_SYSMEM       
DD_OVERLAY_NOT_ALLOWED          =   IS_DD_OVERLAY_NOT_ALLOWED            
DD_OVERLAY_COLKEY_ERR           =   IS_DD_OVERLAY_COLKEY_ERR             
DD_OVERLAY_NOT_ENABLED          =   IS_DD_OVERLAY_NOT_ENABLED            
DD_GET_DC_ERROR                 =   IS_DD_GET_DC_ERROR                   
DD_DDRAW_DLL_NOT_LOADED         =   IS_DD_DDRAW_DLL_NOT_LOADED           
DD_THREAD_NOT_CREATED           =   IS_DD_THREAD_NOT_CREATED             
DD_CANT_GET_CAPS                =   IS_DD_CANT_GET_CAPS                  
DD_NO_OVERLAYSURFACE            =   IS_DD_NO_OVERLAYSURFACE              
DD_NO_OVERLAYSTRETCH            =   IS_DD_NO_OVERLAYSTRETCH              
DD_CANT_CREATE_OVERLAYSURFACE   =   IS_DD_CANT_CREATE_OVERLAYSURFACE     
DD_CANT_UPDATE_OVERLAYSURFACE   =   IS_DD_CANT_UPDATE_OVERLAYSURFACE     
DD_INVALID_STRETCH              =   IS_DD_INVALID_STRETCH                

EV_INVALID_EVENT_NUMBER         =   IS_EV_INVALID_EVENT_NUMBER          
INVALID_MODE                    =   IS_INVALID_MODE                     
CANT_FIND_FALCHOOK              =   IS_CANT_FIND_FALCHOOK               
CANT_FIND_HOOK                  =   IS_CANT_FIND_HOOK                   
CANT_GET_HOOK_PROC_ADDR         =   IS_CANT_GET_HOOK_PROC_ADDR          
CANT_CHAIN_HOOK_PROC            =   IS_CANT_CHAIN_HOOK_PROC             
CANT_SETUP_WND_PROC             =   IS_CANT_SETUP_WND_PROC              
HWND_NULL                       =   IS_HWND_NULL                        
INVALID_UPDATE_MODE             =   IS_INVALID_UPDATE_MODE              
NO_ACTIVE_IMG_MEM               =   IS_NO_ACTIVE_IMG_MEM                
CANT_INIT_EVENT                 =   IS_CANT_INIT_EVENT                  
FUNC_NOT_AVAIL_IN_OS            =   IS_FUNC_NOT_AVAIL_IN_OS             
CAMERA_NOT_CONNECTED            =   IS_CAMERA_NOT_CONNECTED             
SEQUENCE_LIST_EMPTY             =   IS_SEQUENCE_LIST_EMPTY              
CANT_ADD_TO_SEQUENCE            =   IS_CANT_ADD_TO_SEQUENCE             
LOW_OF_SEQUENCE_RISC_MEM        =   IS_LOW_OF_SEQUENCE_RISC_MEM         
IMGMEM2FREE_USED_IN_SEQ         =   IS_IMGMEM2FREE_USED_IN_SEQ          
IMGMEM_NOT_IN_SEQUENCE_LIST     =   IS_IMGMEM_NOT_IN_SEQUENCE_LIST      
SEQUENCE_BUF_ALREADY_LOCKED     =   IS_SEQUENCE_BUF_ALREADY_LOCKED      
INVALID_DEVICE_ID               =   IS_INVALID_DEVICE_ID                
INVALID_BOARD_ID                =   IS_INVALID_BOARD_ID                 
ALL_DEVICES_BUSY                =   IS_ALL_DEVICES_BUSY                 
HOOK_BUSY                       =   IS_HOOK_BUSY                        
TIMED_OUT                       =   IS_TIMED_OUT                        
NULL_POINTER                    =   IS_NULL_POINTER                     
WRONG_HOOK_VERSION              =   IS_WRONG_HOOK_VERSION               
INVALID_PARAMETER               =   IS_INVALID_PARAMETER                   # a parameter specified was invalid
NOT_ALLOWED                     =   IS_NOT_ALLOWED                      
OUT_OF_MEMORY                   =   IS_OUT_OF_MEMORY                    
INVALID_WHILE_LIVE              =   IS_INVALID_WHILE_LIVE               
ACCESS_VIOLATION                =   IS_ACCESS_VIOLATION                    # an internal exception occurred
UNKNOWN_ROP_EFFECT              =   IS_UNKNOWN_ROP_EFFECT               
INVALID_RENDER_MODE             =   IS_INVALID_RENDER_MODE              
INVALID_THREAD_CONTEXT          =   IS_INVALID_THREAD_CONTEXT           
NO_HARDWARE_INSTALLED           =   IS_NO_HARDWARE_INSTALLED            
INVALID_WATCHDOG_TIME           =   IS_INVALID_WATCHDOG_TIME            
INVALID_WATCHDOG_MODE           =   IS_INVALID_WATCHDOG_MODE            
INVALID_PASSTHROUGH_IN          =   IS_INVALID_PASSTHROUGH_IN           
ERROR_SETTING_PASSTHROUGH_IN    =   IS_ERROR_SETTING_PASSTHROUGH_IN     
FAILURE_ON_SETTING_WATCHDOG     =   IS_FAILURE_ON_SETTING_WATCHDOG      
NO_USB20                        =   IS_NO_USB20                            # the usb port doesnt support usb 2.0
CAPTURE_RUNNING                 =   IS_CAPTURE_RUNNING                     # there is already a capture running

MEMORY_BOARD_ACTIVATED          =   IS_MEMORY_BOARD_ACTIVATED              # operation could not execute while mboard is enabled
MEMORY_BOARD_DEACTIVATED        =   IS_MEMORY_BOARD_DEACTIVATED            # operation could not execute while mboard is disabled
NO_MEMORY_BOARD_CONNECTED       =   IS_NO_MEMORY_BOARD_CONNECTED           # no memory board connected
TOO_LESS_MEMORY                 =   IS_TOO_LESS_MEMORY                     # image size is above memory capacity
IMAGE_NOT_PRESENT               =   IS_IMAGE_NOT_PRESENT                   # requested image is no longer present in the camera
MEMORY_MODE_RUNNING             =   IS_MEMORY_MODE_RUNNING              
MEMORYBOARD_DISABLED            =   IS_MEMORYBOARD_DISABLED             

TRIGGER_ACTIVATED               =   IS_TRIGGER_ACTIVATED                   # operation could not execute while trigger is enabled
WRONG_KEY                       =   IS_WRONG_KEY                        
CRC_ERROR                       =   IS_CRC_ERROR                        
NOT_YET_RELEASED                =   IS_NOT_YET_RELEASED                    # this feature is not available yet
NOT_CALIBRATED                  =   IS_NOT_CALIBRATED                      # the camera is not calibrated
WAITING_FOR_KERNEL              =   IS_WAITING_FOR_KERNEL                  # a request to the kernel exceeded
NOT_SUPPORTED                   =   IS_NOT_SUPPORTED                       # operation mode is not supported
TRIGGER_NOT_ACTIVATED           =   IS_TRIGGER_NOT_ACTIVATED               # operation could not execute while trigger is disabled
OPERATION_ABORTED               =   IS_OPERATION_ABORTED                
BAD_STRUCTURE_SIZE              =   IS_BAD_STRUCTURE_SIZE               
INVALID_BUFFER_SIZE             =   IS_INVALID_BUFFER_SIZE              
INVALID_PIXEL_CLOCK             =   IS_INVALID_PIXEL_CLOCK              
INVALID_EXPOSURE_TIME           =   IS_INVALID_EXPOSURE_TIME            
AUTO_EXPOSURE_RUNNING           =   IS_AUTO_EXPOSURE_RUNNING            
CANNOT_CREATE_BB_SURF           =   IS_CANNOT_CREATE_BB_SURF               # error creating backbuffer surface  
CANNOT_CREATE_BB_MIX            =   IS_CANNOT_CREATE_BB_MIX                # backbuffer mixer surfaces can not be created
BB_OVLMEM_NULL                  =   IS_BB_OVLMEM_NULL                      # backbuffer overlay mem could not be locked  
CANNOT_CREATE_BB_OVL            =   IS_CANNOT_CREATE_BB_OVL                # backbuffer overlay mem could not be created  
NOT_SUPP_IN_OVL_SURF_MODE       =   IS_NOT_SUPP_IN_OVL_SURF_MODE           # function not supported in overlay surface mode  
INVALID_SURFACE                 =   IS_INVALID_SURFACE                     # surface invalid
SURFACE_LOST                    =   IS_SURFACE_LOST                        # surface has been lost  
RELEASE_BB_OVL_DC               =   IS_RELEASE_BB_OVL_DC                   # error releasing backbuffer overlay DC  
BB_TIMER_NOT_CREATED            =   IS_BB_TIMER_NOT_CREATED                # backbuffer timer could not be created  
BB_OVL_NOT_EN                   =   IS_BB_OVL_NOT_EN                       # backbuffer overlay has not been enabled  
ONLY_IN_BB_MODE                 =   IS_ONLY_IN_BB_MODE                     # only possible in backbuffer mode 
INVALID_COLOR_FORMAT            =   IS_INVALID_COLOR_FORMAT                # invalid color format
INVALID_WB_BINNING_MODE         =   IS_INVALID_WB_BINNING_MODE             # invalid binning mode for AWB 
INVALID_I2C_DEVICE_ADDRESS      =   IS_INVALID_I2C_DEVICE_ADDRESS          # invalid I2C device address
COULD_NOT_CONVERT               =   IS_COULD_NOT_CONVERT                   # current image couldn't be converted
TRANSFER_ERROR                  =   IS_TRANSFER_ERROR                      # transfer failed
PARAMETER_SET_NOT_PRESENT       =   IS_PARAMETER_SET_NOT_PRESENT           # the parameter set is not present
INVALID_CAMERA_TYPE             =   IS_INVALID_CAMERA_TYPE                 # the camera type in the ini file doesn't match
INVALID_HOST_IP_HIBYTE          =   IS_INVALID_HOST_IP_HIBYTE              # HIBYTE of host address is invalid
CM_NOT_SUPP_IN_CURR_DISPLAYMODE =   IS_CM_NOT_SUPP_IN_CURR_DISPLAYMODE     # color mode is not supported in the current display mode
NO_IR_FILTER                    =   IS_NO_IR_FILTER                     
STARTER_FW_UPLOAD_NEEDED        =   IS_STARTER_FW_UPLOAD_NEEDED            # device starter firmware is not compatible    

DR_LIBRARY_NOT_FOUND            =   IS_DR_LIBRARY_NOT_FOUND                # the DirectRender library could not be found
DR_DEVICE_OUT_OF_MEMORY         =   IS_DR_DEVICE_OUT_OF_MEMORY             # insufficient graphics adapter video memory
DR_CANNOT_CREATE_SURFACE        =   IS_DR_CANNOT_CREATE_SURFACE            # the image or overlay surface could not be created
DR_CANNOT_CREATE_VERTEX_BUFFER  =   IS_DR_CANNOT_CREATE_VERTEX_BUFFER      # the vertex buffer could not be created
DR_CANNOT_CREATE_TEXTURE        =   IS_DR_CANNOT_CREATE_TEXTURE            # the texture could not be created  
DR_CANNOT_LOCK_OVERLAY_SURFACE  =   IS_DR_CANNOT_LOCK_OVERLAY_SURFACE      # the overlay surface could not be locked
DR_CANNOT_UNLOCK_OVERLAY_SURFACE=   IS_DR_CANNOT_UNLOCK_OVERLAY_SURFACE    # the overlay surface could not be unlocked
DR_CANNOT_GET_OVERLAY_DC        =   IS_DR_CANNOT_GET_OVERLAY_DC            # cannot get the overlay surface DC 
DR_CANNOT_RELEASE_OVERLAY_DC    =   IS_DR_CANNOT_RELEASE_OVERLAY_DC        # cannot release the overlay surface DC
DR_DEVICE_CAPS_INSUFFICIENT     =   IS_DR_DEVICE_CAPS_INSUFFICIENT         # insufficient graphics adapter capabilities

# ----------------------------------------------------------------------------
# common definitions
# ----------------------------------------------------------------------------
OFF                             =   IS_OFF                              
ON                              =   IS_ON                               
IGNORE_PARAMETER                =   IS_IGNORE_PARAMETER                 


# ----------------------------------------------------------------------------
#  device enumeration
# ----------------------------------------------------------------------------
USE_DEVICE_ID                   =   IS_USE_DEVICE_ID                    
ALLOW_STARTER_FW_UPLOAD         =   IS_ALLOW_STARTER_FW_UPLOAD          

# ----------------------------------------------------------------------------
# AutoExit enable/disable
# ----------------------------------------------------------------------------
GET_AUTO_EXIT_ENABLED           =   IS_GET_AUTO_EXIT_ENABLED            
DISABLE_AUTO_EXIT               =   IS_DISABLE_AUTO_EXIT                
ENABLE_AUTO_EXIT                =   IS_ENABLE_AUTO_EXIT                 


# ----------------------------------------------------------------------------
# live/freeze parameters
# ----------------------------------------------------------------------------
GET_LIVE                        =   IS_GET_LIVE                         

WAIT                            =   IS_WAIT                             
DONT_WAIT                       =   IS_DONT_WAIT                        
FORCE_VIDEO_STOP                =   IS_FORCE_VIDEO_STOP                 
FORCE_VIDEO_START               =   IS_FORCE_VIDEO_START                
USE_NEXT_MEM                    =   IS_USE_NEXT_MEM                     


# ----------------------------------------------------------------------------
# video finish constants
# ----------------------------------------------------------------------------
VIDEO_NOT_FINISH                =   IS_VIDEO_NOT_FINISH                 
VIDEO_FINISH                    =   IS_VIDEO_FINISH                     


# ----------------------------------------------------------------------------
# bitmap render modes
# ----------------------------------------------------------------------------
GET_RENDER_MODE                 =   IS_GET_RENDER_MODE                  

RENDER_DISABLED                 =   IS_RENDER_DISABLED                  
RENDER_NORMAL                   =   IS_RENDER_NORMAL                    
RENDER_FIT_TO_WINDOW            =   IS_RENDER_FIT_TO_WINDOW             
RENDER_DOWNSCALE_1_2            =   IS_RENDER_DOWNSCALE_1_2             
RENDER_MIRROR_UPDOWN            =   IS_RENDER_MIRROR_UPDOWN             

# Not defined in 3.90
#RENDER_DOUBLE_HEIGHT            =   IS_RENDER_DOUBLE_HEIGHT             
#RENDER_HALF_HEIGHT              =   IS_RENDER_HALF_HEIGHT               


# ----------------------------------------------------------------------------
# external trigger modes
# ----------------------------------------------------------------------------
GET_EXTERNALTRIGGER             =   IS_GET_EXTERNALTRIGGER              
GET_TRIGGER_STATUS              =   IS_GET_TRIGGER_STATUS               
GET_TRIGGER_MASK                =   IS_GET_TRIGGER_MASK                 
GET_TRIGGER_INPUTS              =   IS_GET_TRIGGER_INPUTS               
GET_SUPPORTED_TRIGGER_MODE      =   IS_GET_SUPPORTED_TRIGGER_MODE       
GET_TRIGGER_COUNTER             =   IS_GET_TRIGGER_COUNTER              



# old defines for compatibility 

# Not defined in 3.90
#SET_TRIG_OFF                    =   IS_SET_TRIG_OFF                     
#SET_TRIG_HI_LO                  =   IS_SET_TRIG_HI_LO                   
#SET_TRIG_LO_HI                  =   IS_SET_TRIG_LO_HI                   
#SET_TRIG_SOFTWARE               =   IS_SET_TRIG_SOFTWARE                
#SET_TRIG_HI_LO_SYNC             =   IS_SET_TRIG_HI_LO_SYNC              
#SET_TRIG_LO_HI_SYNC             =   IS_SET_TRIG_LO_HI_SYNC              

#SET_TRIG_MASK                   =   IS_SET_TRIG_MASK                    

# New defines
SET_TRIGGER_CONTINUOUS          =   IS_SET_TRIGGER_CONTINUOUS           
SET_TRIGGER_OFF                 =   IS_SET_TRIGGER_OFF                  
SET_TRIGGER_HI_LO               =   IS_SET_TRIGGER_HI_LO                 
SET_TRIGGER_LO_HI               =   IS_SET_TRIGGER_LO_HI                 
SET_TRIGGER_SOFTWARE            =   IS_SET_TRIGGER_SOFTWARE              
SET_TRIGGER_HI_LO_SYNC          =   IS_SET_TRIGGER_HI_LO_SYNC           
SET_TRIGGER_LO_HI_SYNC          =   IS_SET_TRIGGER_LO_HI_SYNC           


GET_TRIGGER_DELAY               =   IS_GET_TRIGGER_DELAY                
GET_MIN_TRIGGER_DELAY           =   IS_GET_MIN_TRIGGER_DELAY            
GET_MAX_TRIGGER_DELAY           =   IS_GET_MAX_TRIGGER_DELAY            
GET_TRIGGER_DELAY_GRANULARITY   =   IS_GET_TRIGGER_DELAY_GRANULARITY    


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
# pixelclock
GET_PIXEL_CLOCK                 =   IS_GET_PIXEL_CLOCK                  
GET_DEFAULT_PIXEL_CLK           =   IS_GET_DEFAULT_PIXEL_CLK            
# frame rate
GET_FRAMERATE                   =   IS_GET_FRAMERATE                    
GET_DEFAULT_FRAMERATE           =   IS_GET_DEFAULT_FRAMERATE            
# exposure
#Not defined in 3.90
#GET_EXPOSURE_TIME               =   IS_GET_EXPOSURE_TIME                
#GET_DEFAULT_EXPOSURE            =   IS_GET_DEFAULT_EXPOSURE             
         

# ----------------------------------------------------------------------------
# Gain definitions
# ----------------------------------------------------------------------------
GET_MASTER_GAIN                 =   IS_GET_MASTER_GAIN                  
GET_RED_GAIN                    =   IS_GET_RED_GAIN                     
GET_GREEN_GAIN                  =   IS_GET_GREEN_GAIN                   
GET_BLUE_GAIN                   =   IS_GET_BLUE_GAIN                    
GET_DEFAULT_MASTER              =   IS_GET_DEFAULT_MASTER               
GET_DEFAULT_RED                 =   IS_GET_DEFAULT_RED                  
GET_DEFAULT_GREEN               =   IS_GET_DEFAULT_GREEN                
GET_DEFAULT_BLUE                =   IS_GET_DEFAULT_BLUE                 
GET_GAINBOOST                   =   IS_GET_GAINBOOST                    
SET_GAINBOOST_ON                =   IS_SET_GAINBOOST_ON                 
SET_GAINBOOST_OFF               =   IS_SET_GAINBOOST_OFF                
GET_SUPPORTED_GAINBOOST         =   IS_GET_SUPPORTED_GAINBOOST          
MIN_GAIN                        =   IS_MIN_GAIN                         
MAX_GAIN                        =   IS_MAX_GAIN                         


# ----------------------------------------------------------------------------
# Gain factor definitions
# ----------------------------------------------------------------------------
GET_MASTER_GAIN_FACTOR          =   IS_GET_MASTER_GAIN_FACTOR           
GET_RED_GAIN_FACTOR             =   IS_GET_RED_GAIN_FACTOR              
GET_GREEN_GAIN_FACTOR           =   IS_GET_GREEN_GAIN_FACTOR            
GET_BLUE_GAIN_FACTOR            =   IS_GET_BLUE_GAIN_FACTOR             
SET_MASTER_GAIN_FACTOR          =   IS_SET_MASTER_GAIN_FACTOR           
SET_RED_GAIN_FACTOR             =   IS_SET_RED_GAIN_FACTOR              
SET_GREEN_GAIN_FACTOR           =   IS_SET_GREEN_GAIN_FACTOR            
SET_BLUE_GAIN_FACTOR            =   IS_SET_BLUE_GAIN_FACTOR             
GET_DEFAULT_MASTER_GAIN_FACTOR  =   IS_GET_DEFAULT_MASTER_GAIN_FACTOR   
GET_DEFAULT_RED_GAIN_FACTOR     =   IS_GET_DEFAULT_RED_GAIN_FACTOR      
GET_DEFAULT_GREEN_GAIN_FACTOR   =   IS_GET_DEFAULT_GREEN_GAIN_FACTOR    
GET_DEFAULT_BLUE_GAIN_FACTOR    =   IS_GET_DEFAULT_BLUE_GAIN_FACTOR     
INQUIRE_MASTER_GAIN_FACTOR      =   IS_INQUIRE_MASTER_GAIN_FACTOR       
INQUIRE_RED_GAIN_FACTOR         =   IS_INQUIRE_RED_GAIN_FACTOR          
INQUIRE_GREEN_GAIN_FACTOR       =   IS_INQUIRE_GREEN_GAIN_FACTOR        
INQUIRE_BLUE_GAIN_FACTOR        =   IS_INQUIRE_BLUE_GAIN_FACTOR         


# ----------------------------------------------------------------------------
# Global Shutter definitions
# ----------------------------------------------------------------------------
SET_GLOBAL_SHUTTER_ON           =   IS_SET_GLOBAL_SHUTTER_ON            
SET_GLOBAL_SHUTTER_OFF          =   IS_SET_GLOBAL_SHUTTER_OFF           
GET_GLOBAL_SHUTTER              =   IS_GET_GLOBAL_SHUTTER               
GET_SUPPORTED_GLOBAL_SHUTTER    =   IS_GET_SUPPORTED_GLOBAL_SHUTTER     


# ----------------------------------------------------------------------------
# Black level definitions
# ----------------------------------------------------------------------------
GET_BL_COMPENSATION             =   IS_GET_BL_COMPENSATION              
GET_BL_OFFSET                   =   IS_GET_BL_OFFSET                    
GET_BL_DEFAULT_MODE             =   IS_GET_BL_DEFAULT_MODE              
GET_BL_DEFAULT_OFFSET           =   IS_GET_BL_DEFAULT_OFFSET            
GET_BL_SUPPORTED_MODE           =   IS_GET_BL_SUPPORTED_MODE            

BL_COMPENSATION_DISABLE         =   IS_BL_COMPENSATION_DISABLE          
BL_COMPENSATION_ENABLE          =   IS_BL_COMPENSATION_ENABLE           
BL_COMPENSATION_OFFSET          =   IS_BL_COMPENSATION_OFFSET           

MIN_BL_OFFSET                   =   IS_MIN_BL_OFFSET                    
MAX_BL_OFFSET                   =   IS_MAX_BL_OFFSET                    

# ----------------------------------------------------------------------------
# hardware gamma definitions
# ----------------------------------------------------------------------------
GET_HW_GAMMA                    =   IS_GET_HW_GAMMA                     
GET_HW_SUPPORTED_GAMMA          =   IS_GET_HW_SUPPORTED_GAMMA           

SET_HW_GAMMA_OFF                =   IS_SET_HW_GAMMA_OFF                 
SET_HW_GAMMA_ON                 =   IS_SET_HW_GAMMA_ON                  

# ----------------------------------------------------------------------------
# camera LUT
# ----------------------------------------------------------------------------
ENABLE_CAMERA_LUT               =   IS_ENABLE_CAMERA_LUT                
SET_CAMERA_LUT_VALUES           =   IS_SET_CAMERA_LUT_VALUES            
ENABLE_RGB_GRAYSCALE            =   IS_ENABLE_RGB_GRAYSCALE             
GET_CAMERA_LUT_USER             =   IS_GET_CAMERA_LUT_USER              
GET_CAMERA_LUT_COMPLETE         =   IS_GET_CAMERA_LUT_COMPLETE          

# ----------------------------------------------------------------------------
# camera LUT presets
# ----------------------------------------------------------------------------
CAMERA_LUT_IDENTITY             =   IS_CAMERA_LUT_IDENTITY              
CAMERA_LUT_NEGATIV              =   IS_CAMERA_LUT_NEGATIV               
CAMERA_LUT_GLOW1                =   IS_CAMERA_LUT_GLOW1                 
CAMERA_LUT_GLOW2                =   IS_CAMERA_LUT_GLOW2                 
CAMERA_LUT_ASTRO1               =   IS_CAMERA_LUT_ASTRO1                
CAMERA_LUT_RAINBOW1             =   IS_CAMERA_LUT_RAINBOW1              
CAMERA_LUT_MAP1                 =   IS_CAMERA_LUT_MAP1                  
CAMERA_LUT_COLD_HOT             =   IS_CAMERA_LUT_COLD_HOT              
CAMERA_LUT_SEPIC                =   IS_CAMERA_LUT_SEPIC                 
CAMERA_LUT_ONLY_RED             =   IS_CAMERA_LUT_ONLY_RED              
CAMERA_LUT_ONLY_GREEN           =   IS_CAMERA_LUT_ONLY_GREEN            
CAMERA_LUT_ONLY_BLUE            =   IS_CAMERA_LUT_ONLY_BLUE             

CAMERA_LUT_64                   =   IS_CAMERA_LUT_64                    
CAMERA_LUT_128                  =   IS_CAMERA_LUT_128                   


# ----------------------------------------------------------------------------
# image parameters
# ----------------------------------------------------------------------------
# brightness


#Not defined in 3.90
#GET_BRIGHTNESS                  =   IS_GET_BRIGHTNESS                   
#MIN_BRIGHTNESS                  =   IS_MIN_BRIGHTNESS                   
#MAX_BRIGHTNESS                  =   IS_MAX_BRIGHTNESS                   
#DEFAULT_BRIGHTNESS              =   IS_DEFAULT_BRIGHTNESS               

# contrast
#Not defined in 3.90
#GET_CONTRAST                    =   IS_GET_CONTRAST                     
#MIN_CONTRAST                    =   IS_MIN_CONTRAST                     
#MAX_CONTRAST                    =   IS_MAX_CONTRAST                     
#DEFAULT_CONTRAST                =   IS_DEFAULT_CONTRAST                 
# gamma
GET_GAMMA                       =   IS_GET_GAMMA                        
MIN_GAMMA                       =   IS_MIN_GAMMA                        
MAX_GAMMA                       =   IS_MAX_GAMMA                        
DEFAULT_GAMMA                   =   IS_DEFAULT_GAMMA                    
# saturation   (Falcon)
GET_SATURATION_U                =   IS_GET_SATURATION_U                 
MIN_SATURATION_U                =   IS_MIN_SATURATION_U                 
MAX_SATURATION_U                =   IS_MAX_SATURATION_U                 
DEFAULT_SATURATION_U            =   IS_DEFAULT_SATURATION_U             
GET_SATURATION_V                =   IS_GET_SATURATION_V                 
MIN_SATURATION_V                =   IS_MIN_SATURATION_V                 
MAX_SATURATION_V                =   IS_MAX_SATURATION_V                 
DEFAULT_SATURATION_V            =   IS_DEFAULT_SATURATION_V             
# hue  (Falcon)
#Not defined in 3.90
#GET_HUE                         =   IS_GET_HUE                          
#MIN_HUE                         =   IS_MIN_HUE                          
#MAX_HUE                         =   IS_MAX_HUE                          
#DEFAULT_HUE                     =   IS_DEFAULT_HUE                      


# ----------------------------------------------------------------------------
# Image position and size
# ----------------------------------------------------------------------------
#Not defined in 3.90
#GET_IMAGE_SIZE_X                =   IS_GET_IMAGE_SIZE_X                 
#GET_IMAGE_SIZE_Y                =   IS_GET_IMAGE_SIZE_Y                 
#GET_IMAGE_SIZE_X_INC            =   IS_GET_IMAGE_SIZE_X_INC             
#GET_IMAGE_SIZE_Y_INC            =   IS_GET_IMAGE_SIZE_Y_INC             
#GET_IMAGE_SIZE_X_MIN            =   IS_GET_IMAGE_SIZE_X_MIN             
#GET_IMAGE_SIZE_Y_MIN            =   IS_GET_IMAGE_SIZE_Y_MIN             
#GET_IMAGE_SIZE_X_MAX            =   IS_GET_IMAGE_SIZE_X_MAX             
#GET_IMAGE_SIZE_Y_MAX            =   IS_GET_IMAGE_SIZE_Y_MAX             

#GET_IMAGE_POS_X                 =   IS_GET_IMAGE_POS_X                  
#GET_IMAGE_POS_Y                 =   IS_GET_IMAGE_POS_Y                  
#GET_IMAGE_POS_X_ABS             =   IS_GET_IMAGE_POS_X_ABS              
#GET_IMAGE_POS_Y_ABS             =   IS_GET_IMAGE_POS_Y_ABS              
#GET_IMAGE_POS_X_INC             =   IS_GET_IMAGE_POS_X_INC              
#GET_IMAGE_POS_Y_INC             =   IS_GET_IMAGE_POS_Y_INC              
#GET_IMAGE_POS_X_MIN             =   IS_GET_IMAGE_POS_X_MIN              
#GET_IMAGE_POS_Y_MIN             =   IS_GET_IMAGE_POS_Y_MIN              
#GET_IMAGE_POS_X_MAX             =   IS_GET_IMAGE_POS_X_MAX              
#GET_IMAGE_POS_Y_MAX             =   IS_GET_IMAGE_POS_Y_MAX              

#SET_IMAGE_POS_X_ABS             =   IS_SET_IMAGE_POS_X_ABS              
#SET_IMAGE_POS_Y_ABS             =   IS_SET_IMAGE_POS_Y_ABS              

# Compatibility
#SET_IMAGEPOS_X_ABS              =   IS_SET_IMAGEPOS_X_ABS               
#SET_IMAGEPOS_Y_ABS              =   IS_SET_IMAGEPOS_Y_ABS               


# ----------------------------------------------------------------------------
# ROP effect constants
# ----------------------------------------------------------------------------
GET_ROP_EFFECT                  =   IS_GET_ROP_EFFECT                   
GET_SUPPORTED_ROP_EFFECT        =   IS_GET_SUPPORTED_ROP_EFFECT         

SET_ROP_NONE                    =   IS_SET_ROP_NONE                     
SET_ROP_MIRROR_UPDOWN           =   IS_SET_ROP_MIRROR_UPDOWN            
SET_ROP_MIRROR_UPDOWN_ODD       =   IS_SET_ROP_MIRROR_UPDOWN_ODD        
SET_ROP_MIRROR_UPDOWN_EVEN      =   IS_SET_ROP_MIRROR_UPDOWN_EVEN       
SET_ROP_MIRROR_LEFTRIGHT        =   IS_SET_ROP_MIRROR_LEFTRIGHT         


# ----------------------------------------------------------------------------
# Subsampling
# ----------------------------------------------------------------------------
GET_SUBSAMPLING                 =   IS_GET_SUBSAMPLING                      
GET_SUPPORTED_SUBSAMPLING       =   IS_GET_SUPPORTED_SUBSAMPLING            
GET_SUBSAMPLING_TYPE            =   IS_GET_SUBSAMPLING_TYPE                 
GET_SUBSAMPLING_FACTOR_HORIZONTAL       =   IS_GET_SUBSAMPLING_FACTOR_HORIZONTAL    
GET_SUBSAMPLING_FACTOR_VERTICAL =   IS_GET_SUBSAMPLING_FACTOR_VERTICAL      

SUBSAMPLING_DISABLE             =   IS_SUBSAMPLING_DISABLE                  

SUBSAMPLING_2X_VERTICAL         =   IS_SUBSAMPLING_2X_VERTICAL              
SUBSAMPLING_2X_HORIZONTAL       =   IS_SUBSAMPLING_2X_HORIZONTAL            
SUBSAMPLING_4X_VERTICAL         =   IS_SUBSAMPLING_4X_VERTICAL              
SUBSAMPLING_4X_HORIZONTAL       =   IS_SUBSAMPLING_4X_HORIZONTAL            
SUBSAMPLING_3X_VERTICAL         =   IS_SUBSAMPLING_3X_VERTICAL              
SUBSAMPLING_3X_HORIZONTAL       =   IS_SUBSAMPLING_3X_HORIZONTAL            
SUBSAMPLING_5X_VERTICAL         =   IS_SUBSAMPLING_5X_VERTICAL              
SUBSAMPLING_5X_HORIZONTAL       =   IS_SUBSAMPLING_5X_HORIZONTAL            
SUBSAMPLING_6X_VERTICAL         =   IS_SUBSAMPLING_6X_VERTICAL              
SUBSAMPLING_6X_HORIZONTAL       =   IS_SUBSAMPLING_6X_HORIZONTAL            
SUBSAMPLING_8X_VERTICAL         =   IS_SUBSAMPLING_8X_VERTICAL              
SUBSAMPLING_8X_HORIZONTAL       =   IS_SUBSAMPLING_8X_HORIZONTAL            
SUBSAMPLING_16X_VERTICAL        =   IS_SUBSAMPLING_16X_VERTICAL             
SUBSAMPLING_16X_HORIZONTAL      =   IS_SUBSAMPLING_16X_HORIZONTAL           

SUBSAMPLING_COLOR               =   IS_SUBSAMPLING_COLOR                    
SUBSAMPLING_MONO                =   IS_SUBSAMPLING_MONO                     

SUBSAMPLING_MASK_VERTICAL       =   IS_SUBSAMPLING_MASK_VERTICAL            
SUBSAMPLING_MASK_HORIZONTAL     =   IS_SUBSAMPLING_MASK_HORIZONTAL          

# Compatibility
#Not in 3.90
#SUBSAMPLING_VERT                =   IS_SUBSAMPLING_VERT                     
#SUBSAMPLING_HOR                 =   IS_SUBSAMPLING_HOR                      


# ----------------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------------
GET_BINNING                     =   IS_GET_BINNING                      
GET_SUPPORTED_BINNING           =   IS_GET_SUPPORTED_BINNING            
GET_BINNING_TYPE                =   IS_GET_BINNING_TYPE                 
GET_BINNING_FACTOR_HORIZONTAL   =   IS_GET_BINNING_FACTOR_HORIZONTAL    
GET_BINNING_FACTOR_VERTICAL     =   IS_GET_BINNING_FACTOR_VERTICAL      

BINNING_DISABLE                 =   IS_BINNING_DISABLE                  

BINNING_2X_VERTICAL             =   IS_BINNING_2X_VERTICAL              
BINNING_2X_HORIZONTAL           =   IS_BINNING_2X_HORIZONTAL            
BINNING_4X_VERTICAL             =   IS_BINNING_4X_VERTICAL              
BINNING_4X_HORIZONTAL           =   IS_BINNING_4X_HORIZONTAL            
BINNING_3X_VERTICAL             =   IS_BINNING_3X_VERTICAL              
BINNING_3X_HORIZONTAL           =   IS_BINNING_3X_HORIZONTAL            
BINNING_5X_VERTICAL             =   IS_BINNING_5X_VERTICAL              
BINNING_5X_HORIZONTAL           =   IS_BINNING_5X_HORIZONTAL            
BINNING_6X_VERTICAL             =   IS_BINNING_6X_VERTICAL              
BINNING_6X_HORIZONTAL           =   IS_BINNING_6X_HORIZONTAL            
BINNING_8X_VERTICAL             =   IS_BINNING_8X_VERTICAL              
BINNING_8X_HORIZONTAL           =   IS_BINNING_8X_HORIZONTAL            
BINNING_16X_VERTICAL            =   IS_BINNING_16X_VERTICAL             
BINNING_16X_HORIZONTAL          =   IS_BINNING_16X_HORIZONTAL           

BINNING_COLOR                   =   IS_BINNING_COLOR                    
BINNING_MONO                    =   IS_BINNING_MONO                     

BINNING_MASK_VERTICAL           =   IS_BINNING_MASK_VERTICAL            
BINNING_MASK_HORIZONTAL         =   IS_BINNING_MASK_HORIZONTAL          

# Compatibility
#Not in 3.90
#BINNING_VERT                    =   IS_BINNING_VERT                     
#BINNING_HOR                     =   IS_BINNING_HOR                      

# ----------------------------------------------------------------------------
# Auto Control Parameter
# ----------------------------------------------------------------------------
SET_ENABLE_AUTO_GAIN            =   IS_SET_ENABLE_AUTO_GAIN             
GET_ENABLE_AUTO_GAIN            =   IS_GET_ENABLE_AUTO_GAIN             
SET_ENABLE_AUTO_SHUTTER         =   IS_SET_ENABLE_AUTO_SHUTTER          
GET_ENABLE_AUTO_SHUTTER         =   IS_GET_ENABLE_AUTO_SHUTTER          
SET_ENABLE_AUTO_WHITEBALANCE    =   IS_SET_ENABLE_AUTO_WHITEBALANCE     
GET_ENABLE_AUTO_WHITEBALANCE    =   IS_GET_ENABLE_AUTO_WHITEBALANCE     
SET_ENABLE_AUTO_FRAMERATE       =   IS_SET_ENABLE_AUTO_FRAMERATE        
GET_ENABLE_AUTO_FRAMERATE       =   IS_GET_ENABLE_AUTO_FRAMERATE        
SET_ENABLE_AUTO_SENSOR_GAIN     =   IS_SET_ENABLE_AUTO_SENSOR_GAIN      
GET_ENABLE_AUTO_SENSOR_GAIN     =   IS_GET_ENABLE_AUTO_SENSOR_GAIN      
SET_ENABLE_AUTO_SENSOR_SHUTTER  =   IS_SET_ENABLE_AUTO_SENSOR_SHUTTER   
GET_ENABLE_AUTO_SENSOR_SHUTTER  =   IS_GET_ENABLE_AUTO_SENSOR_SHUTTER   
SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER     =   IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER  
GET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER     =   IS_GET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER  
SET_ENABLE_AUTO_SENSOR_FRAMERATE=   IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE     
GET_ENABLE_AUTO_SENSOR_FRAMERATE=   IS_GET_ENABLE_AUTO_SENSOR_FRAMERATE     

SET_AUTO_REFERENCE              =   IS_SET_AUTO_REFERENCE               
GET_AUTO_REFERENCE              =   IS_GET_AUTO_REFERENCE               
SET_AUTO_GAIN_MAX               =   IS_SET_AUTO_GAIN_MAX                
GET_AUTO_GAIN_MAX               =   IS_GET_AUTO_GAIN_MAX                
SET_AUTO_SHUTTER_MAX            =   IS_SET_AUTO_SHUTTER_MAX             
GET_AUTO_SHUTTER_MAX            =   IS_GET_AUTO_SHUTTER_MAX             
SET_AUTO_SPEED                  =   IS_SET_AUTO_SPEED                   
GET_AUTO_SPEED                  =   IS_GET_AUTO_SPEED                   
SET_AUTO_WB_OFFSET              =   IS_SET_AUTO_WB_OFFSET               
GET_AUTO_WB_OFFSET              =   IS_GET_AUTO_WB_OFFSET               
SET_AUTO_WB_GAIN_RANGE          =   IS_SET_AUTO_WB_GAIN_RANGE           
GET_AUTO_WB_GAIN_RANGE          =   IS_GET_AUTO_WB_GAIN_RANGE           
SET_AUTO_WB_SPEED               =   IS_SET_AUTO_WB_SPEED                
GET_AUTO_WB_SPEED               =   IS_GET_AUTO_WB_SPEED                
SET_AUTO_WB_ONCE                =   IS_SET_AUTO_WB_ONCE                 
GET_AUTO_WB_ONCE                =   IS_GET_AUTO_WB_ONCE                 
SET_AUTO_BRIGHTNESS_ONCE        =   IS_SET_AUTO_BRIGHTNESS_ONCE         
GET_AUTO_BRIGHTNESS_ONCE        =   IS_GET_AUTO_BRIGHTNESS_ONCE         
SET_AUTO_HYSTERESIS             =   IS_SET_AUTO_HYSTERESIS              
GET_AUTO_HYSTERESIS             =   IS_GET_AUTO_HYSTERESIS              
GET_AUTO_HYSTERESIS_RANGE       =   IS_GET_AUTO_HYSTERESIS_RANGE        
SET_AUTO_WB_HYSTERESIS          =   IS_SET_AUTO_WB_HYSTERESIS           
GET_AUTO_WB_HYSTERESIS          =   IS_GET_AUTO_WB_HYSTERESIS           
GET_AUTO_WB_HYSTERESIS_RANGE    =   IS_GET_AUTO_WB_HYSTERESIS_RANGE     
SET_AUTO_SKIPFRAMES             =   IS_SET_AUTO_SKIPFRAMES              
GET_AUTO_SKIPFRAMES             =   IS_GET_AUTO_SKIPFRAMES              
GET_AUTO_SKIPFRAMES_RANGE       =   IS_GET_AUTO_SKIPFRAMES_RANGE        
SET_AUTO_WB_SKIPFRAMES          =   IS_SET_AUTO_WB_SKIPFRAMES           
GET_AUTO_WB_SKIPFRAMES          =   IS_GET_AUTO_WB_SKIPFRAMES           
GET_AUTO_WB_SKIPFRAMES_RANGE    =   IS_GET_AUTO_WB_SKIPFRAMES_RANGE     

# ----------------------------------------------------------------------------
# Auto Control definitions
# ----------------------------------------------------------------------------
MIN_AUTO_BRIGHT_REFERENCE       =   IS_MIN_AUTO_BRIGHT_REFERENCE        
MAX_AUTO_BRIGHT_REFERENCE       =   IS_MAX_AUTO_BRIGHT_REFERENCE        
DEFAULT_AUTO_BRIGHT_REFERENCE   =   IS_DEFAULT_AUTO_BRIGHT_REFERENCE    
MIN_AUTO_SPEED                  =   IS_MIN_AUTO_SPEED                   
MAX_AUTO_SPEED                  =   IS_MAX_AUTO_SPEED                   
DEFAULT_AUTO_SPEED              =   IS_DEFAULT_AUTO_SPEED               

DEFAULT_AUTO_WB_OFFSET          =   IS_DEFAULT_AUTO_WB_OFFSET           
MIN_AUTO_WB_OFFSET              =   IS_MIN_AUTO_WB_OFFSET               
MAX_AUTO_WB_OFFSET              =   IS_MAX_AUTO_WB_OFFSET               
DEFAULT_AUTO_WB_SPEED           =   IS_DEFAULT_AUTO_WB_SPEED            
MIN_AUTO_WB_SPEED               =   IS_MIN_AUTO_WB_SPEED                
MAX_AUTO_WB_SPEED               =   IS_MAX_AUTO_WB_SPEED                
MIN_AUTO_WB_REFERENCE           =   IS_MIN_AUTO_WB_REFERENCE            
MAX_AUTO_WB_REFERENCE           =   IS_MAX_AUTO_WB_REFERENCE            


# ----------------------------------------------------------------------------
# AOI types to set/get
# ----------------------------------------------------------------------------
SET_AUTO_BRIGHT_AOI             =   IS_SET_AUTO_BRIGHT_AOI              
GET_AUTO_BRIGHT_AOI             =   IS_GET_AUTO_BRIGHT_AOI              
SET_IMAGE_AOI                   =   IS_SET_IMAGE_AOI                    
GET_IMAGE_AOI                   =   IS_GET_IMAGE_AOI                    
SET_AUTO_WB_AOI                 =   IS_SET_AUTO_WB_AOI                  
GET_AUTO_WB_AOI                 =   IS_GET_AUTO_WB_AOI                  


# ----------------------------------------------------------------------------
# color modes
# ----------------------------------------------------------------------------
GET_COLOR_MODE                  =   IS_GET_COLOR_MODE                   

SET_CM_RGB32                    =   IS_SET_CM_RGB32                     
SET_CM_RGB24                    =   IS_SET_CM_RGB24                     
SET_CM_RGB16                    =   IS_SET_CM_RGB16                     
SET_CM_RGB15                    =   IS_SET_CM_RGB15                     
SET_CM_Y8                       =   IS_SET_CM_Y8                        
SET_CM_RGB8                     =   IS_SET_CM_RGB8                      
SET_CM_BAYER                    =   IS_SET_CM_BAYER                     
SET_CM_UYVY                     =   IS_SET_CM_UYVY                      
SET_CM_UYVY_MONO                =   IS_SET_CM_UYVY_MONO                 
SET_CM_UYVY_BAYER               =   IS_SET_CM_UYVY_BAYER                
SET_CM_CBYCRY                   =   IS_SET_CM_CBYCRY                    

SET_CM_RGBY                     =   IS_SET_CM_RGBY                      
SET_CM_RGB30                    =   IS_SET_CM_RGB30                     
SET_CM_Y12                      =   IS_SET_CM_Y12                       
SET_CM_BAYER12                  =   IS_SET_CM_BAYER12                   
SET_CM_Y16                      =   IS_SET_CM_Y16                       
SET_CM_BAYER16                  =   IS_SET_CM_BAYER16                   

CM_MODE_MASK                    =   IS_CM_MODE_MASK                     

# planar vs packed format
CM_FORMAT_PACKED                =   IS_CM_FORMAT_PACKED                 
CM_FORMAT_PLANAR                =   IS_CM_FORMAT_PLANAR                 
CM_FORMAT_MASK                  =   IS_CM_FORMAT_MASK                   

# BGR vs. RGB order
CM_ORDER_BGR                    =   IS_CM_ORDER_BGR                     
CM_ORDER_RGB                    =   IS_CM_ORDER_RGB                     
CM_ORDER_MASK                   =   IS_CM_ORDER_MASK                     


# define compliant color format names
CM_MONO8                        =   IS_CM_MONO8                                                             # occupies 8 Bit
CM_MONO12                       =   IS_CM_MONO12                                                           # occupies 16 Bit
CM_MONO16                       =   IS_CM_MONO16                                                           # occupies 16 Bit

#CM_BAYER_RG8                    =   IS_CM_BAYER_RG8                                                      # occupies 8 Bit
#CM_BAYER_RG12                   =   IS_CM_BAYER_RG12                                                   # occupies 16 Bit
#CM_BAYER_RG16                   =   IS_CM_BAYER_RG16                                                   # occupies 16 Bit

#CM_BGR555_PACKED                =   IS_CM_BGR555_PACKED          # occupies 16 Bit
CM_BGR565_PACKED                =   IS_CM_BGR565_PACKED          # occupies 16 Bit 

CM_RGB8_PACKED                  =   IS_CM_RGB8_PACKED            # occupies 24 Bit
CM_BGR8_PACKED                  =   IS_CM_BGR8_PACKED            # occupies 24 Bit  
CM_RGBA8_PACKED                 =   IS_CM_RGBA8_PACKED           # occupies 32 Bit
CM_BGRA8_PACKED                 =   IS_CM_BGRA8_PACKED           # occupies 32 Bit
CM_RGBY8_PACKED                 =   IS_CM_RGBY8_PACKED           # occupies 32 Bit
CM_BGRY8_PACKED                 =   IS_CM_BGRY8_PACKED           # occupies 32 Bit
#CM_RGB10V2_PACKED               =   IS_CM_RGB10V2_PACKED         # occupies 32 Bit
#CM_BGR10V2_PACKED               =   IS_CM_BGR10V2_PACKED         # occupies 32 Bit

# CM_YUV422_PACKED                =   IS_CM_YUV422_PACKED         #no compliant version      
CM_UYVY_PACKED                  =   IS_CM_UYVY_PACKED                               # occupies 16 Bit
CM_UYVY_MONO_PACKED             =   IS_CM_UYVY_MONO_PACKED      
CM_UYVY_BAYER_PACKED            =   IS_CM_UYVY_BAYER_PACKED     
CM_CBYCRY_PACKED                =   IS_CM_CBYCRY_PACKED                           # occupies 16 Bit

#CM_RGB8_PLANAR                  =   IS_CM_RGB8_PLANAR           
#CM_RGB12_PLANAR                 =   IS_CM_RGB12_PLANAR          
#CM_RGB16_PLANAR                 =   IS_CM_RGB16_PLANAR          


CM_ALL_POSSIBLE                 =   IS_CM_ALL_POSSIBLE                  

# ----------------------------------------------------------------------------
# Hotpixel correction
# ----------------------------------------------------------------------------
#Not in 390
#GET_BPC_MODE                    =   IS_GET_BPC_MODE                      
#GET_BPC_THRESHOLD               =   IS_GET_BPC_THRESHOLD                 
#GET_BPC_SUPPORTED_MODE          =   IS_GET_BPC_SUPPORTED_MODE            

#BPC_DISABLE                     =   IS_BPC_DISABLE                       
#BPC_ENABLE_LEVEL_1              =   IS_BPC_ENABLE_LEVEL_1                
#BPC_ENABLE_LEVEL_2              =   IS_BPC_ENABLE_LEVEL_2                
#BPC_ENABLE_USER                 =   IS_BPC_ENABLE_USER                   
#BPC_ENABLE_SOFTWARE             =   IS_BPC_ENABLE_SOFTWARE          
#BPC_ENABLE_HARDWARE             =   IS_BPC_ENABLE_HARDWARE          

#The following constants where removed in the 3.9 drivers
#SET_BADPIXEL_LIST               =   IS_SET_BADPIXEL_LIST                 
#GET_BADPIXEL_LIST               =   IS_GET_BADPIXEL_LIST                 
#GET_LIST_SIZE                   =   IS_GET_LIST_SIZE                     


# ----------------------------------------------------------------------------
# color correction definitions
# ----------------------------------------------------------------------------
GET_CCOR_MODE                   =   IS_GET_CCOR_MODE                    
GET_SUPPORTED_CCOR_MODE         =   IS_GET_SUPPORTED_CCOR_MODE          
GET_DEFAULT_CCOR_MODE           =   IS_GET_DEFAULT_CCOR_MODE            
GET_CCOR_FACTOR                 =   IS_GET_CCOR_FACTOR                  
GET_CCOR_FACTOR_MIN             =   IS_GET_CCOR_FACTOR_MIN              
GET_CCOR_FACTOR_MAX             =   IS_GET_CCOR_FACTOR_MAX              
GET_CCOR_FACTOR_DEFAULT         =   IS_GET_CCOR_FACTOR_DEFAULT          

CCOR_DISABLE                    =   IS_CCOR_DISABLE                     
CCOR_ENABLE                     =   IS_CCOR_ENABLE                      
CCOR_ENABLE_NORMAL              =   IS_CCOR_ENABLE_NORMAL           
CCOR_ENABLE_BG40_ENHANCED       =   IS_CCOR_ENABLE_BG40_ENHANCED        
CCOR_ENABLE_HQ_ENHANCED         =   IS_CCOR_ENABLE_HQ_ENHANCED          
CCOR_SET_IR_AUTOMATIC           =   IS_CCOR_SET_IR_AUTOMATIC            
CCOR_FACTOR                     =   IS_CCOR_FACTOR                      

CCOR_ENABLE_MASK                =   IS_CCOR_ENABLE_MASK             


# ----------------------------------------------------------------------------
# bayer algorithm modes
# ----------------------------------------------------------------------------
GET_BAYER_CV_MODE               =   IS_GET_BAYER_CV_MODE                

SET_BAYER_CV_NORMAL             =   IS_SET_BAYER_CV_NORMAL              
SET_BAYER_CV_BETTER             =   IS_SET_BAYER_CV_BETTER              
SET_BAYER_CV_BEST               =   IS_SET_BAYER_CV_BEST                


# ----------------------------------------------------------------------------
# color converter modes
# ----------------------------------------------------------------------------
CONV_MODE_NONE                  =   IS_CONV_MODE_NONE                   
CONV_MODE_SOFTWARE              =   IS_CONV_MODE_SOFTWARE               
CONV_MODE_SOFTWARE_3X3          =   IS_CONV_MODE_SOFTWARE_3X3           
CONV_MODE_SOFTWARE_5X5          =   IS_CONV_MODE_SOFTWARE_5X5           
CONV_MODE_HARDWARE_3X3          =   IS_CONV_MODE_HARDWARE_3X3           


# ----------------------------------------------------------------------------
# Edge enhancement
# ----------------------------------------------------------------------------
GET_EDGE_ENHANCEMENT            =   IS_GET_EDGE_ENHANCEMENT             

EDGE_EN_DISABLE                 =   IS_EDGE_EN_DISABLE                  
EDGE_EN_STRONG                  =   IS_EDGE_EN_STRONG                   
EDGE_EN_WEAK                    =   IS_EDGE_EN_WEAK                     


# ----------------------------------------------------------------------------
# white balance modes
# ----------------------------------------------------------------------------
GET_WB_MODE                     =   IS_GET_WB_MODE                      

SET_WB_DISABLE                  =   IS_SET_WB_DISABLE                   
SET_WB_USER                     =   IS_SET_WB_USER                      
SET_WB_AUTO_ENABLE              =   IS_SET_WB_AUTO_ENABLE               
SET_WB_AUTO_ENABLE_ONCE         =   IS_SET_WB_AUTO_ENABLE_ONCE          

SET_WB_DAYLIGHT_65              =   IS_SET_WB_DAYLIGHT_65               
SET_WB_COOL_WHITE               =   IS_SET_WB_COOL_WHITE                
SET_WB_U30                      =   IS_SET_WB_U30                       
SET_WB_ILLUMINANT_A             =   IS_SET_WB_ILLUMINANT_A              
SET_WB_HORIZON                  =   IS_SET_WB_HORIZON                   


# ----------------------------------------------------------------------------
# flash strobe constants
# ----------------------------------------------------------------------------
#Not in 3.90
#GET_FLASHSTROBE_MODE            =   IS_GET_FLASHSTROBE_MODE             
#GET_FLASHSTROBE_LINE            =   IS_GET_FLASHSTROBE_LINE             
#GET_SUPPORTED_FLASH_IO_PORTS    =   IS_GET_SUPPORTED_FLASH_IO_PORTS     

#SET_FLASH_OFF                   =   IS_SET_FLASH_OFF                    
#SET_FLASH_ON                    =   IS_SET_FLASH_ON                     
#SET_FLASH_LO_ACTIVE             =   IS_SET_FLASH_LO_ACTIVE          
#SET_FLASH_HI_ACTIVE             =   IS_SET_FLASH_HI_ACTIVE              
#SET_FLASH_HIGH                  =   IS_SET_FLASH_HIGH                   
#SET_FLASH_LOW                   =   IS_SET_FLASH_LOW                    
#SET_FLASH_LO_ACTIVE_FREERUN     =   IS_SET_FLASH_LO_ACTIVE_FREERUN      
#SET_FLASH_HI_ACTIVE_FREERUN     =   IS_SET_FLASH_HI_ACTIVE_FREERUN      
#SET_FLASH_IO_1                  =   IS_SET_FLASH_IO_1                   
#SET_FLASH_IO_2                  =   IS_SET_FLASH_IO_2                   
#SET_FLASH_IO_3                  =   IS_SET_FLASH_IO_3                   
#SET_FLASH_IO_4                  =   IS_SET_FLASH_IO_4                   
#FLASH_IO_PORT_MASK              =   IS_FLASH_IO_PORT_MASK             

#GET_FLASH_DELAY                 =   IS_GET_FLASH_DELAY                  
#GET_FLASH_DURATION              =   IS_GET_FLASH_DURATION               
#GET_MAX_FLASH_DELAY             =   IS_GET_MAX_FLASH_DELAY              
#GET_MAX_FLASH_DURATION          =   IS_GET_MAX_FLASH_DURATION           
#GET_MIN_FLASH_DELAY             =   IS_GET_MIN_FLASH_DELAY              
#GET_MIN_FLASH_DURATION          =   IS_GET_MIN_FLASH_DURATION           
#GET_FLASH_DELAY_GRANULARITY     =   IS_GET_FLASH_DELAY_GRANULARITY      
#GET_FLASH_DURATION_GRANULARITY  =   IS_GET_FLASH_DURATION_GRANULARITY   

# ----------------------------------------------------------------------------
# Digital IO constants
# ----------------------------------------------------------------------------
#Not in 3.90
#GET_IO                          =   IS_GET_IO                           
#GET_IO_MASK                     =   IS_GET_IO_MASK                      
#GET_INPUT_MASK                  =   IS_GET_INPUT_MASK                   
#GET_OUTPUT_MASK                 =   IS_GET_OUTPUT_MASK                  
#GET_SUPPORTED_IO_PORTS          =   IS_GET_SUPPORTED_IO_PORTS           


# ----------------------------------------------------------------------------
# EEPROM defines
# ----------------------------------------------------------------------------
EEPROM_MIN_USER_ADDRESS         =   IS_EEPROM_MIN_USER_ADDRESS          
EEPROM_MAX_USER_ADDRESS         =   IS_EEPROM_MAX_USER_ADDRESS          
EEPROM_MAX_USER_SPACE           =   IS_EEPROM_MAX_USER_SPACE            


# ----------------------------------------------------------------------------
# error report modes
# ----------------------------------------------------------------------------
GET_ERR_REP_MODE                =   IS_GET_ERR_REP_MODE                 
ENABLE_ERR_REP                  =   IS_ENABLE_ERR_REP                   
DISABLE_ERR_REP                 =   IS_DISABLE_ERR_REP                  


# ----------------------------------------------------------------------------
# display mode selectors
# ----------------------------------------------------------------------------
#Not in 3.90
GET_DISPLAY_MODE                =   IS_GET_DISPLAY_MODE                 
#GET_DISPLAY_SIZE_X              =   IS_GET_DISPLAY_SIZE_X               
#GET_DISPLAY_SIZE_Y              =   IS_GET_DISPLAY_SIZE_Y               
#GET_DISPLAY_POS_X               =   IS_GET_DISPLAY_POS_X                
#GET_DISPLAY_POS_Y               =   IS_GET_DISPLAY_POS_Y                

SET_DM_DIB                      =   IS_SET_DM_DIB                       
#SET_DM_DIRECTDRAW               =   IS_SET_DM_DIRECTDRAW                
SET_DM_DIRECT3D                 =   IS_SET_DM_DIRECT3D                  
#SET_DM_ALLOW_SYSMEM             =   IS_SET_DM_ALLOW_SYSMEM              
#SET_DM_ALLOW_PRIMARY            =   IS_SET_DM_ALLOW_PRIMARY             

# -- overlay display mode ---
#GET_DD_OVERLAY_SCALE            =   IS_GET_DD_OVERLAY_SCALE             

#SET_DM_ALLOW_OVERLAY            =   IS_SET_DM_ALLOW_OVERLAY             
#SET_DM_ALLOW_SCALING            =   IS_SET_DM_ALLOW_SCALING             
#SET_DM_ALLOW_FIELDSKIP          =   IS_SET_DM_ALLOW_FIELDSKIP           
SET_DM_MONO                     =   IS_SET_DM_MONO                      
SET_DM_BAYER                    =   IS_SET_DM_BAYER                     
SET_DM_YCBCR                    =   IS_SET_DM_YCBCR                     

# -- backbuffer display mode ---
#Not in 3.90
#SET_DM_BACKBUFFER               =   IS_SET_DM_BACKBUFFER                


# ----------------------------------------------------------------------------
# DirectRenderer commands
# ----------------------------------------------------------------------------
GET_OVERLAY_DC                  =   DR_GET_OVERLAY_DC                       
GET_MAX_OVERLAY_SIZE            =   DR_GET_MAX_OVERLAY_SIZE                 
GET_OVERLAY_KEY_COLOR           =   DR_GET_OVERLAY_KEY_COLOR                
RELEASE_OVERLAY_DC              =   DR_RELEASE_OVERLAY_DC                   
SHOW_OVERLAY                    =   DR_SHOW_OVERLAY                                  
HIDE_OVERLAY                    =   DR_HIDE_OVERLAY                                        
SET_OVERLAY_SIZE                =   DR_SET_OVERLAY_SIZE                                            
SET_OVERLAY_POSITION            =   DR_SET_OVERLAY_POSITION                     
SET_OVERLAY_KEY_COLOR           =   DR_SET_OVERLAY_KEY_COLOR                 
SET_HWND                        =   DR_SET_HWND                              
ENABLE_SCALING                  =   DR_ENABLE_SCALING                       
DISABLE_SCALING                 =   DR_DISABLE_SCALING                      
CLEAR_OVERLAY                   =   DR_CLEAR_OVERLAY                        
ENABLE_SEMI_TRANSPARENT_OVERLAY =   DR_ENABLE_SEMI_TRANSPARENT_OVERLAY      
DISABLE_SEMI_TRANSPARENT_OVERLAY=   DR_DISABLE_SEMI_TRANSPARENT_OVERLAY     
CHECK_COMPATIBILITY             =   DR_CHECK_COMPATIBILITY                  
SET_VSYNC_OFF                   =   DR_SET_VSYNC_OFF                        
SET_VSYNC_AUTO                  =   DR_SET_VSYNC_AUTO                       
SET_USER_SYNC                   =   DR_SET_USER_SYNC                        
GET_USER_SYNC_POSITION_RANGE    =   DR_GET_USER_SYNC_POSITION_RANGE         
LOAD_OVERLAY_FROM_FILE          =   DR_LOAD_OVERLAY_FROM_FILE               
STEAL_NEXT_FRAME                =   DR_STEAL_NEXT_FRAME                     
SET_STEAL_FORMAT                =   DR_SET_STEAL_FORMAT                     
GET_STEAL_FORMAT                =   DR_GET_STEAL_FORMAT                     
ENABLE_IMAGE_SCALING            =   DR_ENABLE_IMAGE_SCALING                 
GET_OVERLAY_SIZE                =   DR_GET_OVERLAY_SIZE                     

# ----------------------------------------------------------------------------
# DirectDraw keying color constants
# ----------------------------------------------------------------------------
#Not in 3.90
#GET_KC_RED                      =   IS_GET_KC_RED                       
#GET_KC_GREEN                    =   IS_GET_KC_GREEN                     
#GET_KC_BLUE                     =   IS_GET_KC_BLUE                      
#GET_KC_RGB                      =   IS_GET_KC_RGB                       
#GET_KC_INDEX                    =   IS_GET_KC_INDEX                     
#GET_KEYOFFSET_X                 =   IS_GET_KEYOFFSET_X                  
#GET_KEYOFFSET_Y                 =   IS_GET_KEYOFFSET_Y                  

# RGB-triple for default key-color in 15,16,24,32 bit mode
#SET_KC_DEFAULT                  =   IS_SET_KC_DEFAULT                      # 0xbbggrr
# color index for default key-color in 8bit palette mode
#SET_KC_DEFAULT_8                =   IS_SET_KC_DEFAULT_8                 


# ----------------------------------------------------------------------------
# Memoryboard
# ----------------------------------------------------------------------------
#MEMORY_GET_COUNT                =   IS_MEMORY_GET_COUNT                 
#MEMORY_GET_DELAY                =   IS_MEMORY_GET_DELAY                 
#MEMORY_MODE_DISABLE             =   IS_MEMORY_MODE_DISABLE              
#MEMORY_USE_TRIGGER              =   IS_MEMORY_USE_TRIGGER               


# ----------------------------------------------------------------------------
# Test image modes
# ----------------------------------------------------------------------------
#GET_TEST_IMAGE                  =   IS_GET_TEST_IMAGE                   

#SET_TEST_IMAGE_DISABLED         =   IS_SET_TEST_IMAGE_DISABLED          
#SET_TEST_IMAGE_MEMORY_1         =   IS_SET_TEST_IMAGE_MEMORY_1          
#SET_TEST_IMAGE_MEMORY_2         =   IS_SET_TEST_IMAGE_MEMORY_2          
#SET_TEST_IMAGE_MEMORY_3         =   IS_SET_TEST_IMAGE_MEMORY_3          


# ----------------------------------------------------------------------------
# Led settings
# ----------------------------------------------------------------------------
#SET_LED_OFF                     =   IS_SET_LED_OFF                      
#SET_LED_ON                      =   IS_SET_LED_ON                       
#SET_LED_TOGGLE                  =   IS_SET_LED_TOGGLE                   
#GET_LED                         =   IS_GET_LED                          


# ----------------------------------------------------------------------------
# save options
# ----------------------------------------------------------------------------
SAVE_USE_ACTUAL_IMAGE_SIZE      =   IS_SAVE_USE_ACTUAL_IMAGE_SIZE       

# ----------------------------------------------------------------------------
# renumeration modes
# ----------------------------------------------------------------------------
RENUM_BY_CAMERA                 =   IS_RENUM_BY_CAMERA                  
RENUM_BY_HOST                   =   IS_RENUM_BY_HOST                    

# ----------------------------------------------------------------------------
# event constants
# ----------------------------------------------------------------------------
SET_EVENT_ODD                   =   IS_SET_EVENT_ODD                    
SET_EVENT_EVEN                  =   IS_SET_EVENT_EVEN                   
SET_EVENT_FRAME                 =   IS_SET_EVENT_FRAME                  
SET_EVENT_EXTTRIG               =   IS_SET_EVENT_EXTTRIG                
SET_EVENT_VSYNC                 =   IS_SET_EVENT_VSYNC                  
SET_EVENT_SEQ                   =   IS_SET_EVENT_SEQ                    
SET_EVENT_STEAL                 =   IS_SET_EVENT_STEAL                  
SET_EVENT_VPRES                 =   IS_SET_EVENT_VPRES                  
#SET_EVENT_TRANSFER_FAILED       =   IS_SET_EVENT_TRANSFER_FAILED        
SET_EVENT_DEVICE_RECONNECTED    =   IS_SET_EVENT_DEVICE_RECONNECTED     
SET_EVENT_MEMORY_MODE_FINISH    =   IS_SET_EVENT_MEMORY_MODE_FINISH     
SET_EVENT_FRAME_RECEIVED        =   IS_SET_EVENT_FRAME_RECEIVED         
SET_EVENT_WB_FINISHED           =   IS_SET_EVENT_WB_FINISHED            
SET_EVENT_AUTOBRIGHTNESS_FINISHED       =   IS_SET_EVENT_AUTOBRIGHTNESS_FINISHED 

SET_EVENT_REMOVE                =   IS_SET_EVENT_REMOVE                 
SET_EVENT_REMOVAL               =   IS_SET_EVENT_REMOVAL                
SET_EVENT_NEW_DEVICE            =   IS_SET_EVENT_NEW_DEVICE             
SET_EVENT_STATUS_CHANGED        =   IS_SET_EVENT_STATUS_CHANGED         


# ----------------------------------------------------------------------------
# Window message defines
# ----------------------------------------------------------------------------
UEYE_MESSAGE                    =   IS_UEYE_MESSAGE                      
FRAME                           =   IS_FRAME                          
SEQUENCE                        =   IS_SEQUENCE                       
TRIGGER                         =   IS_TRIGGER                        
#TRANSFER_FAILED                 =   IS_TRANSFER_FAILED                
DEVICE_RECONNECTED              =   IS_DEVICE_RECONNECTED             
MEMORY_MODE_FINISH              =   IS_MEMORY_MODE_FINISH             
FRAME_RECEIVED                  =   IS_FRAME_RECEIVED                 
GENERIC_ERROR                   =   IS_GENERIC_ERROR                  
STEAL_VIDEO                     =   IS_STEAL_VIDEO                    
WB_FINISHED                     =   IS_WB_FINISHED                    
AUTOBRIGHTNESS_FINISHED         =   IS_AUTOBRIGHTNESS_FINISHED        

DEVICE_REMOVED                  =   IS_DEVICE_REMOVED                 
DEVICE_REMOVAL                  =   IS_DEVICE_REMOVAL                 
NEW_DEVICE                      =   IS_NEW_DEVICE                     
DEVICE_STATUS_CHANGED           =   IS_DEVICE_STATUS_CHANGED          


# ----------------------------------------------------------------------------
# camera id constants
# ----------------------------------------------------------------------------
GET_CAMERA_ID                   =   IS_GET_CAMERA_ID                    


# ----------------------------------------------------------------------------
# camera info constants
# ----------------------------------------------------------------------------
GET_STATUS                      =   IS_GET_STATUS                       

EXT_TRIGGER_EVENT_CNT           =   IS_EXT_TRIGGER_EVENT_CNT            
FIFO_OVR_CNT                    =   IS_FIFO_OVR_CNT                     
SEQUENCE_CNT                    =   IS_SEQUENCE_CNT                     
LAST_FRAME_FIFO_OVR             =   IS_LAST_FRAME_FIFO_OVR              
SEQUENCE_SIZE                   =   IS_SEQUENCE_SIZE                    
VIDEO_PRESENT                   =   IS_VIDEO_PRESENT                    
STEAL_FINISHED                  =   IS_STEAL_FINISHED                   
STORE_FILE_PATH                 =   IS_STORE_FILE_PATH                  
LUMA_BANDWIDTH_FILTER           =   IS_LUMA_BANDWIDTH_FILTER            
BOARD_REVISION                  =   IS_BOARD_REVISION                   
MIRROR_BITMAP_UPDOWN            =   IS_MIRROR_BITMAP_UPDOWN             
BUS_OVR_CNT                     =   IS_BUS_OVR_CNT                      
STEAL_ERROR_CNT                 =   IS_STEAL_ERROR_CNT                  
LOW_COLOR_REMOVAL               =   IS_LOW_COLOR_REMOVAL                
CHROMA_COMB_FILTER              =   IS_CHROMA_COMB_FILTER               
CHROMA_AGC                      =   IS_CHROMA_AGC                       
WATCHDOG_ON_BOARD               =   IS_WATCHDOG_ON_BOARD                
PASSTHROUGH_ON_BOARD            =   IS_PASSTHROUGH_ON_BOARD             
EXTERNAL_VREF_MODE              =   IS_EXTERNAL_VREF_MODE               
WAIT_TIMEOUT                    =   IS_WAIT_TIMEOUT                     
TRIGGER_MISSED                  =   IS_TRIGGER_MISSED                   
LAST_CAPTURE_ERROR              =   IS_LAST_CAPTURE_ERROR               
PARAMETER_SET_1                 =   IS_PARAMETER_SET_1                  
PARAMETER_SET_2                 =   IS_PARAMETER_SET_2                  
STANDBY                         =   IS_STANDBY                          
STANDBY_SUPPORTED               =   IS_STANDBY_SUPPORTED                
QUEUED_IMAGE_EVENT_CNT          =   IS_QUEUED_IMAGE_EVENT_CNT           

# ----------------------------------------------------------------------------
# interface type defines
# ----------------------------------------------------------------------------
INTERFACE_TYPE_USB              =   IS_INTERFACE_TYPE_USB               
INTERFACE_TYPE_ETH              =   IS_INTERFACE_TYPE_ETH               

# ----------------------------------------------------------------------------
# board type defines
# ----------------------------------------------------------------------------
#BOARD_TYPE_FALCON               =   IS_BOARD_TYPE_FALCON                
#BOARD_TYPE_EAGLE                =   IS_BOARD_TYPE_EAGLE                 
#BOARD_TYPE_FALCON2              =   IS_BOARD_TYPE_FALCON2               
#BOARD_TYPE_FALCON_PLUS          =   IS_BOARD_TYPE_FALCON_PLUS           
#BOARD_TYPE_FALCON_QUATTRO       =   IS_BOARD_TYPE_FALCON_QUATTRO        
#BOARD_TYPE_FALCON_DUO           =   IS_BOARD_TYPE_FALCON_DUO            
#BOARD_TYPE_EAGLE_QUATTRO        =   IS_BOARD_TYPE_EAGLE_QUATTRO         
#BOARD_TYPE_EAGLE_DUO            =   IS_BOARD_TYPE_EAGLE_DUO             
BOARD_TYPE_UEYE_USB             =   IS_BOARD_TYPE_UEYE_USB                   # 0x40
BOARD_TYPE_UEYE_USB_SE          =   IS_BOARD_TYPE_UEYE_USB_SE                     # 0x40
BOARD_TYPE_UEYE_USB_RE          =   IS_BOARD_TYPE_UEYE_USB_RE                     # 0x40
BOARD_TYPE_UEYE_USB_ME          =   IS_BOARD_TYPE_UEYE_USB_ME             # 0x41
BOARD_TYPE_UEYE_USB_LE          =   IS_BOARD_TYPE_UEYE_USB_LE             # 0x42
BOARD_TYPE_UEYE_ETH             =   IS_BOARD_TYPE_UEYE_ETH                         # 0x80
BOARD_TYPE_UEYE_ETH_HE          =   IS_BOARD_TYPE_UEYE_ETH_HE                     # 0x80
BOARD_TYPE_UEYE_ETH_SE          =   IS_BOARD_TYPE_UEYE_ETH_SE             # 0x81
BOARD_TYPE_UEYE_ETH_RE          =   IS_BOARD_TYPE_UEYE_ETH_RE                  # 0x81

# ----------------------------------------------------------------------------
# camera type defines
# ----------------------------------------------------------------------------
CAMERA_TYPE_UEYE_USB            =   IS_CAMERA_TYPE_UEYE_USB         
CAMERA_TYPE_UEYE_USB_SE         =   IS_CAMERA_TYPE_UEYE_USB_SE      
CAMERA_TYPE_UEYE_USB_RE         =   IS_CAMERA_TYPE_UEYE_USB_RE      
CAMERA_TYPE_UEYE_USB_ME         =   IS_CAMERA_TYPE_UEYE_USB_ME      
CAMERA_TYPE_UEYE_USB_LE         =   IS_CAMERA_TYPE_UEYE_USB_LE      
CAMERA_TYPE_UEYE_ETH            =   IS_CAMERA_TYPE_UEYE_ETH         
CAMERA_TYPE_UEYE_ETH_HE         =   IS_CAMERA_TYPE_UEYE_ETH_HE      
CAMERA_TYPE_UEYE_ETH_SE         =   IS_CAMERA_TYPE_UEYE_ETH_SE      
CAMERA_TYPE_UEYE_ETH_RE         =   IS_CAMERA_TYPE_UEYE_ETH_RE      

# ----------------------------------------------------------------------------
# readable operation system defines
# ----------------------------------------------------------------------------
OS_UNDETERMINED                 =   IS_OS_UNDETERMINED                  
OS_WIN95                        =   IS_OS_WIN95                         
OS_WINNT40                      =   IS_OS_WINNT40                       
OS_WIN98                        =   IS_OS_WIN98                         
OS_WIN2000                      =   IS_OS_WIN2000                       
OS_WINXP                        =   IS_OS_WINXP                         
OS_WINME                        =   IS_OS_WINME                         
OS_WINNET                       =   IS_OS_WINNET                        
OS_WINSERVER2003                =   IS_OS_WINSERVER2003                 
OS_WINVISTA                     =   IS_OS_WINVISTA                      
OS_LINUX24                      =   IS_OS_LINUX24                       
OS_LINUX26                      =   IS_OS_LINUX26                       
OS_WIN7                         =   IS_OS_WIN7                          


# ----------------------------------------------------------------------------
# Bus speed
# ----------------------------------------------------------------------------
USB_10                          =   IS_USB_10                            #  1,5 Mb/s
USB_11                          =   IS_USB_11                            #   12 Mb/s
USB_20                          =   IS_USB_20                            #  480 Mb/s
USB_30                          =   IS_USB_30                            # 5000 Mb/s
ETHERNET_10                     =   IS_ETHERNET_10                       #   10 Mb/s
ETHERNET_100                    =   IS_ETHERNET_100                      #  100 Mb/s
ETHERNET_1000                   =   IS_ETHERNET_1000                     # 1000 Mb/s
ETHERNET_10000                  =   IS_ETHERNET_10000                    #10000 Mb/s

USB_LOW_SPEED                   =   IS_USB_LOW_SPEED                    
USB_FULL_SPEED                  =   IS_USB_FULL_SPEED                   
USB_HIGH_SPEED                  =   IS_USB_HIGH_SPEED                   
USB_SUPER_SPEED                 =   IS_USB_SUPER_SPEED                  
ETHERNET_10Base                 =   IS_ETHERNET_10Base                  
ETHERNET_100Base                =   IS_ETHERNET_100Base                 
ETHERNET_1000Base               =   IS_ETHERNET_1000Base                
ETHERNET_10GBase                =   IS_ETHERNET_10GBase                 

# ----------------------------------------------------------------------------
# HDR
# ----------------------------------------------------------------------------
HDR_NOT_SUPPORTED               =   IS_HDR_NOT_SUPPORTED                
HDR_KNEEPOINTS                  =   IS_HDR_KNEEPOINTS                   
DISABLE_HDR                     =   IS_DISABLE_HDR                      
ENABLE_HDR                      =   IS_ENABLE_HDR                       


# ----------------------------------------------------------------------------
# Test images
# ----------------------------------------------------------------------------
TEST_IMAGE_NONE                 =   IS_TEST_IMAGE_NONE                          
TEST_IMAGE_WHITE                =   IS_TEST_IMAGE_WHITE                         
TEST_IMAGE_BLACK                =   IS_TEST_IMAGE_BLACK                         
TEST_IMAGE_HORIZONTAL_GREYSCALE =   IS_TEST_IMAGE_HORIZONTAL_GREYSCALE          
TEST_IMAGE_VERTICAL_GREYSCALE   =   IS_TEST_IMAGE_VERTICAL_GREYSCALE            
TEST_IMAGE_DIAGONAL_GREYSCALE   =   IS_TEST_IMAGE_DIAGONAL_GREYSCALE            
TEST_IMAGE_WEDGE_GRAY           =   IS_TEST_IMAGE_WEDGE_GRAY                    
TEST_IMAGE_WEDGE_COLOR          =   IS_TEST_IMAGE_WEDGE_COLOR                   
TEST_IMAGE_ANIMATED_WEDGE_GRAY  =   IS_TEST_IMAGE_ANIMATED_WEDGE_GRAY           

TEST_IMAGE_ANIMATED_WEDGE_COLOR =   IS_TEST_IMAGE_ANIMATED_WEDGE_COLOR          
TEST_IMAGE_MONO_BARS            =   IS_TEST_IMAGE_MONO_BARS                     
TEST_IMAGE_COLOR_BARS1          =   IS_TEST_IMAGE_COLOR_BARS1                   
TEST_IMAGE_COLOR_BARS2          =   IS_TEST_IMAGE_COLOR_BARS2                   
TEST_IMAGE_GREYSCALE1           =   IS_TEST_IMAGE_GREYSCALE1                    
TEST_IMAGE_GREY_AND_COLOR_BARS  =   IS_TEST_IMAGE_GREY_AND_COLOR_BARS           
TEST_IMAGE_MOVING_GREY_AND_COLOR_BARS   =   IS_TEST_IMAGE_MOVING_GREY_AND_COLOR_BARS    
TEST_IMAGE_ANIMATED_LINE        =   IS_TEST_IMAGE_ANIMATED_LINE                 

TEST_IMAGE_ALTERNATE_PATTERN    =   IS_TEST_IMAGE_ALTERNATE_PATTERN             
TEST_IMAGE_VARIABLE_GREY        =   IS_TEST_IMAGE_VARIABLE_GREY                 
TEST_IMAGE_MONOCHROME_HORIZONTAL_BARS   =   IS_TEST_IMAGE_MONOCHROME_HORIZONTAL_BARS    
TEST_IMAGE_MONOCHROME_VERTICAL_BARS     =   IS_TEST_IMAGE_MONOCHROME_VERTICAL_BARS      
TEST_IMAGE_CURSOR_H             =   IS_TEST_IMAGE_CURSOR_H                      
TEST_IMAGE_CURSOR_V             =   IS_TEST_IMAGE_CURSOR_V                      
TEST_IMAGE_COLDPIXEL_GRID       =   IS_TEST_IMAGE_COLDPIXEL_GRID                
TEST_IMAGE_HOTPIXEL_GRID        =   IS_TEST_IMAGE_HOTPIXEL_GRID                 

TEST_IMAGE_VARIABLE_RED_PART    =   IS_TEST_IMAGE_VARIABLE_RED_PART             
TEST_IMAGE_VARIABLE_GREEN_PART  =   IS_TEST_IMAGE_VARIABLE_GREEN_PART           
TEST_IMAGE_VARIABLE_BLUE_PART   =   IS_TEST_IMAGE_VARIABLE_BLUE_PART            
TEST_IMAGE_SHADING_IMAGE        =   IS_TEST_IMAGE_SHADING_IMAGE                 
#                                                  0x10000000
#                                                  0x20000000
#                                                  0x40000000
#                                                  0x80000000


# ----------------------------------------------------------------------------
# Sensor scaler
# ----------------------------------------------------------------------------
ENABLE_SENSOR_SCALER            =   IS_ENABLE_SENSOR_SCALER             
ENABLE_ANTI_ALIASING            =   IS_ENABLE_ANTI_ALIASING             


# ----------------------------------------------------------------------------
# Timeouts
# ----------------------------------------------------------------------------
TRIGGER_TIMEOUT                 =   IS_TRIGGER_TIMEOUT                  


# ----------------------------------------------------------------------------
# Auto pixel clock modes
# ----------------------------------------------------------------------------
BEST_PCLK_RUN_ONCE              =   IS_BEST_PCLK_RUN_ONCE               

# ----------------------------------------------------------------------------
# sequence flags
# ----------------------------------------------------------------------------
LOCK_LAST_BUFFER                =   IS_LOCK_LAST_BUFFER                 

# ----------------------------------------------------------------------------
# Image files types
# ----------------------------------------------------------------------------
IMG_BMP                         =   IS_IMG_BMP                          
IMG_JPG                         =   IS_IMG_JPG                          
IMG_PNG                         =   IS_IMG_PNG                          
IMG_RAW                         =   IS_IMG_RAW                          
IMG_TIF                         =   IS_IMG_TIF                          

# ----------------------------------------------------------------------------
# I2C defines
# nRegisterAddr | IS_I2C_16_BIT_REGISTER
# ----------------------------------------------------------------------------
I2C_16_BIT_REGISTER             =   IS_I2C_16_BIT_REGISTER              

# ----------------------------------------------------------------------------
# DirectDraw steal video constants   (Falcon)
# ----------------------------------------------------------------------------
#INIT_STEAL_VIDEO                =   IS_INIT_STEAL_VIDEO                 
#EXIT_STEAL_VIDEO                =   IS_EXIT_STEAL_VIDEO                 
#INIT_STEAL_VIDEO_MANUAL         =   IS_INIT_STEAL_VIDEO_MANUAL          
#INIT_STEAL_VIDEO_AUTO           =   IS_INIT_STEAL_VIDEO_AUTO            
#SET_STEAL_RATIO                 =   IS_SET_STEAL_RATIO                  
#USE_MEM_IMAGE_SIZE              =   IS_USE_MEM_IMAGE_SIZE               
#STEAL_MODES_MASK                =   IS_STEAL_MODES_MASK                 
#SET_STEAL_COPY                  =   IS_SET_STEAL_COPY                   
#SET_STEAL_NORMAL                =   IS_SET_STEAL_NORMAL                 

# ----------------------------------------------------------------------------
# AGC modes   (Falcon)
# ----------------------------------------------------------------------------
#GET_AGC_MODE                    =   IS_GET_AGC_MODE                     
#SET_AGC_OFF                     =   IS_SET_AGC_OFF                      
#SET_AGC_ON                      =   IS_SET_AGC_ON                       


# ----------------------------------------------------------------------------
# Gamma modes   (Falcon)
# ----------------------------------------------------------------------------
GET_GAMMA_MODE                  =   IS_GET_GAMMA_MODE                   
SET_GAMMA_OFF                   =   IS_SET_GAMMA_OFF                    
SET_GAMMA_ON                    =   IS_SET_GAMMA_ON                     


# ----------------------------------------------------------------------------
# sync levels   (Falcon)
# ----------------------------------------------------------------------------
#GET_SYNC_LEVEL                  =   IS_GET_SYNC_LEVEL                   
#SET_SYNC_75                     =   IS_SET_SYNC_75                      
#SET_SYNC_125                    =   IS_SET_SYNC_125                     


# ----------------------------------------------------------------------------
# color bar modes   (Falcon)
# ----------------------------------------------------------------------------
#GET_CBARS_MODE                  =   IS_GET_CBARS_MODE                   
#SET_CBARS_OFF                   =   IS_SET_CBARS_OFF                    
#SET_CBARS_ON                    =   IS_SET_CBARS_ON                     


# ----------------------------------------------------------------------------
# horizontal filter defines   (Falcon)
# ----------------------------------------------------------------------------
#GET_HOR_FILTER_MODE             =   IS_GET_HOR_FILTER_MODE              
#GET_HOR_FILTER_STEP             =   IS_GET_HOR_FILTER_STEP              

#DISABLE_HOR_FILTER              =   IS_DISABLE_HOR_FILTER               
#ENABLE_HOR_FILTER               =   IS_ENABLE_HOR_FILTER                
        #int IS_HOR_FILTER_STEP(_s_)         ((_s_ + 1) << 1)
#HOR_FILTER_STEP1                =   IS_HOR_FILTER_STEP1                 
#HOR_FILTER_STEP2                =   IS_HOR_FILTER_STEP2                 
#HOR_FILTER_STEP3                =   IS_HOR_FILTER_STEP3                 


# ----------------------------------------------------------------------------
# vertical filter defines   (Falcon)
# ----------------------------------------------------------------------------
#GET_VERT_FILTER_MODE            =   IS_GET_VERT_FILTER_MODE             
#GET_VERT_FILTER_STEP            =   IS_GET_VERT_FILTER_STEP             

#DISABLE_VERT_FILTER             =   IS_DISABLE_VERT_FILTER              
#ENABLE_VERT_FILTER              =   IS_ENABLE_VERT_FILTER               
        #int IS_VERT_FILTER_STEP(_s_)        ((_s_ + 1) << 1)
#VERT_FILTER_STEP1               =   IS_VERT_FILTER_STEP1                
#VERT_FILTER_STEP2               =   IS_VERT_FILTER_STEP2                
#VERT_FILTER_STEP3               =   IS_VERT_FILTER_STEP3                


# ----------------------------------------------------------------------------
# scaler modes   (Falcon)
# ----------------------------------------------------------------------------
#GET_SCALER_MODE                 =   IS_GET_SCALER_MODE          
#SET_SCALER_OFF                  =   IS_SET_SCALER_OFF           
#SET_SCALER_ON                   =   IS_SET_SCALER_ON            

#MIN_SCALE_X                     =   IS_MIN_SCALE_X              
#MAX_SCALE_X                     =   IS_MAX_SCALE_X              
#MIN_SCALE_Y                     =   IS_MIN_SCALE_Y              
#MAX_SCALE_Y                     =   IS_MAX_SCALE_Y              


# ----------------------------------------------------------------------------
# video source selectors   (Falcon)
# ----------------------------------------------------------------------------
#GET_VIDEO_IN                    =   IS_GET_VIDEO_IN                     
#GET_VIDEO_PASSTHROUGH           =   IS_GET_VIDEO_PASSTHROUGH            
#GET_VIDEO_IN_TOGGLE             =   IS_GET_VIDEO_IN_TOGGLE              
#GET_TOGGLE_INPUT_1              =   IS_GET_TOGGLE_INPUT_1               
#GET_TOGGLE_INPUT_2              =   IS_GET_TOGGLE_INPUT_2               
#GET_TOGGLE_INPUT_3              =   IS_GET_TOGGLE_INPUT_3               
#GET_TOGGLE_INPUT_4              =   IS_GET_TOGGLE_INPUT_4               

#SET_VIDEO_IN_1                  =   IS_SET_VIDEO_IN_1                   
#SET_VIDEO_IN_2                  =   IS_SET_VIDEO_IN_2                   
#SET_VIDEO_IN_S                  =   IS_SET_VIDEO_IN_S                   
#SET_VIDEO_IN_3                  =   IS_SET_VIDEO_IN_3                   
#SET_VIDEO_IN_4                  =   IS_SET_VIDEO_IN_4                   
#SET_VIDEO_IN_1S                 =   IS_SET_VIDEO_IN_1S                  
#SET_VIDEO_IN_2S                 =   IS_SET_VIDEO_IN_2S                  
#SET_VIDEO_IN_3S                 =   IS_SET_VIDEO_IN_3S                  
#SET_VIDEO_IN_4S                 =   IS_SET_VIDEO_IN_4S                  
#SET_VIDEO_IN_EXT                =   IS_SET_VIDEO_IN_EXT                 
#SET_TOGGLE_OFF                  =   IS_SET_TOGGLE_OFF                   
#SET_VIDEO_IN_SYNC               =   IS_SET_VIDEO_IN_SYNC                


# ----------------------------------------------------------------------------
# video crossbar selectors   (Falcon)
# ----------------------------------------------------------------------------
#GET_CROSSBAR                    =   IS_GET_CROSSBAR                     

#CROSSBAR_1                      =   IS_CROSSBAR_1                       
#CROSSBAR_2                      =   IS_CROSSBAR_2                       
#CROSSBAR_3                      =   IS_CROSSBAR_3                       
#CROSSBAR_4                      =   IS_CROSSBAR_4                       
#CROSSBAR_5                      =   IS_CROSSBAR_5                       
#CROSSBAR_6                      =   IS_CROSSBAR_6                       
#CROSSBAR_7                      =   IS_CROSSBAR_7                       
#CROSSBAR_8                      =   IS_CROSSBAR_8                       
#CROSSBAR_9                      =   IS_CROSSBAR_9                       
#CROSSBAR_10                     =   IS_CROSSBAR_10                      
#CROSSBAR_11                     =   IS_CROSSBAR_11                      
#CROSSBAR_12                     =   IS_CROSSBAR_12                      
#CROSSBAR_13                     =   IS_CROSSBAR_13                      
#CROSSBAR_14                     =   IS_CROSSBAR_14                      
#CROSSBAR_15                     =   IS_CROSSBAR_15                      
#CROSSBAR_16                     =   IS_CROSSBAR_16                      
#SELECT_AS_INPUT                 =   IS_SELECT_AS_INPUT                  


# ----------------------------------------------------------------------------
# video format selectors   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_VIDEO_MODE                  =   IS_GET_VIDEO_MODE                   
#~ 
#~ SET_VM_PAL                      =   IS_SET_VM_PAL                       
#~ SET_VM_NTSC                     =   IS_SET_VM_NTSC                      
#~ SET_VM_SECAM                    =   IS_SET_VM_SECAM                     
#~ SET_VM_AUTO                     =   IS_SET_VM_AUTO                      


# ----------------------------------------------------------------------------
# capture modes   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_CAPTURE_MODE                =   IS_GET_CAPTURE_MODE                 
#~ 
#~ SET_CM_ODD                      =   IS_SET_CM_ODD                       
#~ SET_CM_EVEN                     =   IS_SET_CM_EVEN                      
#~ SET_CM_FRAME                    =   IS_SET_CM_FRAME                     
#~ SET_CM_NONINTERLACED            =   IS_SET_CM_NONINTERLACED             
#~ SET_CM_NEXT_FRAME               =   IS_SET_CM_NEXT_FRAME                
#~ SET_CM_NEXT_FIELD               =   IS_SET_CM_NEXT_FIELD                
#~ SET_CM_BOTHFIELDS               =   IS_SET_CM_BOTHFIELDS            
#~ SET_CM_FRAME_STEREO             =   IS_SET_CM_FRAME_STEREO              


# ----------------------------------------------------------------------------
# display update mode constants   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_UPDATE_MODE                 =   IS_GET_UPDATE_MODE                  
#~ SET_UPDATE_TIMER                =   IS_SET_UPDATE_TIMER                 
#~ SET_UPDATE_EVENT                =   IS_SET_UPDATE_EVENT                 
#~ 

# ----------------------------------------------------------------------------
# sync generator mode constants   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_SYNC_GEN                    =   IS_GET_SYNC_GEN                     
#~ SET_SYNC_GEN_OFF                =   IS_SET_SYNC_GEN_OFF                 
#~ SET_SYNC_GEN_ON                 =   IS_SET_SYNC_GEN_ON                  


# ----------------------------------------------------------------------------
# decimation modes   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_DECIMATION_MODE             =   IS_GET_DECIMATION_MODE              
#~ GET_DECIMATION_NUMBER           =   IS_GET_DECIMATION_NUMBER            
#~ 
#~ DECIMATION_OFF                  =   IS_DECIMATION_OFF                   
#~ DECIMATION_CONSECUTIVE          =   IS_DECIMATION_CONSECUTIVE           
#~ DECIMATION_DISTRIBUTED          =   IS_DECIMATION_DISTRIBUTED           


# ----------------------------------------------------------------------------
# hardware watchdog defines   (Falcon)
# ----------------------------------------------------------------------------
#~ GET_WATCHDOG_TIME               =   IS_GET_WATCHDOG_TIME                
#~ GET_WATCHDOG_RESOLUTION         =   IS_GET_WATCHDOG_RESOLUTION          
#~ GET_WATCHDOG_ENABLE             =   IS_GET_WATCHDOG_ENABLE              
#~ 
#~ WATCHDOG_MINUTES                =   IS_WATCHDOG_MINUTES                 
#~ WATCHDOG_SECONDS                =   IS_WATCHDOG_SECONDS                 
#~ DISABLE_WATCHDOG                =   IS_DISABLE_WATCHDOG                 
#~ ENABLE_WATCHDOG                 =   IS_ENABLE_WATCHDOG                  
#~ RETRIGGER_WATCHDOG              =   IS_RETRIGGER_WATCHDOG               
#~ ENABLE_AUTO_DEACTIVATION        =   IS_ENABLE_AUTO_DEACTIVATION         
#~ DISABLE_AUTO_DEACTIVATION       =   IS_DISABLE_AUTO_DEACTIVATION        
#~ WATCHDOG_RESERVED               =   IS_WATCHDOG_RESERVED                

