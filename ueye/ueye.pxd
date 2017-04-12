from ueyeh cimport *

cdef class Cam:
    cdef char ** Imgs
    cdef int BufCount
    cdef int * BufIds
    cdef char * LastSeqBuf
    cdef char * LastSeqBuf1
    cdef char LastSeqBufLocked
    cdef public HIDS cid
    cdef public INT nMaxWidth, nMaxHeight, nColorMode, colormode
    cdef public object SerNo, ID, Version, Date, Select, SensorID, strSensorName, \
    bMasterGain, bRGain, bGGain, bBGain, \
    bGlobShutter, bitspixel
    cdef public INT LineInc
    cdef public int ImgMemId
    cdef public int LiveMode
    cdef INT AOIx0, AOIy0, AOIx1, AOIy1  ##Buffers to save the AOI to speed up the image grabbing

    cdef char * GetNextBuffer(self)
    cpdef CheckNoSuccess(self,INT rv, description=?)
    cpdef GrabImage(self, BGR=?, UINT Timeout = ?, LeaveLocked = ?, char AOI = ?)
    cdef unsigned char [:,:] GrabImageGS(self, UINT Timeout=?, bint LeaveLocked=?, bint AOI = ?)