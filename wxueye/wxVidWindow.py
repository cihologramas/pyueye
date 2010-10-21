#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wx


import threading,numpy

#~ def GetBitmap( ):
    #~ b=cam.GrabImage()  # Grab an image
    #~ height,width,c=b.shape
    #~ 
    #~ array = b.astype(numpy.uint8)
    #~ 
    #~ image = wx.EmptyImage(width,height)
    #~ image.SetData( array.tostring())
    #~ return image.ConvertToBitmap() # wx.BitmapFromImage(image)
                
class VidThread(threading.Thread):
    """Thread that executes a task every N seconds"""
    
    def __init__(self,frame):
        threading.Thread.__init__(self)
        self._finished = threading.Event()
        
        self._running= threading.Event()
        
        self._interval = 0.2
        self.frame=frame
        
    def setInterval(self, interval):
        """Set the number of seconds we sleep between executing our task"""
        self._interval = interval
    
    def preview(self, p=True):
        if p:
            self._running.set()
            
        else:
            self._running.clear()
    
    def shutdown(self):
        """Stop this thread"""
        self._finished.set()
    
    def run(self):
        while 1:
            
            self._running.wait()
            
            if self._finished.isSet(): return
            self.task()
            
            # sleep for interval or until shutdown
            #self._finished.wait(self._interval)
    
    def task(self):
        """The task done by this thread - override in subclasses"""
        self.frame.refreshlock.acquire()
        if self.frame.cam!=None:
            array=self.frame.cam.GrabImage()
            height,width,c=array.shape
            #array = b.astype(numpy.uint8)
            image = wx.EmptyImage(width,height)
            image.SetData( array.tostring())
            self.frame.vidlock.acquire()
            self.frame.image=image
            self.frame.vidlock.release()
            wx.CallAfter(self.frame.draw)
        
        

class wxVidWindow(wx.Window):
    def __init__(self, *args, **kwds):
        self.cam=kwds.get("cam",None)
        
        # begin wxGlade: wxVidCap.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Window.__init__(self, *args, **kwds)
        self.panel = wx.ScrolledWindow(self, -1, style=wx.TAB_TRAVERSAL)
        self.vidpanel = wx.Panel(self.panel, -1)

        self.__set_properties()
        self.__do_layout()
        # end wxGlade
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_SIZE,self.OnSize)
        
        
        self.vidlock=threading.Lock()
        self.refreshlock=threading.Lock()
        
        self.vidThread=VidThread(self)
        self.vidThread.start()
        self.bitmap=None
        self.image=None
        
    def __del__(self):
        self.vidThread.shutdown()
        self.vidThread.join()
        


    def __set_properties(self):
        # begin wxGlade: wxVidCap.__set_properties
        #self.SetTitle("Video Frame")
        #self.SetSize((400, 400))
        self.panel.SetScrollRate(10, 10)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: wxVidCap.__do_layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2.Add(self.vidpanel, 1, wx.EXPAND, 0)
        self.panel.SetSizer(sizer_2)
        sizer.Add(self.panel, 1, wx.EXPAND, 0)
        self.SetSizer(sizer)
        self.Layout()
        # end wxGlade
    
    def draw(self):
        if self.IsShown():
            self.vidlock.acquire()
            self.bitmap=self.image.ConvertToBitmap()
            self.vidlock.release()
            
            w,h=self.bitmap.Size
            
            h1,w1=self.vidpanel.GetMinSize()
            
            
            #TODO: Check if this can be done at the capture thread when the thread is activated
            #This is nedes so the scroll bars work
            if (w!=w1) or (h!=h1):
                self.vidpanel.SetMinSize((w,h))
                self.panel.SetVirtualSize((w,h))

            
            #self.SetMaxSize((w,h))
            ##dc=wx.MemoryDC()
            ##dc.SelectObject(self.bitmap)
            ###self.frame.panel.Draw(dc)
            ##wx.ClientDC(self.vidpanel).Blit(0,0,w,h,dc,0,0)
            dc = wx.PaintDC(self.vidpanel)
            dc.DrawBitmap(self.bitmap,0,0)
        self.refreshlock.release()  
    
    def StartLiveVideo(self):
        self.vidThread.preview(True)
        
         
    def StopLiveVideo(self):
        self.vidThread.preview(False)
    
        
    def OnPaint(self,event):
        if self.bitmap!=None:
            dc = wx.PaintDC(self.vidpanel)
            dc.DrawBitmap(self.bitmap,0,0)
    
    def OnClose(self,event):
        self.vidThread.shutdown()
        self.vidThread.join()
        self.Destroy()
        
    def OnSize(self,event):
        self.panel.SetSize(event.GetSize())
        event.Skip() 

    
