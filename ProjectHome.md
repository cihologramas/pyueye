pyueye is a wrapper for the uEye camera drivers written in  [Cython](http://www.cython.org),
currently under development. This wrapper allows to use the IDS uEye
cameras under python, capturing the images directly as a  Numpy array, where they can be used in image processing libraries such as OpenCV.

This wrapper is not a product from [IDS](http://www.ids-imaging.com/) nor is it in any way affiliated with or approved by them.

This project is developed by the technological development groups of [CIHologramas](http://www.cihologramas.com) and [Pratt & Miller](http://prattmiller.com).

This software is in an alpha state, and not ready for production use. Use it at your own risk.

## News ##

**October 18, 2013** - ReadEEPROM and WriteEEPROM now implemented. New version available in SVN Trunk and the [Downloads](http://code.google.com/p/pyueye/downloads/list) section.
<br>- Chris<br>
<br>
<b>August 14, 2013</b> - Now compatible with uEye driver version 4.20. This driver can be found in the <a href='http://code.google.com/p/pyueye/downloads/list'>Downloads</a> section.<br>
<b>Note:</b> The 4.20 driver seems slightly less stable than 4.02; occasionally the GigE daemon crashes. Pyueye will continue to work with both versions.<br>
<br>- Chris<br>
<br>
<b>July 2 2013</b> - New version available in Trunk. Updates include functional free-running video mode, waitEvent, and captureStatus. Also added RGB/BGR flag to <code>GrabImage</code>. Should be backwards compatible with all code, but please let us know if you find bugs.<br>
<br>- Chris<br>
<br>
<b>Jan 29 2013</b> - The version in trunk should work with the UEYE drivers for linux version 4.02. This driver can be found in the <a href='http://code.google.com/p/pyueye/downloads/list'>Downloads</a> section<br>
<br>
<b>Jul 12 2012</b> - pyueye now works with ueye driver version 3.90<br>
<br>
<br>
<h2>Installation</h2>


To compile this wrapper, you need to have installed  Cython > 0.12,<br>
Numpy and the  ueye drivers.<br>
<br>
<h3>Under Linux</h3>

To install the wrapper under linux (tested in recent Ubuntu distros) do:<br>
<br>
<pre><code>sudo python setup.py install<br>
</code></pre>

<h3>Under Windows</h3>

The wrapper should install under windows, but we haven't done it yet.<br>
If any one has successfully installed it under Windows, please let us know.<br>
<br>
<h2>Usage</h2>


This wrapper has been tested with UI-1545LE-M and UI-1645LE-C ueye USB<br>
cameras under Linux (Ubuntu 10.04), and the UI-5240CP-C GigE camera under Linux (Mint 13), but should work with any uEye camera<br>
if the driver is correctly installed. Almost all the configuration<br>
methods work as their C counterpart. A documentation of each method can<br>
be found in its python docstring.<br>
<br>
The following is a short example to capture an image and show it as<br>
a pylab image from a ipython console (ipython -pylab).<br>
<br>
<pre><code>   import ueye<br>
   a=ueye.Cam()     # Get the first available cam<br>
   b=a.GrabImage()  # Grab an image<br>
   imshow(b)<br>
</code></pre>

<h2>Comments, Bugs, Suggestions</h2>

<blockquote>ramezquitao@cihologramas.com</blockquote>


<a href='http://code.google.com/p/ciprojects/'>Return to ciprojects</a>