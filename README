
pyUeye is a wrapper for the uEye cameras drivers written in  Cython, 
currently under development. This wrapper allows to use the  uEye 
cameras under python, capturing the images directly as a  Numpy array.

This wrapper is not a product from IDS nor is it in any way approved as 
such.

This software is in a very alpha state, and not ready for production 
use. Use it at your own risk.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'

Installation 
============

To compile this wrapper, you need to have installed  Cython > 0.12,  
Numpy and the  ueye drivers.


Usage
=====

This wrapper has been tested with UI-1545LE-M and UI-1645LE-C ueye USB 
cameras under Linux (Ubuntu 10.04), but should work with any uEye camera
if the driver is correctly installed. Almost all the configuration 
methods work as their C counterpart. A documentation of each method can 
be found in its python docstring.

The following is a short example to capture an image and show it as 
a pylab image from a ipython console (ipython -pylab).


   import ueye
   a=ueye.Cam()     # Get the first available cam
   b=a.GrabImage()  # Grab an image
   imshow(b)
