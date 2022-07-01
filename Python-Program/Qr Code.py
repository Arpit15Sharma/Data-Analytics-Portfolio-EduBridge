import pyqrcode 
from pyqrcode import QRCode 
  
s = "My Name is Arpit Sharma"
  
url = pyqrcode.create(s) 
  
url.svg("arpit.svg", scale = 8) 
