#!/usr/bin/env python
# coding: utf-8

# # <b><basefont  size="30px" color ="#ff0000"> Converting Images to Pencil Sketch

# <b>Importing Modules

# In[5]:


import cv2
import matplotlib.pyplot as plt


# <b>Loading and Plotting Original Image

# <b>Show Image using OpenCV
# 

# In[8]:


img = cv2.imread("thor_big.jpg")
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,8))
plt.imshow(img1)
plt.axis("off")
plt.title("Original Image")
plt.show()


# <b>Display Using Matplotlib

# In[9]:


plt.imshow(img)
plt.axis(False)
plt.show()


# <b>Matplotlib vs OpenCV</b><br>
# We can observe that the image displayed using matplotlib is not consistent with the original image.
# This is because OpenCV uses BGR color scheme whereas matplotlib uses RGB colors scheme.

# <b>Convert BGR to RGB

# In[23]:


RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)
plt.axis(False)
plt.show()


# <b>Convert to Grey Image

# In[10]:


grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey_img


# <b>Invert Image

# In[12]:


invert_img=cv2.bitwise_not(grey_img)


# <b>Blur image

# Apply Gaussian blur to the image. The second argument to the function is the kernel size, if should be a pair of odd numbers.
# Larger the kernel size, more blurred the image will be and it will lose its subtle features.
# For creating sketch, we require only the prominent features (contrasting edges) from the image.
# For small images, kernel size of (3,3), (5,5) etc. will be sufficient, whereas for larger images, small kernel size do not create any impact.
# Appropriate kernel size can be selected by trial and error method.

# In[13]:


blur_img=cv2.GaussianBlur(invert_img, (111,111),0)


# <b>Invert Blurred Image

# In[17]:


invblur_img=cv2.bitwise_not(blur_img)


# <b>Sketch

# In[19]:


sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)


# <b>Save Sketch

# In[20]:


cv2.imwrite("sketch.png", sketch_img)


# <b>Display sketch

# In[16]:


cv2.imshow("sketch image",sketch_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# <b><p style="font-size:30px">Original Image vs Sketch
# 

# In[24]:


plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
plt.title('Original image', size=18)
plt.imshow(RGB_img)
plt.axis('off')


plt.subplot(1,2,2)
plt.title('Sketch', size=18)
rgb_sketch=cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_sketch)
plt.axis('off')
plt.show()

