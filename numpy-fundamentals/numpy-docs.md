---
title: NumPy Tutorials
date: 18-05-2022
---
# Linear Algebra on n-dimensional arrays
**Learning Objectives:**
* Understand the difference between one-, two- and n-dimensional arrays in NumPy;
* Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops.
* Understand axis and shape properties for n-dimensional arrays.

## Content
Use a matrix decomposition from linear algebra, the Singular Value Decomposition, to generate a compressed approximation of an image.
We'll use the `face` image from the `scipy.misc` module:
```python
from scipy import misc
img = misc.face()
```

The image is now loaded as a Numpy array: `type(img)` --> `numpy.ndarray`

## Shape, axis, and array
In numpy the dimension of a vector refers to the number of axes in an array.
If we look at the shape of our image: `img.shape` we see that it is a 3D array, with 768 entries on the first axis, 1024 on the second axis, and 3 on the thirs.
We can get the dimensions by using `img.ndim`.

We'll scale the values to lie between 0 and 1 in each entry: `img_array = img / 255`.
This operation, dividing an array by a scalar, works because o NumPy's broadcasting rules.
We can assign each color channel to a separate matrix using the slice syntax:
```python
red_array = img_array[:, :, 0]
green_array = img_array[:, :, 1]
blue_array = img_array[:, :, 2]
```

## Operations on an axis



