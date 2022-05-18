---
title: From Python to Numpy
author: Nicolas P. Rougier
date: 18-05-2022

---
**Link**: [From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
# 2 - Introduction
## 2.1 - Simple Example
Numpy is all about vectorization.
If you are familiar with Python, this is the main difficulty you'll face because you will need to change your way of thinking and your new friends are named `vectors`, `arrays`, `views` or `ufuncs`.

Below we will implement random walk:

**Object Oriented Approach**:
```python
class RandomWalker:
    def __init__(self):
        self.position = 0
    def walk(self, n):
        self.position = 0
        for i in range(n):
            yield self.position
            self.position += 2 * random.randint(0,1) - 1

walker = RandomWalker()
walk = [position for position in walker.walk(1000)]

%timeit [position for position in walker.walk(1000)]
#--> 10.8 ms ± 421 us per loop
```

**Procedural Approach**:
```python
def random_walk(n):
    position = 0
    walk = [position]
    for i in range(n):
        position += 2*random.randint(0,1)-1
        walk.append(position)
    return walk

walk = random_walk(1000)

%timeit random_walk(10000)
#--> 9.94 ms ± 41.2 us
```
This new method saves some cpu cycles but not that much because the function is pretty much the same.

**Vectorized Approach**_
Using the `itertools` Python module we can do better.
We can rewrite the function by first generating all the steps and accumulate them without any loop:
```python
def random_walk_faster(n=1000):
    from itertools import accumulate
    steps = random.choices([-1, +1], k=n)
    return [0]+list(accumulate(steps))

walk = random_walk_faster(1000)

%timeit random_walk_faster(10000)
#--> 1.55 ms ± 24.4 us
```
By vectorizing our function we got an approximate speedup of 5x.
Instead of looping, we first generated all the steps at once and used the `accumulate` function to compute all teh positions.

**Numpy Approach**:
```python
def random_walk_numpy(n=1000):
    steps = np.random.choice([-1, 1], n)
    return np.cumsum(steps)

walk = random_walk_numpy(1000)

%timeit random_walk_numpy(10000)
#--> 115 us ± 1.79 us per loop.
```
We just gained a 500x speedup using Numpy.

# 3 - Anatomy of an Array
## 3.1 - Introduction
The obvious way to do a thing in Numpy may not be the fastest.
This is due to the internal Numpy machinery and the compiler optimization.
Therefore it is important to know the this.

## 3.2 - Memory Layout
The numpy documentation defines the ndarray class:
> An instance of class ndarray consists of a contiguous one-dimensional segement of computer memory (owned by the array, or by some other object), combined with an indexing scheme that maps N integers into the location of an item in the block.

The indexing scheme is defined by a shape and a data type and this is precisely what is needed when you define a new array: `Z = np.arange(9).reshape(3,3).astype(np.int16)`

The memory layout of the numpy array is in big endian.

If we take a slice of Z, the result is a view of the base array Z: `V = Z[::2, ::2]`.
Such a view is specified using a shape, a dtype and strides because strides cannot be deduced anymore from the dtype and the shape only.

## 3.3 - Views and Copies
Veiws and copies are important concepts for the optimization of your numerical coputations.

### 3.3.1 - Direct and Indirect Access
First, we have to distinguish between indexing and fancy indexing. 
The first will always return a view while the second will return a copy.
This difference is important because in the first case, modifying the view modifies the base array while this is not true in the second case:
```python
Z = np.zeros(9)
Z_view = Z[:3]
Z_view[...] = 1
print(Z)
#--> [1., 1., 1., 0., 0., 0., 0., 0., 0.]

Z = np.zeros(9)
Z_copy = Z[[0,1,2]]
Z_copy[...] = 1
print(Z)
#--> [0., 0., 0., 0., 0., 0., 0., 0., 0.]
```
Thus, if you need fancy indexing, it is better to keep a copy of your fancy index (especially if it was complex to compute) and work with it:
```python
Z = np.zeros(9)
index = [0, 1, 2]
Z[index] = 1
print(Z)
#--> [1., 1., 1., 0., 0., 0., 0., 0., 0.]
```
Note that some numpy functions return a view when possible (e.g. `ravel`) while some others always return a copy (e.g. `flatten`).

### 3.3.2 - Temporary Copy
Copies can be made explicitly like in the previous section, but the most general case is the implicit creation of intermediate copies.
This is the case when you are doing some arithmetic with arrays:
```python
X = np.ones(10, dtype=np.int)
Y = np.ones(10, dtype=np.int)
A = 2 * X + 2 * Y
```
Here, three intermediate arrays have been created.
One for holding the result of `2 * X`, one for holding the result of `2 * Y` and the last for holding the result of `2 * X + 2 * Y`.
If your arrays are big, you have to be careful with such expressions and wonder if you can do differently.
For example, if only the final result matters and you don't need X nor Y afterwards, an alternate solution would be:
```python
X = np.ones(10, dtype=np.int)
Y = np.ones(10, dtype=np.int)
np.multiply(X, 2, out=X)
np.multiply(Y, 2, out=Y)
np.add(X, Y, out=X)
```
Here no temporary arrays were created.
Problem is that there are many other cases where such copies needs to be created and this impact the performance.




# 4 - Code Vectorization
## 4.1 - Introduction
Code vectorization means that the problem you are trying to solve is inherently vectorizable and only requires a few numpy tricks to make it faster.
Illustration with sum of two lists of integers:
```python
def add_python(Z1, Z2):
    return [z1 + z2 for (z1, z2) in zip(Z1, Z2)]

def add_numpy(Z1, Z2):
    return np.add(Z1, Z2)
```
Not only is the second approach faster, but it also functions in the way that we would expect from mathematics.

## 4.2 - Uniform Vectorization
Uniform vectorizaiton is the simplest form of vectorization where all the elements share the same computation at every time step with no specific processing for any element.

In pure Python, we can code the Game of Life using a list of lists representing the board where cells are supposed to evolve.
Such a board will be equipped with a border of 0 that allows to accelerate thigns a bit by avoiding having specific tests for borders when counting the number of neighbors.



## 4.3 - Temporal Vectorization
The python implementation of the mandelbrot set:
```python
def mandelbrot_python(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    def mandelbrot(z, maxiter):
        c = z
        for n in range(maxiter):
            if abs(z) > horizon:
                return n
            z = z*z + c
        return maxiter
    r1 = [xmin + i*(xmax - xmin) / xn for i in range(xn)]
    r2 = [ymin + i*(ymax - ymin) / yn for i in range(yn)]
    return [mandelbrot(complex(r, i), maxiter) for r in r1 for i in r2]
```
The interesting and slow part of this code is the `mandelbrot()` function that actually computes the sequence.
The vectorization of such code is not totally straightforward because the internal `return` implies a differential processing of the element.
Once it has diverged, we don't need to iterate anymore and we can safely return the iteration count at divergence.

**Numpy Implementation**:
The trick is to search at each iteration values that have not yet diverged and update relevant information for these values and only these values.
Because we start from Z = 0, we known that each value will be updated at least once and will stop being updated as soon as they've diverged.
To do that, we'll use numpy fancy indexing with the `less(x1,x2)` function that return the truth value of `(x1 < x2)` element-wise
```python
def mandelbrot_numpy(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, ymin, xn, dtype=np.float32)
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)
    C = X + Y[:, None] * 1j
    N = np.zeros(C.shape, dtype=int)
    Z = np.zeros(C.shape, np.complex64)
    for n in range(maxiter):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter - 1] = 0
    return Z, N
```
Here the numpy implementation is 6 times faster than the Python implementation.

## 4.4 - Spatial Vectorization

## 4.5 - Conclusion

# 5 - Problem Vectorization

# 6 - Custom Vectorization

# 7 - Beyond NumPy

# 8 - Conclusion

# 9 - Quick References

# 10 - Bibliography
