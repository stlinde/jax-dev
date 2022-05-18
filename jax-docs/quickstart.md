# JAX Official Documentation
**Link**: [JAX Documentation](https://jax.readthedocs.io/en/latest/index.html)

JAX is Autograd and XLA, brought together for high-performance numerical computing and machine learning research.
It provides composable transformations of Python+NumPy programs:
* differentiate
* vectorize
* parallelize
* jit compile to GPU/TPU

## JAX Quickstart
JAX uses XLA (Accelerated Linear Algebra) to compile and run the NumPy ocde on accelerators, like GPUs and TPUs.
JAX lets you just-in-time compile your own Python functions into XLA-optimized kernels.
To generate random numbers in JAX we need to do the following:
```python
key = random.PRNGKey(0)
x = random.nomral(key, (10,))
```
This generates a vector with 10 random numbers.

You can use `device_put()` to put NumPy ndarrays on the GPU.

JAX is much more than just a GPU-backed NumPy.
It also comes with a few program transformations that are useful when writing numerical code: 
* `jit()` for speeding up your code.
* `grad()` for taking derivatives.
* `vmap()` for automatic vectorization or batching.

### Using `jit()` to speed up functions
JAX is dispatching kernels to the GPU one operation at a time.
If we have a sequence of operations, we can use the `@jit` decorator to compile multiple operations together using `XLA`.
```python
# No jit compilation
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Two ways of jit compiling
# With function
jit_selu = jit(selu)

# With decorator
@jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
```

### Taking derivatives with `grad()`
In addition to evaluating numerical functions, we also want to transform them.
One transformation is automatic differentiation.
In JAX, just like in Autograd, you can compute gradients with the `grad()` function:
```python
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

`grad()` and `jit()` compose and can be mixed arbitrarily.
For more advanced autodifferentiation, you can use `jax.vjp()` for reverse-mode vector-Jacobian products and `jax.jvp()` for forward-mode Jacobian-vector products.
The two can be composed arbitrarily with one another, and with other JAX transformations.
For example, this can be used to efficiently compute full Hessian matrices:
```python
from jax import jacfwd, jacrev
def hessian(fun):
    return jit(jacfwd(jacrev(fun)))
```

### Auto-vectorization with `vmap()`
`vmap()` is the vectorizing map.
It has similar sematics of mapping a function along array axes, but instead of keeping the loop on the outside, it pushes the loop down into a function's primitive operations for better performance.
When composed with `jit()`, it can be as fast as adding the batch dimensions by hand.

## How to Think in JAX
### JAX vs. NumPy
**Key Concepts**:
* JAX provides a NumPy-inspired interface, `jax.numpy`
* Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
* Unlike NumPy arrays, JAX arrays are always immutable.

JAX arrays are immutable, and thus they cannot be changed after creation.
For updating individual elements, JAX provides an indexed update syntax that returns an updated copy: `y = x.at[0].set(10)`

### NumPy, lax & XLA: JAX API layering
**Key Concepts**:
* `jax.numpy` is a high-level wrapper that provides a familiar interface.
* `jax.lax` is a lower-level API that is stricter and often more powerful.
* All JAX operations are implemented in terms of operations in XLA - the Accelerated Linear Algebra compiler.

If you look at the source of `jax.numpy`, you'll see that all the operations are eventually expressed in terms o ffunctions defined in `jax.lax`.
You can think of `jax.lax` as a stricter, but often more powerful, API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitely promote arguments to allow operations between mixed data types, `jax.lax` will not.
Thus, if using `jax.lax` directly, you'll have to do type promotion.
Along with this strictness, `jax.lax` also provides efficient APIs for some more general operations than are supported in NumPy.

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA.
Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.

### To JIT or not to JIT
**Key Concepts**:
* By default JAX executes operations one at a time, in sequence.
* Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
* Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The fact that all JAX operations are expressed in terms of XLA allows JAX to use the XLA compiler to execute blocks of code very efficiently.

Some JAX operations are incompatible with JIT compilation.
For example `def get_negatives(x): return x[x < 0]` cannot be JIT compiled as the shape of the array is not known at compile time.

### JIT mechanics: tracing and static variables
**Key Concepts**:
* JIT and other JAX transforms work by tracing a function to determine its effect on inputs of a specifc shape and type.
* Variables that you don't want to be traced can be marked as static.

Tracer objects are what `jax.jit` uses to extract the sequence of operations specified by functions.
These encode `shape` and `dtype` of arrays, but are agnostic to the values.
The recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.
When we call teh compiled function again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python.

The extracted sequence of operations is encoded in a JAX expression, or jaxpr for short. 

### Static vs Traced Operations
**Key Concepts**:
* Just as values can be either static or traced, operations can be static or traced.
* Static operations are evaluated at compile-time in Python; traced operations are compiled and evaluated at run-time in XLA.
* Use `numpy` for operations that you want to be static; use `jax.numpy` for operations that you want to be traced.

A useful pattern is to use `numpy` for operations that should be static (i.e. done at compile-time), and use `jax.numpy` for operations that should be traced (i.e. compiled and executed at run-time). 



