# JAX - The Sharp Bits
JAX is a language for expressing and composing transformations of numerical programs.
JAX is also able to compile numerical programs for CPU, GPU or TPU.
JAX works great for many numerical and scientific programs, but only if they are written with certain constraints described below.

## Pure Functions
JAX is designed to work only on Python functions that are functionally pure:
* All input data is passed through the function parameters.
* All results are outputted through the function results.
A pure function will always return the same result if invoked with the same inputs.

A function can be pure even if it actually uses stateful objects internally, as long as it does not read or write external state. 
Iterators are not recommended if you want to use `jit` or in any control-flow primitive, as they induce state.

## In-Place Updates
Allowing mutation of variables in-place makes program analysis and transformation difficult.
JAX requires that programs are pure functions.
Instead, JAX offers a functional array update using the `.at` property on JAX arrays.

Inside `jit()` compiled code and `lax.while_loop` or `lax.fori_loop` the size of slices cannot be functions of argument values but only functions of argument shapes - teh slice start indices have no such restriction.

JAX's array update functions operate out-of-place.
That is, the updated array is returned as a new array and the original array is not modified by the update. 

However, inside jit-compiled code, if the input value `x` of `x.at[idx].set(y)`is not reused, the compiler will optimie the array update to occur in-place.

## Out-of-Bounds Indexing
When the indexing operation is an array index update, updates at out-of-bounds indices will be skipped.
When the operation is an array index retrieval the index is clamped to the out of bounds of the array since somehting must be returned.
For example, the last value of the array will be returned from this indexing operation:
```python
jnp.arange(10)[11]
# -> DeviceArray(9, dtype=int32)
```
Note that due to this behavior for index retrieval, functions like `jnp.nanargmin` and `jnp.nanargmax` return `-1` for slices consisting of NaNs.

Note also that, as the two behaviors described above are not inverses of each other, reverse-mode automatic differentiation (which turns index updates into retrievals and vice versa) will not preserve the semantics of out of bounds indexing.
Thus, it may be a good idea to think of out-of-bounds indexing in JAX as a case of undefined behavior. 

## Non-array inputs: NumPy vs JAX
JAX cannot take Python lists as inputs as it may lead to silent performance degradation that might otherwise be difficult to detect. 

In JAX's tracing and JIT compilation model, each element in a Python list or tuple is treated as a separate JAX variable, and individually processed and pushed to device.
Therefore, JAX avoids implicit conversion of lists and tuples to arrays.

If you would like to pass a tuple or list to a JAX function, you can do so by first explicitly converting it to an array: `jnp.sum(jnp.array(x))`.

## Random Numbers
JAX implements an explicit pseudorandom number generator (PRNG) where entropy production and consumption are handled by explicitly passing and iterating PRNG state. 
JAX uses a modern Threefry counter-based PRNG that is splittable.
That is, its design allows us to fork the PRNG state into new PRNGs for use with parallel stochastic generation.

The random state is described by two unsigned-int32s that we call a key:
```python
from jax import random
key = random.PRNGKey(0)
```
JAX's random functions produce pseudorandom numbers from the PRNG state, but do not change the state.
Instead of reusing the key, we split the PRNG to get usuable subkeys every time we need a new pseudorandom number:
```python
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
```
We probagate the key and make new subkeys wheneer we need a new random number.
We can generate more than one subkey at a time:
```python
key, *subkeys = random.split(key, 4)
for subkey in subkeys:
    print(random.normal(subkey, shape=(1,)))

#-> [-0.37533438]
#-> [0.98645043]
#-> [0.14553197]
```

## Control Flow
If you just want to apply `grad` to your python functions, you can use regular control-flow constructs with no problems, as if you were using Autograd (or PyTorch or TF Eager).
```python
from jax import grad
def f(x):
    if x < 3:
        return 3. * x ** 2
    else:
        return -4. * x
print(grad(f)(2.))
print(grad(f)(4.))
#-> 12.0
#-> -4.0
```

Using control flow with `jit()` is more complicated, and by default it has more constraints.
When we `jit`-compile a function, we usually want to compile a version of the function that works for many different argument values, so that we can cache and reuse the compiled code.

To get a view of your Python code that is valid for many different argument values, JAX traces it on abstract values that represent sets of possible inputs.
There are multiple different levels of abstraction, and different transformations use different abstraction levels.

By default, `jit` traces your code on the `ShapedArray` abstraction level, where each abstract value represents the set of all array values with a fixed shape and dtype.

There is a tradeoff however: if we trace a python function on `ShapedArray((), jnp.float32)` that isn't committed to a specific concrete value, when we hit a line like `if x < 3`, the expression `x < 3` evaluates to an abstract `ShapedArray((), jnp.bool_)` that represents the set `{True, False}`.
When Python attempts to coerce that to a concrete `True` or `False`, we get an error: we don't know what branch to take, and cannot continue tracing.
The tradeoff is that with higher levels of abstraction we gain a more general view of the Python code (and thus save on re-compilations), but we require more constraints on the Python code to complete the trace

By having `jit` trace on more refined abstract values, you can relax traceability constraints.

If your function has global side-effects, JAX's tracer can cause weird things to happen.

### Structured Control Flow Primitives
There are more options for control flow in JAX.
Say you want to avoid re-compilations but still want to use control flow that's traceable, and that avoids un-rolling large loops.
Then you can use these 4 structured control flow primitives:
* `lax.cond` differentiable.
* `lax.while_loop` fwd-mode-differentiable.
* `lax.fori_loop` fwd-mode-differentiable in general; fwd and rev-mode differentiable if endpoints are static.
* `lax.scan` differentiable.

#### `cond`
Python equivalent:
```python
# In Python
def cond(pred, true_fun, false_fun, operand):
    if pred:
        return true_fun(operand)
    else:
        return false_fun(operand)

# In JAX
from jax import lax
operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x - 1, operand)
# --> array([1.], dtype=float32)
```

#### `while_loop`
```python
# Native Python
def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

# JAX
init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x + 1
lax.while_loop(cond_fun, body_fun, init_val)
# --> array(10, dtype=int32)
```

#### `fori_loop`
```python
# Native Python
def fori_loop(start, stop, body_fun, init_val):
    val = init_val
    for i in range(start, stop):
        val = body_fun(i, val)
    return val

# JAX
init_val = 0
start = 0
stop = 10
body_fun = lambda i, x: x + i
lax.fori_loop(start, stop, body_fun, init_val)
# --> array(45, dtype=int32)
```


## NaNs
If you want to trace where NaNs are occuring in your functions or gradients, you can turn on the NaN-checker by:
* setting the `JAX_DEBUG_NANS=True` environment variable.
* adding `from jax.config import config` and `config.update("jax_debug_nans", True)` near the top of your main file.
* adding `from jax.config import config` and `config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`

This will cause computations to error-out immediately on production of a NaN.
switching this option on adds a nan check to every floating point type produced by XLA.
That means that values are pulled back to the host and checked as ndarrays for every primitive operation not under a `@jit`. 
For code under a `@jit`, the output of every `@jit` function is checked and if a nan is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jit` at a time.

You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lost of device-host round-trips and performance regressions.
The NaN-checker doesn't work with `pmap`.
To debug nans in `pmap` code, on thing to try is replacing `pmap` with `vmap`.

## Double (64bit) Precision
To use double-precision numbers, you need to set the `jax_enable_x64` configuration variable at startup.


## Miscellaneous Divergences from NumPy
